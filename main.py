"""
Mini Sentinel Pipeline
======================
Technical Assessment – "CEO Alert" Node
Author  : [Candidate Name]
Version : 1.1.0

LLM Providers supported (set LLM_PROVIDER in .env):
  openrouter  →  uses OpenRouter API (default, free Gemini models)
  openai      →  direct OpenAI API
  anthropic   →  direct Anthropic API

Architecture
------------
  START
    │
    ▼
┌─────────────────────┐
│  Node 1             │
│  data_ingestion     │  ← Reads mock Shodan payload → AgentState
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  Node 2             │
│  auditor_llm        │  ← LLM + PydanticOutputParser → AlertOutput
└─────────────────────┘
    │
    ▼
   END

Output is enforced via Pydantic schema – no raw LLM text ever reaches stdout.
"""

import json
import os
import sys
from typing import Optional

from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

# ─────────────────────────────────────────────────────────────────────────────
# 0. Environment
# ─────────────────────────────────────────────────────────────────────────────
load_dotenv()  # loads .env if present

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openrouter").lower()

# ─────────────────────────────────────────────────────────────────────────────
# 1. Mock Input Data  (simulates a Shodan API response)
# ─────────────────────────────────────────────────────────────────────────────
MOCK_SHODAN_PAYLOAD: dict = {
    "target_domain": "logistics-frankfurt.de",
    "scanned_port": 3389,
    "service": "RDP",
    "vulnerability_detected": "CVE-2019-0708 (BlueKeep)",
    "cvss_score": 9.8,
}

# ─────────────────────────────────────────────────────────────────────────────
# 2. Pydantic Output Schema  (the "guardrail")
# ─────────────────────────────────────────────────────────────────────────────
class AlertOutput(BaseModel):
    """Strictly enforced JSON structure for every CEO Alert."""

    domain: str = Field(description="The target domain from the vulnerability scan.")
    is_critical: bool = Field(
        description="True if the CVSS score is above 8.0, otherwise False."
    )
    recommended_action: str = Field(
        description="A concise remediation action. Maximum 15 words."
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. AgentState  (shared across all graph nodes)
# ─────────────────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    raw_data: dict                   # populated by Node 1
    alert: Optional[AlertOutput]     # populated by Node 2


# ─────────────────────────────────────────────────────────────────────────────
# 4. LLM initialisation  (OpenRouter default, OpenAI/Anthropic optional)
# ─────────────────────────────────────────────────────────────────────────────

# Priority list of free OpenRouter models – tried in order until one succeeds.
# All route through OpenRouter's OpenAI-compatible endpoint.
OPENROUTER_FREE_MODELS = [
    "mistralai/mistral-small-3.1-24b-instruct:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "nousresearch/hermes-3-llama-3.1-405b:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "qwen/qwen3-coder:free",
]


def _build_openrouter_llm(model_id: str):
    """Return a ChatOpenAI instance pointed at OpenRouter for the given model."""
    from langchain_openai import ChatOpenAI  # type: ignore
    return ChatOpenAI(
        model=model_id,
        temperature=0,
        openai_api_key=os.environ["OPENROUTER_API_KEY"],
        openai_api_base="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "https://github.com/sentinel-pipeline",
            "X-Title": "Mini Sentinel Pipeline",
        },
    )


def _build_llm():
    if LLM_PROVIDER == "anthropic":
        from langchain_anthropic import ChatAnthropic  # type: ignore
        return ChatAnthropic(
            model="claude-3-5-sonnet-20241022",
            temperature=0,
            api_key=os.environ["ANTHROPIC_API_KEY"],
        )
    elif LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI  # type: ignore
        return ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            api_key=os.environ["OPENAI_API_KEY"],
        )
    else:
        # For single-model mode: respect OPENROUTER_MODEL env var if set,
        # otherwise return None to signal waterfall mode.
        single = os.getenv("OPENROUTER_MODEL")
        if single:
            return _build_openrouter_llm(single)
        return None  # waterfall mode


# ─────────────────────────────────────────────────────────────────────────────
# 5. Graph Nodes
# ─────────────────────────────────────────────────────────────────────────────

# ── Node 1: Data Ingestion ───────────────────────────────────────────────────
def data_ingestion_node(state: AgentState) -> AgentState:
    """
    Reads the mock Shodan payload and writes it into the shared AgentState.
    In a production system this node would call the live Shodan API.
    """
    print("\n[Node 1 – Data Ingestion]")
    print("  Payload received:")
    for key, value in MOCK_SHODAN_PAYLOAD.items():
        print(f"    {key}: {value}")

    return {**state, "raw_data": MOCK_SHODAN_PAYLOAD, "alert": None}


# ── Node 2: Auditor LLM ─────────────────────────────────────────────────────
def auditor_llm_node(state: AgentState) -> AgentState:
    """
    Passes the vulnerability data to the LLM and enforces structured output
    via PydanticOutputParser – the model CANNOT return free-form text.
    """
    print("\n[Node 2 – Auditor LLM]")

    data = state["raw_data"]
    llm = _build_llm()

    # Output parser wired to our Pydantic schema
    parser = PydanticOutputParser(pydantic_object=AlertOutput)
    format_instructions = parser.get_format_instructions()

    # Prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a cybersecurity analyst producing structured CEO alerts. "
                    "Evaluate the vulnerability data provided and adhere strictly to "
                    "the output format. The 'recommended_action' must be 15 words or fewer.\n\n"
                    "{format_instructions}"
                ),
            ),
            (
                "human",
                (
                    "Vulnerability scan results:\n"
                    "  Domain        : {target_domain}\n"
                    "  Port          : {scanned_port}\n"
                    "  Service       : {service}\n"
                    "  Vulnerability : {vulnerability_detected}\n"
                    "  CVSS Score    : {cvss_score}\n\n"
                    "Rules:\n"
                    "  - Set is_critical = true  if CVSS score > 8.0\n"
                    "  - Set is_critical = false if CVSS score ≤ 8.0\n"
                    "  - recommended_action must be ≤ 15 words\n\n"
                    "Generate the CEO alert now."
                ),
            ),
        ]
    )

    print(f"  Provider       : {LLM_PROVIDER.upper()}")

    import time

    # ── Determine which models to try ───────────────────────────────────────
    llm_single = _build_llm()   # None = waterfall mode (OpenRouter)
    if llm_single is not None:
        # Non-OpenRouter or single-model override
        model_candidates = [(os.getenv("OPENROUTER_MODEL", LLM_PROVIDER), llm_single)]
    else:
        # Waterfall: try each free model in priority order
        model_candidates = [
            (m, _build_openrouter_llm(m)) for m in OPENROUTER_FREE_MODELS
        ]

    alert: Optional[AlertOutput] = None
    for model_id, llm in model_candidates:
        chain = prompt | llm | parser
        print(f"  Trying model   : {model_id}")
        try:
            alert = chain.invoke(
                {
                    "format_instructions": format_instructions,
                    **data,
                }
            )
            print(f"  ✅ Success with : {model_id}")
            break
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "rate" in err_str.lower() or "404" in err_str:
                print(f"  ⚠  Skipping {model_id}: {err_str[:80]}")
                time.sleep(2)   # brief pause before next model
                continue
            raise  # unexpected error – bubble up

    if alert is None:
        raise RuntimeError("All OpenRouter free models are rate-limited. Try again in a few minutes.")

    return {**state, "alert": alert}


# ─────────────────────────────────────────────────────────────────────────────
# 6. Build the LangGraph
# ─────────────────────────────────────────────────────────────────────────────
def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("ingest", data_ingestion_node)
    graph.add_node("audit",  auditor_llm_node)

    graph.add_edge(START,    "ingest")
    graph.add_edge("ingest", "audit")
    graph.add_edge("audit",  END)

    return graph.compile()


# ─────────────────────────────────────────────────────────────────────────────
# 7. Entry Point
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 60)
    print("  Mini Sentinel Pipeline – CEO Alert Generator")
    print("=" * 60)

    # Preflight: check for required API key
    if LLM_PROVIDER == "anthropic" and not os.getenv("ANTHROPIC_API_KEY"):
        sys.exit("ERROR: ANTHROPIC_API_KEY is not set. Add it to your .env file.")
    elif LLM_PROVIDER == "openai" and not os.getenv("OPENAI_API_KEY"):
        sys.exit("ERROR: OPENAI_API_KEY is not set. Add it to your .env file.")
    elif LLM_PROVIDER not in ("openai", "anthropic") and not os.getenv("OPENROUTER_API_KEY"):
        sys.exit("ERROR: OPENROUTER_API_KEY is not set. Add it to your .env file.")

    pipeline = build_graph()

    # Run the graph with an empty initial state
    initial_state: AgentState = {"raw_data": {}, "alert": None}
    final_state: AgentState = pipeline.invoke(initial_state)

    # ── Output ──────────────────────────────────────────────────────────────
    alert = final_state["alert"]
    if alert is None:
        sys.exit("ERROR: Pipeline completed but no alert was generated.")

    alert_dict = alert.model_dump()

    print("\n" + "=" * 60)
    print("  ✅  CEO ALERT  –  STRUCTURED OUTPUT (Pydantic-enforced JSON)")
    print("=" * 60)
    print(json.dumps(alert_dict, indent=2))
    print("=" * 60)

    # Post-run validation summary
    word_count = len(alert_dict["recommended_action"].split())
    print("\n📋  Validation Summary")
    print(f"   domain             : {alert_dict['domain']}")
    print(f"   is_critical        : {alert_dict['is_critical']}  "
          f"(CVSS {MOCK_SHODAN_PAYLOAD['cvss_score']} {'> 8.0 ✓' if MOCK_SHODAN_PAYLOAD['cvss_score'] > 8.0 else '≤ 8.0 ✓'})")
    print(f"   recommended_action : {word_count} words  "
          f"({'≤ 15 ✓' if word_count <= 15 else '⚠ EXCEEDS 15 WORDS'})")
    print(f"   output type        : {type(alert).__name__} (Pydantic model) ✓\n")


if __name__ == "__main__":
    main()
