"""
Mini Sentinel Pipeline – Flask Demo Server
==========================================
Run: venv/bin/python app.py
Then open http://127.0.0.1:5000 in your browser.
"""

import json
import os
import sys
import time
from typing import Optional

from dotenv import load_dotenv
from flask import Flask, Response, render_template, stream_with_context
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

load_dotenv()

app = Flask(__name__)

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openrouter").lower()

# ─────────────────────────────────────────────────────────────────────────────
# Shared pipeline components (same as main.py)
# ─────────────────────────────────────────────────────────────────────────────
MOCK_SHODAN_PAYLOAD: dict = {
    "target_domain": "logistics-frankfurt.de",
    "scanned_port": 3389,
    "service": "RDP",
    "vulnerability_detected": "CVE-2019-0708 (BlueKeep)",
    "cvss_score": 9.8,
}

OPENROUTER_FREE_MODELS = [
    "mistralai/mistral-small-3.1-24b-instruct:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "nousresearch/hermes-3-llama-3.1-405b:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "qwen/qwen3-coder:free",
]


class AlertOutput(BaseModel):
    domain: str = Field(description="The target domain from the vulnerability scan.")
    is_critical: bool = Field(description="True if the CVSS score is above 8.0, otherwise False.")
    recommended_action: str = Field(description="A concise remediation action. Maximum 15 words.")


class AgentState(TypedDict):
    raw_data: dict
    alert: Optional[AlertOutput]


def _build_openrouter_llm(model_id: str):
    from langchain_openai import ChatOpenAI
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


# ─────────────────────────────────────────────────────────────────────────────
# SSE streaming pipeline runner
# ─────────────────────────────────────────────────────────────────────────────
def _sse(event: str, data: dict) -> str:
    """Format a Server-Sent Event."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def run_pipeline_stream():
    """Generator that yields SSE events as the pipeline executes."""

    # ── Step 1: Node 1 – Data Ingestion ──────────────────────────────────
    yield _sse("node1_start", {"message": "Node 1 activated: reading mock Shodan payload…"})
    time.sleep(0.4)
    yield _sse("node1_data", {"payload": MOCK_SHODAN_PAYLOAD})
    time.sleep(0.6)
    yield _sse("node1_done", {"message": "Payload validated and loaded into AgentState ✓"})
    time.sleep(0.5)

    # ── Step 2: Node 2 – Auditor LLM ─────────────────────────────────────
    yield _sse("node2_start", {"message": "Node 2 activated: building Pydantic parser + prompt…"})
    time.sleep(0.5)

    parser = PydanticOutputParser(pydantic_object=AlertOutput)
    format_instructions = parser.get_format_instructions()

    prompt = ChatPromptTemplate.from_messages([
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
    ])

    yield _sse("node2_calling", {"message": "Sending request to LLM via OpenRouter waterfall…"})

    # Waterfall model loop
    alert: Optional[AlertOutput] = None
    tried_models = []

    for model_id in OPENROUTER_FREE_MODELS:
        llm = _build_openrouter_llm(model_id)
        chain = prompt | llm | parser
        yield _sse("model_try", {"model": model_id, "status": "trying"})
        try:
            alert = chain.invoke({
                "format_instructions": format_instructions,
                **MOCK_SHODAN_PAYLOAD,
            })
            tried_models.append({"model": model_id, "status": "success"})
            yield _sse("model_try", {"model": model_id, "status": "success"})
            break
        except Exception as e:
            err = str(e)
            if "429" in err or "rate" in err.lower() or "404" in err:
                tried_models.append({"model": model_id, "status": "rate_limited"})
                yield _sse("model_try", {"model": model_id, "status": "rate_limited"})
                time.sleep(1)
                continue
            yield _sse("error", {"message": f"Unexpected error: {err[:200]}"})
            return

    if alert is None:
        yield _sse("error", {"message": "All free models are currently rate-limited. Please retry in 1–2 minutes."})
        return

    # ── Step 3: Output ────────────────────────────────────────────────────
    alert_dict = alert.model_dump()
    word_count = len(alert_dict["recommended_action"].split())
    cvss = MOCK_SHODAN_PAYLOAD["cvss_score"]

    yield _sse("result", {
        "alert": alert_dict,
        "validation": {
            "is_critical_correct": alert_dict["is_critical"] == (cvss > 8.0),
            "word_count": word_count,
            "word_count_ok": word_count <= 15,
            "pydantic_enforced": True,
            "cvss": cvss,
        },
    })


# ─────────────────────────────────────────────────────────────────────────────
# Flask routes
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", payload=MOCK_SHODAN_PAYLOAD)


@app.route("/run")
def run():
    def generate():
        yield _sse("start", {"message": "Pipeline starting…"})
        yield from run_pipeline_stream()
        yield _sse("done", {"message": "Pipeline complete."})

    return Response(
        stream_with_context(generate()),
        content_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    # Preflight check
    if not os.getenv("OPENROUTER_API_KEY") and LLM_PROVIDER == "openrouter":
        sys.exit("ERROR: OPENROUTER_API_KEY is not set in .env")
    print("\n🚀  Sentinel Demo running at → http://127.0.0.1:5000\n")
    app.run(debug=False, threaded=True, port=5000)
