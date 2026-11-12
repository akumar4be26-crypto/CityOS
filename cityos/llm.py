"""
LLM reasoning layer – provider-agnostic (Anthropic, OpenAI, any OpenAI-compatible endpoint such as a local server).

The LLM is used for reasoning, planning, tool selection, prioritisation and explanation. It never produces
numbers itself: every quantitative statement must come from a tool result, and its decisions are validated
against the candidate actions the domain agents generated. Without an API key the platform falls back to a
deterministic reasoner and says so.
"""
from __future__ import annotations

import json
import ssl
import time
import urllib.error
import urllib.request

from .config import settings

TOOL_SPECS = [
    ("get_traffic_status", "Current traffic KPIs, congested roads, closures and collisions.", {}),
    ("predict_congestion", "ML forecast (15/30/60 min) of volume/capacity and congestion probability for a road.", {"road_id": "string"}),
    ("detect_safety_incident", "Open incidents, zone crowd levels, crowding risk and emergency coverage.", {}),
    ("get_bin_status", "Smart-bin fill levels, optionally for one zone (Z1..Z6).", {"zone": "string"}),
    ("predict_waste_overflow", "ML fill trajectory and overflow time for a bin.", {"bin_id": "string"}),
    ("get_energy_status", "Grid import, demand, solar, battery, EV load and anomalies.", {}),
    ("predict_energy_demand", "ML 24-hour demand and solar forecast with expected peak, cost and carbon.", {}),
    ("detect_infrastructure_failure", "Failed and at-risk assets with ML failure probability and remaining useful life.", {}),
    ("predict_maintenance", "Maintenance priority queue (Risk × Failure probability × Impact × Criticality).", {}),
    ("optimize_route", "Alternative paths around a road (spare-capacity weighted) or an emergency route between two nodes.",
     {"road_id": "string", "from_node": "string", "to_node": "string"}),
    ("optimize_signal_timing", "Webster-optimal signal plan for an intersection with HCM delay before/after.", {"node_id": "string"}),
    ("optimize_resources", "Availability and positions of emergency units, maintenance crews and collection trucks.", {}),
    ("calculate_cost", "Operational cost totals (traffic delay, fleet, waste collection, energy).", {}),
    ("calculate_carbon", "CO2e totals (traffic idling, fleet, landfill, electricity).", {}),
    ("simulate_action", "Counterfactual digital-twin simulation of a candidate action (by proposal_id).", {"proposal_id": "string"}),
    ("verify_action", "Verification status of an executed action (by action_id).", {"action_id": "string"}),
]

SYSTEM_PROMPT = """You are the reasoning layer of CityOS, an autonomous city-operations agent for the city of {city}.
Domain agents (Traffic, Safety, Waste, Energy, Infrastructure) have already processed sensor data with ML models and
optimisers and produced prioritised issues with candidate actions. Your job: reason over them, call tools when you need
more evidence, decide which candidate action (if any) to take for each issue, order the issues, and explain briefly.

Hard rules:
- Never invent sensor values, ML metrics, predictions, tool results, vehicle locations or action results. Quote numbers
  only if they appear in the briefing or in a tool result.
- You may only choose proposal_ids that appear in the briefing. Choosing none ("defer") is allowed.
- All actions execute in a SIMULATED digital twin unless stated otherwise; never claim a physical action happened.
- Actions marked HIGH risk go to a human for authorisation; you cannot bypass the safety layer.
- Prefer actions that protect life safety first, then service continuity, then cost and carbon.

Money is in {currency}.

Respond at the end with ONLY a JSON object:
{{"decisions":[{{"issue_key":"...","proposal_id":"... or null","rationale":"<= 40 words"}}],
 "priority_order":["issue_key", ...], "summary":"<= 60 words operational summary"}}""".format(city=settings.city_name, currency=settings.currency)


class LLMClient:
    def __init__(self, tool_executor=None):
        self.provider = settings.llm_provider
        self.model = settings.llm_model or {"anthropic": "claude-sonnet-5-5", "openai": "gpt-4.1-mini"}.get(self.provider, "")
        self.key = settings.llm_api_key
        self.base = settings.llm_base_url.rstrip("/")
        self.exec_tool = tool_executor
        self.last_error: str | None = None
        self.calls = 0
        self.last_call_ts = 0.0
        self.ctx = ssl.create_default_context()

    @property
    def enabled(self) -> bool:
        if self.provider in ("anthropic", "openai"):
            return bool(self.key)
        return self.provider == "openai_compatible" and bool(self.base)

    def status(self) -> dict:
        return dict(provider=self.provider if self.enabled else "none", model=self.model if self.enabled else None, enabled=self.enabled,
                    calls=self.calls, last_error=self.last_error,
                    mode="LLM reasoning" if self.enabled else "Deterministic reasoner (no LLM configured — set LLM_PROVIDER / LLM_API_KEY in .env)")

    # ---------------------------------------------------------------- transport
    def _post(self, url: str, body: dict, headers: dict) -> dict:
        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"content-type": "application/json", **headers})
        with urllib.request.urlopen(req, timeout=settings.llm_timeout_s, context=self.ctx) as r:
            return json.loads(r.read().decode())

    # ---------------------------------------------------------------- tool loop
    def run(self, user_content: str, system: str = SYSTEM_PROMPT, use_tools: bool = True) -> tuple[str, list[dict]]:
        """Returns (final_text, tool_trace)."""
        self.calls += 1
        self.last_call_ts = time.time()
        trace: list[dict] = []
        try:
            if self.provider == "anthropic":
                return self._anthropic(user_content, system, use_tools, trace), trace
            return self._openai(user_content, system, use_tools, trace), trace
        except urllib.error.HTTPError as e:
            self.last_error = f"HTTP {e.code}: {e.read().decode()[:300]}"
        except Exception as e:  # network, parse …
            self.last_error = f"{type(e).__name__}: {e}"
        raise RuntimeError(self.last_error)

    def _schema(self, params: dict) -> dict:
        return {"type": "object", "properties": {k: {"type": v} for k, v in params.items()}, "required": []}

    def _call_tool(self, name: str, args: dict, trace: list) -> str:
        try:
            res = self.exec_tool(name, args or {}) if self.exec_tool else {"error": "no executor"}
        except Exception as e:
            res = {"error": f"{type(e).__name__}: {e}"}
        trace.append(dict(tool=name, args=args, result_preview=json.dumps(res, default=str)[:400]))
        return json.dumps(res, default=str)[:12000]

    def _anthropic(self, user, system, use_tools, trace) -> str:
        url = (self.base or "https://api.anthropic.com") + "/v1/messages"
        headers = {"x-api-key": self.key, "anthropic-version": "2023-06-01"}
        tools = [dict(name=n, description=d, input_schema=self._schema(p)) for n, d, p in TOOL_SPECS]
        msgs = [{"role": "user", "content": user}]
        for _ in range(settings.llm_max_tool_rounds + 1):
            body = dict(model=self.model, max_tokens=1800, system=system, messages=msgs)
            if use_tools:
                body["tools"] = tools
            r = self._post(url, body, headers)
            blocks = r.get("content", [])
            uses = [b for b in blocks if b.get("type") == "tool_use"]
            if not uses or r.get("stop_reason") != "tool_use":
                self.last_error = None
                return "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
            msgs.append({"role": "assistant", "content": blocks})
            msgs.append({"role": "user", "content": [dict(type="tool_result", tool_use_id=u["id"], content=self._call_tool(u["name"], u.get("input"), trace))
                                                     for u in uses]})
        return ""

    def _openai(self, user, system, use_tools, trace) -> str:
        url = (self.base or "https://api.openai.com") + "/v1/chat/completions"
        headers = {"authorization": f"Bearer {self.key}"} if self.key else {}
        tools = [dict(type="function", function=dict(name=n, description=d, parameters=self._schema(p))) for n, d, p in TOOL_SPECS]
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        for _ in range(settings.llm_max_tool_rounds + 1):
            body = dict(model=self.model, messages=msgs, temperature=0.2)
            if use_tools:
                body["tools"] = tools
            r = self._post(url, body, headers)
            m = r["choices"][0]["message"]
            calls = m.get("tool_calls") or []
            if not calls:
                self.last_error = None
                return m.get("content") or ""
            msgs.append(m)
            for tc in calls:
                try:
                    args = json.loads(tc["function"].get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": self._call_tool(tc["function"]["name"], args, trace)})
        return ""


def extract_json(text: str) -> dict | None:
    if not text:
        return None
    s, e = text.find("{"), text.rfind("}")
    if s < 0 or e <= s:
        return None
    try:
        return json.loads(text[s:e + 1])
    except json.JSONDecodeError:
        return None
