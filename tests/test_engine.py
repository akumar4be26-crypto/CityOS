"""Headless checks of the closed loop (no server needed):  python -m pytest -q tests   or   python tests/test_engine.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cityos.orchestrator import Orchestrator  # noqa: E402


def test_closed_loop_runs_and_verifies():
    o = Orchestrator()
    for _ in range(150):
        o.cycle()
    phases = {e["phase"] for e in o.activity}
    assert {"DETECT", "DECIDE", "ACT", "VERIFY"} <= phases
    verified = [a for a in o.actions.values() if (a.get("verification") or {}).get("result") in ("SUCCESS", "PARTIAL", "FAILED")]
    assert verified, "expected at least one verified action"
    assert all(a["execution_mode"].startswith("SIMULATED") for a in o.actions.values())
    assert o.health_hist[-1]["score"] > 0


def test_high_impact_actions_need_authorisation():
    o = Orchestrator()
    for _ in range(20):
        o.cycle()
    sc = o.safety_check(dict(type="road_closure", params=dict(road="R28"), target="R28"))
    assert sc["requires_authorization"] and sc["risk"] == "HIGH"


def test_cross_domain_closure_creates_plan():
    o = Orchestrator()
    for _ in range(10):
        o.cycle()
    o.inject("road_closure")
    for _ in range(4):
        o.cycle()
    assert o.plans, "closure should produce a coordinated plan"


def test_whatif_simulation():
    o = Orchestrator()
    o.cycle()
    r = o.run_scenario("population_increase", hours=6)
    assert r["comparison"]["energy_kwh"]["scenario"] > r["comparison"]["energy_kwh"]["baseline"]


def test_llm_validation_rejects_unknown_proposals():
    o = Orchestrator()
    for _ in range(30):
        o.cycle()
    k = next(iter(o.issues))
    n = len(o.actions)
    o._apply_llm(dict(json={"decisions": [{"issue_key": k, "proposal_id": "FAKE#1", "rationale": "x"}]}, trace=[], issues=[k]))
    assert len(o.actions) == n


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
