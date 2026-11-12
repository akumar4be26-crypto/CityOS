"""
City Orchestrator – the closed loop:

 OBSERVE → UNDERSTAND → ANALYZE → DETECT → INVESTIGATE → PREDICT → PLAN → DECIDE → ACT → VERIFY → ADAPT → REPEAT

 * Runs the domain agents, collects their findings and cross-domain messages
 * Priority engine:  Priority = Severity × Probability × Impact × Urgency  (configurable thresholds)
 * Decision engine:  deterministic rules + (optional) LLM reasoning with tool calls, validated against candidates
 * Safety layer:     AI decision → safety validation → authorisation (HIGH impact) → action
 * Action engine:    executes in the digital twin (SIMULATED); no physical system is touched
 * Verification:     before / after / counterfactual (no-action twin) comparison → SUCCESS / PARTIAL / FAILED
 * Adaptation:       action effectiveness, detection thresholds, monitoring frequency, ML bias + retraining
"""
from __future__ import annotations

import json
import math
import threading
import time
from collections import defaultdict, deque

import numpy as np
import pandas as pd

from . import optimize as opt
from .agents import (LEVELS, Bus, Ctx, EnergyAgent, InfrastructureAgent, SafetyAgent, TrafficAgent, WasteAgent)
from .config import settings
from .history import generate_history, load_history
from .llm import LLMClient, extract_json
from .ml import CityML
from .processing import Processor
from .store import Store
from .twin import CityTwin

PHASES = ["OBSERVE", "UNDERSTAND", "ANALYZE", "DETECT", "INVESTIGATE", "PREDICT", "PLAN", "DECIDE", "ACT", "VERIFY", "ADAPT"]
EXECUTION_MODE = "SIMULATED — digital twin (no external system connected)"

# Safety policy: action type → (risk, requires human authorisation, rationale)
POLICY = {
    "optimize_signal_timing": ("LOW", False, "bounded signal-plan change (splits 0.22–0.78, cycle 50–140 s)"),
    "reroute_traffic": ("MEDIUM", False, "advisory route recommendation (VMS / navigation feeds)"),
    "traffic_diversion": ("MEDIUM", False, "advisory diversion around an existing closure"),
    "emergency_dispatch": ("HIGH", True, "emergency system — dispatch & signal pre-emption need a human dispatcher"),
    "emergency_preposition": ("HIGH", True, "moves an emergency unit — needs a human dispatcher"),
    "patrol_dispatch": ("HIGH", True, "public-safety resource deployment"),
    "crowd_management": ("MEDIUM", False, "field-staff deployment advisory"),
    "safety_alert": ("LOW", False, "information only"),
    "incident_escalation": ("MEDIUM", False, "notification to supervisors"),
    "dispatch_collection": ("LOW", False, "routine fleet routing"),
    "reoptimize_collection": ("LOW", False, "routine fleet routing"),
    "segregation_advisory": ("LOW", False, "public information"),
    "inspection_request": ("LOW", False, "creates an inspection work order"),
    "battery_dispatch": ("MEDIUM", False, "battery set-points within inverter and SoC limits"),
    "ev_charging_shift": ("MEDIUM", False, "flexible-load shifting within charging SLAs"),
    "hvac_adjustment": ("LOW", False, "set-point change within comfort band (≤ +2 °C, not clinical spaces)"),
    "lighting_schedule": ("LOW", False, "restores lighting schedule"),
    "shutdown_idle_equipment": ("MEDIUM", False, "BMS command to non-critical plug loads"),
    "maintenance_request": ("MEDIUM", False, "creates a maintenance work order and dispatches a crew"),
    "road_closure": ("HIGH", True, "critical infrastructure / public safety — needs human authorisation"),
}

DETECTION_MAP = {   # finding kind → ground-truth kinds it should catch (for precision / recall)
    "abnormal_activity": ("abnormal_activity",), "illegal_dumping": ("illegal_dumping",), "poor_segregation": ("poor_segregation",),
    "hvac_inefficiency": ("hvac_fault", "unexpected_consumption", "excessive_lighting"),
    "excessive_lighting": ("excessive_lighting", "hvac_fault", "unexpected_consumption"),
    "unexpected_consumption": ("unexpected_consumption", "hvac_fault", "excessive_lighting"),
    "failure_risk": ("infra_fault",), "asset_failure": ("infra_fault",), "water_leakage": ("infra_fault",),
    "streetlight_failure": ("infra_fault",), "infrastructure_anomaly": ("infra_fault",), "sensor": ("sensor_failure",),
}
DETECTION_GROUP = {"abnormal_activity": "safety", "illegal_dumping": "waste", "poor_segregation": "waste", "hvac_fault": "energy",
                   "unexpected_consumption": "energy", "excessive_lighting": "energy", "infra_fault": "infrastructure", "sensor_failure": "sensor"}


class Orchestrator:
    def __init__(self, train_if_missing: bool = True, seed: int | None = None):
        self.lock = threading.RLock()
        self.boot_log: list[str] = []
        self.ml = CityML()
        if not self.ml.load():
            if not train_if_missing:
                raise RuntimeError("models missing")
            self._boot("Generating synthetic history (%d days) …" % settings.history_days)
            h = load_history() or generate_history()
            self._boot("Training ML models …")
            self.ml.train(h)
        self._boot(f"Models ready (v{self.ml.version}, {self.ml.metrics.get('engine')})")
        self.tw = CityTwin(seed=seed)
        self.proc = Processor()
        self.store = Store()
        self.bus = Bus()
        self.agents = [InfrastructureAgent(), TrafficAgent(), SafetyAgent(), WasteAgent(), EnergyAgent()]
        self.agent_by_domain = {a.domain: a for a in self.agents}
        self.memory = {a.domain: {} for a in self.agents}
        self.llm = LLMClient(tool_executor=self.call_tool)
        self.thresholds = dict(congestion_vc=settings.congestion_vc, z=settings.z_threshold, overflow_fill=settings.overflow_fill,
                               peak_kw=settings.peak_threshold_kw or self.ml.baselines.get("peak_kw_p80", 6500) * 0.98,
                               failure_prob=settings.failure_prob_threshold)
        self.reports: dict[str, dict] = {}
        self.issues: dict[str, dict] = {}
        self.resolved: deque = deque(maxlen=150)
        self.actions: dict[str, dict] = {}
        self.action_order: deque = deque(maxlen=400)
        self.plans: deque = deque(maxlen=30)
        self.activity: deque = deque(maxlen=400)
        self.health_hist: deque = deque(maxlen=600)
        self.state_engine: dict = {}
        self.effectiveness: dict = defaultdict(lambda: dict(success=0, partial=0, failed=0, total=0))
        self.det = defaultdict(lambda: dict(tp=0, fp=0, fn=0))
        self.truth_seen: set = set()
        self.truth_detected: set = set()
        self.llm_pending: dict | None = None
        self.llm_result: dict | None = None
        self.llm_summary: str = ""
        self.llm_busy = False
        self.monitor = defaultdict(lambda: 1)
        self.live_rows = dict(traffic=[], energy=[])
        self.counter = 0
        self.last_cycle_ms = 0.0
        self.running = True
        self.speed = 1.0
        self._prev_obs = {}
        self.events: deque = deque(maxlen=200)
        self._seen_log = 0

    def _boot(self, msg: str):
        self.boot_log.append(msg)
        print(f"[CityOS] {msg}", flush=True)

    # =============================================================== logging
    def log(self, phase: str, agent: str, message: str, ref: str = "", domain: str | None = None):
        e = dict(ts=self.tw.t.isoformat(timespec="minutes"), tick=self.tw.tick, phase=phase, agent=agent, message=message, ref=ref,
                 domain=domain or {"Traffic Agent": "traffic", "Safety Agent": "safety", "Waste Agent": "waste", "Energy Agent": "energy",
                                   "Infrastructure Agent": "infrastructure"}.get(agent, "city"))
        self.activity.append(e)
        self.store.log(e)

    def _nid(self, p: str) -> str:
        self.counter += 1
        return f"{p}-{self.counter:05d}"

    # =============================================================== main loop
    def cycle(self) -> dict:
        with self.lock:
            t0 = time.perf_counter()
            tw = self.tw
            # ---------------- OBSERVE
            raw = tw.step()
            processed = self.proc.process(raw)
            if tw.tick % settings.persist_sensor_every == 0:
                self.store.sensors(processed["records"])
            self._observe_log()
            self._record_live()
            # physical events from the twin → event stream
            for ev in tw.log[self._seen_log:] if self._seen_log <= len(tw.log) else tw.log:
                self.events.append(ev)
            self._seen_log = len(tw.log)
            # ---------------- ML feedback (prediction → real outcome → error → model update)
            for p in self.ml.resolve(tw.tick, self._actual):
                self.store.prediction(p, tw.t.isoformat(timespec="minutes"))
            # ---------------- domain agents: ANALYZE / DETECT / INVESTIGATE / PREDICT / PLAN
            busy = {a["issue"] for a in self.actions.values() if a["status"] in ("PENDING_APPROVAL", "ACTIVE", "VERIFYING")}
            ctx_common = dict(tw=tw, ml=self.ml, proc=self.proc, bus=self.bus, tick=tw.tick, thresholds=self.thresholds, busy=busy,
                              effectiveness=self.effectiveness)
            findings: list[dict] = []
            for ag in self.agents:
                rep = ag.run(Ctx(memory=self.memory[ag.domain], **ctx_common))
                self.reports[ag.domain] = rep
                findings.extend(rep["findings"])
            findings.extend(self._sensor_findings())
            # ---------------- UNDERSTAND: city state engine
            self._state_engine()
            # ---------------- priority engine + issue registry (DETECT / INVESTIGATE / PREDICT / PLAN logging)
            self._update_issues(findings)
            self._coordinate()
            # ---------------- DECIDE → safety → ACT
            self._decide()
            # ---------------- VERIFY
            self._verify()
            # ---------------- ADAPT
            self._adapt()
            self._health()
            self._expire_approvals()
            if tw.tick % 12 == 0:
                self.store.infra([q for q in self.reports["infrastructure"]["extra"]["priority_queue"]], tw.t.isoformat(timespec="minutes"))
            self.last_cycle_ms = (time.perf_counter() - t0) * 1000
            return dict(tick=tw.tick)

    # --------------------------------------------------------------- observe helpers
    def _observe_log(self):
        tw = self.tw
        occ = float(np.mean([e.occupancy for e in tw.roads.values() if not e.closed]))
        prev = self._prev_obs.get("occ", occ)
        if occ - prev > 2.5:
            self.log("OBSERVE", "Traffic Agent", f"Traffic density increased ({prev:.1f}% → {occ:.1f}% average occupancy).")
        self._prev_obs["occ"] = 0.8 * prev + 0.2 * occ if abs(occ - prev) <= 2.5 else occ
        g = tw.grid_kw
        pg = self._prev_obs.get("grid", g)
        if g - pg > 400:
            self.log("OBSERVE", "Energy Agent", f"Grid import rose {pg:.0f} → {g:.0f} kW.")
        self._prev_obs["grid"] = g
        if self.proc.last_batch.get("unit_conversions") and tw.tick == 1:
            self.log("ANALYZE", "orchestrator", f"Ingestion online — {self.proc.last_batch['received']} records/interval; unit conversions, "
                                                "de-duplication, validation and imputation active.")

    def _record_live(self):
        tw = self.tw
        ts = tw.t
        for e in tw.roads.values():
            self.live_rows["traffic"].append((ts, e.id, e.zone, e.flow, e.speed, e.vc, e.density, e.occupancy, e.queue_m, e.capacity * e.cap_factor,
                                              e.base_flow, int(e.closed), e.event_mult, tw.weather.temp, tw.weather.rain))
        self.live_rows["energy"].append((ts, tw.grid_kw, tw.load_kw, tw.ev_kw, tw.solar_kw, tw.weather.temp, tw.weather.cloud, tw.weather.rain))
        for k in self.live_rows:
            if len(self.live_rows[k]) > 60000:
                self.live_rows[k] = self.live_rows[k][-40000:]

    def _actual(self, p: dict):
        tw = self.tw
        if p["model"] == "traffic_flow":
            e = tw.roads.get(p["entity"])
            return None if (e is None or e.closed) else e.flow
        if p["model"] == "energy_demand":
            return tw.load_kw + tw.ev_kw
        if p["model"] == "waste_fill":
            b = tw.bins.get(p["entity"])
            if b is None or b.last_collected_tick > p["made_tick"] or b.dumping_flag_until > p["made_tick"]:
                return None
            return b.fill
        if p["model"] == "infra_failure_24h":
            a = tw.assets.get(p["entity"])
            return None if a is None else float(a.status in ("FAILED", "UNDER_REPAIR") or a.fault > 0.9)
        return None

    def _sensor_findings(self) -> list[dict]:
        out = []
        for s in list(self.proc.sensor_issues.values())[:12]:
            ent = s["entity"]
            x = self.proc.entity_state.get(ent, {}).get("_x")
            y = self.proc.entity_state.get(ent, {}).get("_y")
            out.append(dict(key=f"sensor:sensor:{s['sensor_id']}", domain="sensor", kind="sensor", entity=s["sensor_id"], entity_name=s["sensor_id"],
                            zone=self.proc.entity_state.get(ent, {}).get("_zone"), x=x, y=y, title=f"Sensor failure suspected — {s['sensor_id']}",
                            detail=f"{s['issue'].replace('_', ' ')} on {s['metric']} since {s['since'][11:16]}; values quarantined and imputed.",
                            severity=0.2, probability=0.85, impact=0.25, urgency=0.3, evidence=s, root_causes=[], forecast={},
                            proposals=[dict(type="inspection_request", domain=s["domain"], target=s["sensor_id"], title=f"Schedule sensor check for {s['sensor_id']}",
                                            params=dict(sensor=s["sensor_id"]), expected=dict(metric="data quality"),
                                            measure=dict(metric="none", entities=[], direction="none", ticks=1), risk="LOW", tool="verify_action")]))
        return out

    # =============================================================== UNDERSTAND
    def _state_engine(self):
        new = {d: dict(level=r["level"], label=r["label"], health=r["health"]) for d, r in self.reports.items()}
        for d, s in new.items():
            old = self.state_engine.get(d)
            if old and old["level"] != s["level"]:
                up = LEVELS.index(s["level"]) > LEVELS.index(old["level"])
                self.log("UNDERSTAND", self.agent_by_domain[d].name, f"{d.title()} state {old['label']} → {s['label']}"
                                                                     f"{' (escalating)' if up else ' (recovering)'}.")
                if up and LEVELS.index(s["level"]) >= 2 and self.monitor[d] == 1:
                    self.monitor[d] = 2
                    self.log("ADAPT", "orchestrator", f"Increase monitoring frequency for {d} (forecast refresh every tick).")
            elif old and old["level"] == s["level"] and LEVELS.index(s["level"]) < 2 and self.monitor[d] == 2:
                self.monitor[d] = 1
        self.state_engine = new

    # =============================================================== priority engine + issues
    def category(self, score: float) -> str:
        if score >= settings.priority_critical:
            return "CRITICAL"
        if score >= settings.priority_high:
            return "HIGH"
        if score >= settings.priority_medium:
            return "MEDIUM"
        return "LOW"

    def _update_issues(self, findings: list[dict]):
        tw = self.tw
        now = tw.t.isoformat(timespec="minutes")
        seen = set()
        for f in findings:
            score = 100 * f["severity"] * f["probability"] * f["impact"] * f["urgency"]
            f["priority"] = round(score, 2)
            f["category"] = self.category(score)
            for i, p in enumerate(f["proposals"]):
                p["id"] = f"{f['key']}#{i}"
            seen.add(f["key"])
            old = self.issues.get(f["key"])
            if old:
                f["first_seen"], f["first_tick"] = old["first_seen"], old["first_tick"]
                f["status"], f["missing"] = old.get("status", "OPEN"), 0
                f["failed_types"] = old.get("failed_types", [])
                f["failed_reset"] = old.get("failed_reset", tw.tick)
                if f["failed_types"] and tw.tick - f["failed_reset"] > 72:      # failed options become retryable after 6 h
                    f["failed_types"], f["failed_reset"] = [], tw.tick
                f["actions"] = old.get("actions", [])
                f["llm"] = old.get("llm")
                if LEVELS.index({"LOW": "NORMAL", "MEDIUM": "WATCH", "HIGH": "WARNING", "CRITICAL": "CRITICAL"}[f["category"]]) > \
                        LEVELS.index({"LOW": "NORMAL", "MEDIUM": "WATCH", "HIGH": "WARNING", "CRITICAL": "CRITICAL"}[old["category"]]) and f["category"] in ("HIGH", "CRITICAL"):
                    self.log("DETECT", self.agent_name(f["domain"]), f"{f['title']} escalated to {f['category']} (priority {f['priority']:.0f}).", f["key"])
                self.issues[f["key"]] = f
                continue
            f.update(first_seen=now, first_tick=tw.tick, status="OPEN", missing=0, failed_types=[], actions=[], llm=None)
            self.issues[f["key"]] = f
            self._score_detection(f)
            if f["category"] == "LOW" and f["kind"] in ("signal_opportunity", "sensor", "poor_solar_utilization"):
                continue
            ag = self.agent_name(f["domain"])
            self.log("DETECT", ag, f"{f['title']} — {f['category']} (priority {f['priority']:.0f}).", f["key"])
            self.store.event(f, now)
            if f["root_causes"]:
                top = ", ".join(f"{c['factor']} {c['contribution']}%" for c in f["root_causes"][:3])
                self.log("INVESTIGATE", ag, f"Likely contributors: {top}.", f["key"])
            fc = f.get("forecast") or {}
            msg = self._forecast_text(f, fc)
            if msg:
                self.log("PREDICT", ag, msg, f["key"])
            if f["proposals"]:
                p = f["proposals"][0]
                ex = p.get("expected", {})
                tail = f" (expected {ex.get('metric')}: {ex.get('before')} → {ex.get('after')})" if "before" in ex and "after" in ex else ""
                self.log("PLAN", ag, f"{p['title']}{tail}.", f["key"])
        for k in list(self.issues):
            if k not in seen:
                it = self.issues[k]
                it["missing"] = it.get("missing", 0) + 1
                if it["missing"] >= 2:
                    it["status"] = "RESOLVED"
                    it["resolved_at"] = now
                    self.resolved.appendleft(it)
                    self.store.close_event(k)
                    del self.issues[k]

    @staticmethod
    def _forecast_text(f, fc) -> str | None:
        if "vc_60" in fc:
            return f"v/c forecast {fc.get('vc_30')} (+30 min), {fc['vc_60']} (+60 min); congestion probability {fc.get('p_congestion_30', 0):.0%}; travel time {fc.get('travel_time_increase_pct', 0):+.0f}%."
        if "onset_min" in fc:
            return f"Congestion onset in ~{fc['onset_min']} min; travel time {fc.get('travel_time_increase_pct', 0):+.0f}%."
        if "overflow_hours" in fc:
            o = fc["overflow_hours"]
            return f"Fill trajectory {', '.join(f'{k} {min(v, 1.2):.0%}' for k, v in list(fc['trajectory'].items())[:3])}; predicted overflow {'now' if o == 0 else f'~{o:g} h' if o else '> 12 h'}."
        if "failure_probability_24h" in fc:
            return f"Failure probability (24 h) {fc['failure_probability_24h']:.0%}; remaining useful life ≈ {fc['rul_hours']:.0f} h."
        if "peak_kw" in fc:
            return f"Peak {fc['peak_kw']} kW expected at {fc['peak_time'][11:16]}; 24 h cost ≈ {settings.currency}{fc['expected_cost_24h']:,}, CO₂ ≈ {fc['expected_co2_24h_kg']:,} kg."
        if "congestion_increase_pct" in fc:
            return f"Congestion on adjacent roads predicted +{fc['congestion_increase_pct']}%."
        if "crowding_risk_1h" in fc:
            return f"Crowding risk next hour {fc['crowding_risk_1h']:.0%}."
        return None

    def agent_name(self, domain: str) -> str:
        a = self.agent_by_domain.get(domain)
        return a.name if a else "orchestrator"

    def _score_detection(self, f: dict):
        kinds = DETECTION_MAP.get(f["kind"])
        if not kinds:
            return
        tick = self.tw.tick
        match = None
        for i, g in enumerate(self.tw.ground_truth):
            if g["kind"] in kinds and g["entity"] == f["entity"] and g["start"] - 2 <= tick <= g["end"] + 6:
                match = (g["kind"], g["entity"], g["start"])
                break
        grp = DETECTION_GROUP.get(kinds[0], f["domain"])
        if match:
            if match not in self.truth_detected:
                self.truth_detected.add(match)
                self.det[grp]["tp"] += 1
        else:
            self.det[grp]["fp"] += 1


    def _count_misses(self):
        tick = self.tw.tick
        for g in self.tw.ground_truth:
            key = (g["kind"], g["entity"], g["start"])
            if g["kind"] not in DETECTION_GROUP or key in self.truth_detected or key in self.truth_seen:
                continue
            if min(g["end"], g["start"] + 72) + 6 < tick:
                self.truth_seen.add(key)
                self.det[DETECTION_GROUP[g["kind"]]]["fn"] += 1

    # =============================================================== cross-domain coordination
    def _coordinate(self):
        tw = self.tw
        msgs = [m for m in self.bus.messages if m["tick"] == tw.tick and m["to"] == "orchestrator"]
        closures = [m for m in msgs if m["kind"] == "road_closure"]
        for m in closures:
            rid = m["payload"]["road"]
            related = [x for x in msgs if x["payload"].get("road") == rid and x is not m]
            steps = [dict(agent=m["sender"], text=m["text"])] + [dict(agent=x["sender"], text=x["text"]) for x in related]
            keys = [k for k in (f"traffic:closure_impact:{rid}", f"safety:emergency_route_affected:{rid}") if k in self.issues]
            keys += [k for k in self.issues if k.startswith("waste:collection_delay:")]
            corridor_nodes = set()
            div = self.issues.get(f"traffic:closure_impact:{rid}")
            retime = []
            if div and div["proposals"]:
                for path, _ in div["proposals"][0]["params"].get("paths", []):
                    for eid in path:
                        corridor_nodes.update((tw.roads[eid].u, tw.roads[eid].v))
                ops = {p["node"]: p for p in opt.signal_opportunities(tw, min_gain_pct=3, min_volume=300)}
                retime = [n for n in corridor_nodes if n in ops][:4]
            plan = dict(id=self._nid("PLAN"), ts=tw.t.isoformat(timespec="minutes"), tick=tw.tick, trigger=f"Road closure — {tw.roads[rid].name}",
                        road=rid, steps=steps, issues=keys, retime_nodes=retime, status="ACTIVE",
                        actions=["Coordinated traffic diversion", "Signal retiming on diversion corridor" if retime else None,
                                 "Emergency-corridor protection" if any("safety" in k for k in keys) else None,
                                 "Waste route adjustment" if any("waste" in k for k in keys) else None])
            plan["actions"] = [a for a in plan["actions"] if a]
            self.plans.appendleft(plan)
            self.log("PLAN", "City Orchestrator", f"Coordinated plan {plan['id']}: " + " + ".join(plan["actions"]) + ".", plan["id"])
            for k in keys:
                if k in self.issues:
                    self.issues[k]["plan"] = plan["id"]
                    self.issues[k]["urgency"] = max(self.issues[k]["urgency"], 0.9)
            if retime:
                plans = {n: opt.webster_plan(tw, n) for n in retime}
                self._execute_proposal(div, dict(id=f"{plan['id']}#retime", type="optimize_signal_timing", domain="traffic", target=",".join(retime),
                                                 title=f"Retime diversion-corridor signals ({', '.join(retime)})",
                                                 params=dict(plans={n: p["proposed"] for n, p in plans.items()}),
                                                 expected=dict(metric="avg intersection delay (s/veh)"),
                                                 measure=dict(metric="node_delay", entities=retime, direction="decrease", ticks=settings.verification_ticks),
                                                 risk="LOW"), reasoner="City Orchestrator", rationale="Part of coordinated closure plan " + plan["id"])

    # =============================================================== DECIDE
    def _decide(self):
        tw = self.tw
        new_actions = 0
        llm_on = self.llm.enabled
        # apply LLM decisions that arrived (async)
        if self.llm_result:
            res, self.llm_result = self.llm_result, None
            self._apply_llm(res)
        ranked = sorted(self.issues.values(), key=lambda f: -f["priority"])
        llm_queue = []
        for f in ranked:
            if not f["proposals"] or new_actions >= 4:
                continue
            if any(self.actions[a]["status"] in ("PENDING_APPROVAL", "ACTIVE", "VERIFYING") for a in f.get("actions", []) if a in self.actions):
                continue
            last = [self.actions[a] for a in f.get("actions", []) if a in self.actions]
            cands = [p for p in f["proposals"] if p["type"] not in f.get("failed_types", [])]
            if not cands:
                continue
            tried = {a["type"] for a in last if tw.tick - a["created_tick"] < 36}
            # escalation: an advisory-only response does not block a stronger, untried measure when the issue worsens
            stronger = [p for p in cands if p["measure"]["metric"] != "none" and p["type"] not in tried]
            if last and last[-1]["measure"]["metric"] == "none" and stronger and last[-1]["status"] == "COMPLETED":
                cands = stronger
            elif last and tw.tick - last[-1]["created_tick"] < 18 and last[-1]["status"] not in ("FAILED", "REJECTED", "EXPIRED"):
                continue   # cooldown after a successful / completed action
            if last and last[-1]["status"] in ("REJECTED", "EXPIRED") and tw.tick - last[-1]["created_tick"] < 36:
                continue
            life_safety = f["kind"] in ("incident", "emergency_route_affected")
            if f["category"] == "LOW" and not life_safety and all(POLICY.get(p["type"], ("HIGH",))[0] != "LOW" for p in cands):
                continue
            if llm_on and f["category"] in ("HIGH", "CRITICAL") and tw.tick - f["first_tick"] < 3 and f.get("llm") is None:
                llm_queue.append(f)
                continue
            p = self._choose(cands)
            rationale = self._rationale(f, p)
            self._execute_proposal(f, p, reasoner="Deterministic decision engine" if not f.get("llm") else "LLM", rationale=f.get("llm") or rationale)
            new_actions += 1
        if llm_queue and not self.llm_busy and time.time() - self.llm.last_call_ts > settings.llm_min_interval_s:
            self._llm_review(llm_queue[:5])

    def _choose(self, cands: list[dict]) -> dict:
        def score(p):
            ex = p.get("expected", {})
            gain = ex.get("improvement_pct")
            gain = 10.0 if gain is None else max(0.0, gain)
            e = self.effectiveness[p["type"]]
            eff = (e["success"] + 0.5 * e["partial"] + 1) / (e["total"] + 2)
            risk_pen = {"LOW": 1.0, "MEDIUM": 0.9, "HIGH": 0.8}[POLICY.get(p["type"], ("HIGH",))[0]]
            safety_bonus = 2.0 if p["type"] in ("emergency_dispatch", "maintenance_request", "road_closure", "crowd_management") else 1.0
            return (gain + 5) * eff * risk_pen * safety_bonus
        return max(cands, key=score)

    def _rationale(self, f: dict, p: dict) -> str:
        ex = p.get("expected", {})
        parts = [f"{f['category']} priority {f['priority']:.0f} (S{f['severity']:.2f}×P{f['probability']:.2f}×I{f['impact']:.2f}×U{f['urgency']:.2f})"]
        if f["root_causes"]:
            parts.append(f"main driver: {f['root_causes'][0]['factor']}")
        if "before" in ex and "after" in ex:
            parts.append(f"tool estimate {ex.get('metric')}: {ex['before']} → {ex['after']}")
        e = self.effectiveness[p["type"]]
        if e["total"]:
            parts.append(f"past success {e['success']}/{e['total']}")
        return "; ".join(parts) + "."

    # --------------------------------------------------------------- LLM (async)
    def _llm_review(self, issues: list[dict]):
        brief = dict(time=self.tw.t.isoformat(timespec="minutes"), city_state=self.state_engine,
                     health=self.health_hist[-1]["score"] if self.health_hist else None,
                     issues=[dict(issue_key=f["key"], title=f["title"], category=f["category"], priority=f["priority"], detail=f["detail"],
                                  root_causes=f["root_causes"][:3], forecast=f.get("forecast"),
                                  candidates=[dict(proposal_id=p["id"], title=p["title"], type=p["type"], risk=POLICY.get(p["type"], ("HIGH",))[0],
                                                   requires_authorization=POLICY.get(p["type"], ("HIGH", True))[1], expected=p.get("expected"))
                                              for p in f["proposals"]]) for f in issues])
        self.llm_busy = True
        for f in issues:
            f["llm"] = "pending"

        def job():
            try:
                text, trace = self.llm.run("Situation briefing (JSON):\n" + json.dumps(brief, default=str))
                self.llm_result = dict(json=extract_json(text), trace=trace, issues=[f["key"] for f in issues], raw=text[:2000])
            except Exception as e:
                self.llm_result = dict(json=None, trace=[], issues=[f["key"] for f in issues], error=str(e))
            finally:
                self.llm_busy = False
        threading.Thread(target=job, daemon=True).start()
        self.log("DECIDE", "City Orchestrator", f"LLM reasoning requested for {len(issues)} high-priority issue(s).")

    def _apply_llm(self, res: dict):
        data = res.get("json")
        if not data:
            self.log("DECIDE", "City Orchestrator", f"LLM unavailable/invalid ({res.get('error', 'unparseable')[:80]}) — deterministic fallback.")
            for k in res["issues"]:
                if k in self.issues:
                    self.issues[k]["llm"] = None
                    self.issues[k]["first_tick"] -= 5
            return
        self.llm_summary = str(data.get("summary", ""))[:600]
        valid = {p["id"]: (f, p) for f in self.issues.values() for p in f["proposals"]}
        tools = ", ".join(sorted({t["tool"] for t in res.get("trace", [])})) or "none"
        for d in data.get("decisions", []):
            key, pid = d.get("issue_key"), d.get("proposal_id")
            f = self.issues.get(key)
            if not f:
                continue
            rationale = f"[LLM, tools: {tools}] " + str(d.get("rationale", ""))[:300]
            if pid and pid in valid and valid[pid][0]["key"] == key:
                f["llm"] = rationale
                self._execute_proposal(f, valid[pid][1], reasoner=f"LLM ({self.llm.model})", rationale=rationale)
            else:
                f["llm"] = rationale + " (deferred)"
                if pid:
                    self.log("DECIDE", "Safety Layer", f"LLM selected unknown proposal '{pid}' — rejected by validation.", key)
        for k in res["issues"]:
            if k in self.issues and self.issues[k].get("llm") == "pending":
                self.issues[k]["llm"] = None

    # =============================================================== SAFETY LAYER + ACT
    def safety_check(self, p: dict) -> dict:
        tw = self.tw
        risk, auth, why = POLICY.get(p["type"], ("HIGH", True, "unknown action type — default deny to human review"))
        checks, blocked = [], []
        prm = p.get("params", {})
        if p["type"] == "optimize_signal_timing":
            for n, pl in prm.get("plans", {}).items():
                ok = 0.2 <= pl["g_ns"] <= 0.8 and 50 <= pl["cycle"] <= 140 and tw.nodes[n].preempt_until < tw.tick
                checks.append(f"{n}: bounds & no active pre-emption {'✓' if ok else '✗'}")
                if not ok:
                    blocked.append(f"signal plan for {n} out of bounds or emergency pre-emption active")
        if p["type"] in ("reroute_traffic", "traffic_diversion"):
            corridor_edges = {eid for c in tw.corridors for eid in tw.path_edges(c["nodes"])}
            paths = prm.get("paths", [])
            other = set(tw.reroutes) - {prm.get("road")}
            clean = [(pp, s) for pp, s in paths if not (set(pp) & corridor_edges) and not any(tw.roads[e].closed for e in pp)
                     and not (set(pp) & other)]
            if len(clean) < len(paths):
                checks.append("removed alternate paths crossing emergency corridors / closures (conflict resolution)")
                tot = sum(s for _, s in clean) or 1
                prm["paths"] = [(pp, s / tot) for pp, s in clean]
            if not prm.get("paths"):
                blocked.append("no conflict-free alternate path")
            if prm.get("fraction", 0) > 0.35:
                blocked.append("reroute fraction above 35% policy limit")
        if p["type"] == "hvac_adjustment":
            b = tw.buildings[prm["building"]]
            if b.kind == "health" or prm.get("offset", 0) > 2 or "EXTREME_HEAT" in tw.weather.alerts and b.kind in ("residential",):
                blocked.append("comfort / health constraint: set-point relaxation not permitted for this building or weather")
            checks.append("comfort band ≤ +2 °C ✓")
        if p["type"] == "ev_charging_shift" and prm.get("fraction", 0) > 0.5:
            blocked.append("EV shift above 50% breaches charging SLA")
        if p["type"] in ("maintenance_request", "inspection_request") and prm.get("unit"):
            if tw.units[prm["unit"]].status != "available":
                alt = opt.best_unit(tw, tw.nearest_node(*self._xy(prm.get("target"))), ("crew",))
                if alt:
                    prm["unit"] = alt["unit"]
                    checks.append(f"crew re-assigned to {alt['unit']}")
                else:
                    blocked.append("no maintenance crew available")
        if p["type"] in ("emergency_dispatch", "patrol_dispatch", "emergency_preposition"):
            if tw.units[prm["unit"]].status != "available":
                blocked.append(f"{prm['unit']} no longer available")
        if p["type"] == "dispatch_collection":
            for tid in prm.get("routes", {}):
                if tw.trucks[tid].status != "idle":
                    blocked.append(f"{tid} is not idle")
        if p["type"] == "road_closure":
            checks.append("critical-infrastructure action → human authorisation required")
        # data-quality guard: never act on a finding that rests only on quarantined sensor data
        suspect = [s for s in self.proc.sensor_issues if s.startswith(f"S-{p['target']}-")]
        if suspect and p["type"] not in ("inspection_request", "safety_alert", "emergency_dispatch"):
            blocked.append(f"underlying sensor data quarantined ({', '.join(suspect)})")
        needs_auth = auth and not settings.auto_approve_high_in_sim
        return dict(risk=risk, requires_authorization=needs_auth, policy=why, checks=checks, blocked=blocked, passed=not blocked)

    def _xy(self, target) -> tuple[float, float]:
        tw = self.tw
        for coll in (tw.assets, tw.buildings, tw.bins):
            if target in coll:
                return coll[target].x, coll[target].y
        return 500.0, 300.0

    def _execute_proposal(self, f: dict, p: dict, reasoner: str, rationale: str):
        tw = self.tw
        sc = self.safety_check(p)
        aid = self._nid("ACT")
        a = dict(id=aid, issue=f["key"], issue_title=f["title"], created=tw.t.isoformat(timespec="minutes"), created_tick=tw.tick,
                 domain=p["domain"], type=p["type"], target=p["target"], title=p["title"], params=p["params"], expected=p.get("expected"),
                 measure=p["measure"], risk=sc["risk"], safety=sc, execution_mode=EXECUTION_MODE, status="NEW", verification=None,
                 decision=dict(reasoner=reasoner, rationale=rationale, priority=f["priority"], category=f["category"],
                               alternatives=[x["title"] for x in f["proposals"] if x is not p]), tool=p.get("tool"))
        self.actions[aid] = a
        self.action_order.appendleft(aid)
        f.setdefault("actions", []).append(aid)
        ag = self.agent_name(p["domain"])
        self.log("DECIDE", "City Orchestrator", f"{p['title']} — {reasoner}: {rationale[:160]}", aid)
        if not sc["passed"]:
            a["status"] = "BLOCKED"
            f.setdefault("failed_types", []).append(p["type"])
            self.log("DECIDE", "Safety Layer", f"Blocked {aid}: {'; '.join(sc['blocked'])}.", aid)
        elif sc["requires_authorization"]:
            a["status"] = "PENDING_APPROVAL"
            self.log("ACT", "Safety Layer", f"{aid} awaits human authorisation ({sc['policy']}).", aid)
        else:
            self._act(a)
        self.store.action(a)
        return a

    def _act(self, a: dict):
        tw = self.tw
        m = a["measure"]
        a["before"] = self.measure(m)
        cf = None
        if m["metric"] not in ("none", "incident_response", "unit_arrival", "asset_status", "coverage", "contamination") and m.get("ticks", 0) <= 300:
            clone = tw.clone()
            peak = 0.0
            for _ in range(m["ticks"]):
                clone.step()
                peak = max(peak, clone.grid_kw)
            cf = round(peak, 1) if m["metric"] == "grid_peak_window" else self.measure(m, clone)
        if m["metric"] == "grid_peak_window":
            a["before"] = (a.get("expected") or {}).get("before")       # forecast peak without action
            a["window_peak"] = 0.0
        result = self.apply(a)
        a["result"] = result
        a["counterfactual"] = cf
        a["executed_tick"] = tw.tick
        a["verify_tick"] = tw.tick + max(1, m.get("ticks", 1))
        a["status"] = "COMPLETED" if m["metric"] == "none" else "ACTIVE"
        if a["status"] == "COMPLETED":
            a["verification"] = dict(result="RECORDED", note="Advisory / record-only action — no physical effect claimed.")
        self.log("ACT", self.agent_name(a["domain"]), f"{a['title']} — executed in simulation. {result.get('summary', '')}".strip(), a["id"])

    # --------------------------------------------------------------- action engine (digital twin)
    def apply(self, a: dict) -> dict:
        tw = self.tw
        p, t = a["params"], a["type"]
        if t == "optimize_signal_timing":
            for n, pl in p["plans"].items():
                tw.nodes[n].g_ns, tw.nodes[n].cycle, tw.nodes[n].plan_source = pl["g_ns"], pl["cycle"], "AI-adaptive"
            return dict(summary=f"Adaptive timing active at {', '.join(p['plans'])} (initial plan " +
                        ", ".join(f"{k}: NS {v['g_ns']:.0%}/C{v['cycle']}s" for k, v in p['plans'].items()) + ").")
        if t == "reroute_traffic":
            tw.reroutes[p["road"]] = dict(fraction=p["fraction"], paths=[(list(x), s) for x, s in p["paths"]], until=tw.tick + p.get("duration_ticks", 12))
            return dict(summary=f"{int(p['fraction'] * 100)}% of trips advised onto {len(p['paths'])} alternate path(s).")
        if t == "traffic_diversion":
            tw.diversions[p["road"]] = dict(paths=[(list(x), s) for x, s in p["paths"]])
            return dict(summary=f"Diversion over {len(p['paths'])} path(s) published.")
        if t == "emergency_dispatch":
            r = tw.dispatch_unit(p["unit"], p["node"], p["incident"], corridor=p.get("corridor", False))
            return dict(summary=f"{p['unit']} en route, ETA {r.get('eta_min')} min{' with signal pre-emption' if p.get('corridor') else ''}.", **r)
        if t in ("emergency_preposition", "patrol_dispatch"):
            r = tw.dispatch_unit(p["unit"], p["node"], f"standby:{p.get('zone')}", corridor=False)
            return dict(summary=f"{p['unit']} moving to {p['node']} (ETA {r.get('eta_min')} min).", **r)
        if t == "crowd_management":
            z = tw.zones[p["zone"]]
            z.crowd_factor, z.crowd_mgmt_until = p["factor"], tw.tick + p["duration_ticks"]
            return dict(summary=f"Crowd-management measures active in {z.name}.")
        if t in ("dispatch_collection",):
            out = [tw.dispatch_truck(tid, bins, source="AI predictive route") for tid, bins in p["routes"].items()]
            return dict(summary="; ".join(f"{o['truck']}: {len(o['stops'])} stops, {o['km']} km" for o in out), routes=out)
        if t == "reoptimize_collection":
            tr = tw.trucks[p["truck"]]
            if tr.status == "collecting":
                o = tw.dispatch_truck(tr.id, [s[0] for s in tr.stops], source="AI re-optimised (closure)")
                return dict(summary=f"{tr.id} re-routed: {o['km']} km.")
            return dict(summary="Truck no longer collecting.")
        if t == "segregation_advisory":
            b = tw.bins[p["bin"]]
            b.contamination_event_until = min(b.contamination_event_until, tw.tick + 6)
            return dict(summary=f"Advisory sent to premises near {b.id}.")
        if t == "battery_dispatch":
            bat = tw.energy_assets["BAT-1"]
            bat.schedule = {int(k): v for k, v in p["schedule"].items()}
            bat.peak_guard = float(p.get("peak_guard_kw") or 0)
            return dict(summary=f"24-hour battery schedule loaded; peak guard {bat.peak_guard:.0f} kW.")
        if t == "ev_charging_shift":
            for x in tw.energy_assets.values():
                if x.kind == "ev_hub":
                    x.ev_shift, x.ev_shift_until = p["fraction"], tw.tick + tw.ticks_for(max(30, (21 - tw.hour) * 60))
            return dict(summary=f"{int(p['fraction'] * 100)}% of flexible EV charging deferred.")
        if t == "hvac_adjustment":
            b = tw.buildings[p["building"]]
            b.hvac_offset, b.hvac_offset_until = p["offset"], tw.tick + p["duration_ticks"]
            return dict(summary=f"{b.name} set-point +{p['offset']} °C for {p['duration_ticks'] * tw.dt_min // 60} h.")
        if t == "lighting_schedule":
            tw.buildings[p["building"]].lighting_fault = False
            return dict(summary="Lighting schedule restored via BMS.")
        if t == "shutdown_idle_equipment":
            tw.buildings[p["building"]].plug_fault = 0.0
            return dict(summary="Idle equipment shutdown scheduled.")
        if t in ("maintenance_request", "inspection_request"):
            target = p.get("target") or p.get("sensor") or p.get("bin")
            wo = self._nid("WO")
            if target in tw.assets or target in tw.buildings:
                unit = p.get("unit") or (opt.best_unit(tw, tw.nearest_node(*self._xy(target)), ("crew",)) or {}).get("unit")
                if unit:
                    node = tw.nearest_node(*self._xy(target))
                    r = tw.dispatch_unit(unit, node, target)
                    if target in tw.assets:
                        tw.assets[target].work_order = wo
                    return dict(summary=f"Work order {wo}: {unit} dispatched (ETA {r.get('eta_min')} min).", work_order=wo, **r)
                return dict(summary=f"Work order {wo} queued — no crew free.", work_order=wo)
            return dict(summary=f"Work order {wo} created for {target}.", work_order=wo)
        if t == "road_closure":
            tw.close_road(p["road"], "precautionary (authorised)")
            return dict(summary=f"{tw.roads[p['road']].name} closed (simulated).")
        if t in ("safety_alert", "incident_escalation"):
            return dict(summary="Notification recorded.")
        return dict(summary="No-op.")

    def measure(self, m: dict, tw=None):
        tw = tw or self.tw
        k, ents = m["metric"], m.get("entities", [])
        if k == "node_delay":
            return round(tw.avg_intersection_delay(ents), 2)
        if k == "network_delay":
            return round(tw.avg_intersection_delay(), 2)
        if k == "road_vc":
            return round(float(np.mean([tw.roads[e].vc for e in ents])), 3)
        if k == "max_vc":
            return round(max((tw.roads[e].vc for e in ents if not tw.roads[e].closed), default=0.0), 3)
        if k == "zone_density":
            return round(tw.zones[ents[0]].density, 3)
        if k == "bins_overflow":
            return int(sum(1 for b in ents if tw.bins[b].fill >= 1.0)) + round(float(np.mean([min(tw.bins[b].fill, 1.3) for b in ents])), 3)
        if k == "building_power":
            return round(tw.buildings[ents[0]].power_kw, 1)
        if k == "grid_kw":
            return round(tw.grid_kw, 1)
        if k == "grid_peak_window":
            return round(tw.energy_totals["day_peak_kw"], 1)
        if k == "ev_kw":
            return round(tw.ev_kw, 1)
        if k == "contamination":
            return round(tw.bins[ents[0]].contamination * 100, 1)
        if k == "asset_status":
            a = tw.assets.get(ents[0])
            return None if a is None else a.status
        if k == "incident_response":
            inc = tw.incidents.get(ents[0])
            return None if inc is None else inc.status
        if k == "unit_arrival":
            return tw.units[ents[0]].status
        if k == "coverage":
            return round(opt.coverage(tw).get(ents[0], math.inf), 2)
        return None

    # =============================================================== VERIFY
    def _verify(self):
        tw = self.tw
        for a in list(self.actions.values()):
            if a["status"] == "ACTIVE" and a["measure"]["metric"] == "grid_peak_window":
                a["window_peak"] = max(a.get("window_peak", 0.0), tw.grid_kw)
            if a["status"] not in ("ACTIVE",) or tw.tick < a["verify_tick"]:
                continue
            m = a["measure"]
            after = round(a.get("window_peak", 0.0), 1) if m["metric"] == "grid_peak_window" else self.measure(m)
            res, note, imp, imp_cf = "SUCCESS", "", None, None
            if m["metric"] == "asset_status":
                if after in ("OPERATIONAL",) and tw.assets[m["entities"][0]].fault == 0:
                    res, note = "SUCCESS", "Asset restored to service; fault cleared."
                elif after == "UNDER_REPAIR" or tw.units.get(a["params"].get("unit", ""), None) and tw.units[a["params"]["unit"]].status in ("en_route", "on_scene"):
                    if tw.tick - a["executed_tick"] < 200:
                        a["verify_tick"] = tw.tick + 6
                        continue
                    res, note = "FAILED", "Repair not completed in time."
                else:
                    res, note = ("SUCCESS", "Inspection completed.") if a["type"] == "inspection_request" else ("FAILED", f"Asset status {after}.")
            elif m["metric"] == "incident_response":
                inc = tw.incidents.get(m["entities"][0])
                if inc is None or after in ("ON_SCENE", "CLEARED"):
                    rt = (inc.responded_tick - a["executed_tick"]) * tw.dt_min if inc and inc.responded_tick > 0 else None
                    res, note = "SUCCESS", f"Unit on scene{f' after {rt} min' if rt is not None else ''} (planned ETA {a['result'].get('eta_min')} min)."
                elif tw.tick - a["executed_tick"] < 24:
                    a["verify_tick"] = tw.tick + 2
                    continue
                else:
                    res, note = "FAILED", "Unit did not reach scene in time."
            elif m["metric"] == "unit_arrival":
                res, note = ("SUCCESS", f"{m['entities'][0]} reached position.") if after in ("on_scene", "returning", "available") else ("PARTIAL", f"unit {after}")
            elif m["metric"] == "contamination":
                imp = (a["before"] - after) / max(a["before"], 1e-6) * 100
                res = "SUCCESS" if after < 20 else "PARTIAL" if imp > 0 else "FAILED"
                note = f"Contamination {a['before']}% → {after}%."
            else:
                before, cf = a.get("before"), a.get("counterfactual")
                if isinstance(before, (int, float)) and isinstance(after, (int, float)):
                    imp = (before - after) / max(abs(before), 1e-6) * 100
                    if isinstance(cf, (int, float)):
                        imp_cf = (cf - after) / max(abs(cf), 1e-6) * 100
                    ref = imp_cf if imp_cf is not None else imp
                    res = "SUCCESS" if ref >= 3 else "PARTIAL" if ref > 0.1 else "FAILED"
                    note = (f"Before {before} → after {after}"
                            + (f"; without action (counterfactual twin) {cf}" if cf is not None else "") + ".")
            a["after"] = after
            a["verification"] = dict(result=res, note=note, before=a.get("before"), after=after, counterfactual=a.get("counterfactual"),
                                     improvement_pct=None if imp is None else round(imp, 1),
                                     improvement_vs_counterfactual_pct=None if imp_cf is None else round(imp_cf, 1),
                                     verified_at=tw.t.isoformat(timespec="minutes"))
            a["status"] = "VERIFIED" if res in ("SUCCESS", "PARTIAL") else "FAILED"
            e = self.effectiveness[a["type"]]
            e["total"] += 1
            e[{"SUCCESS": "success", "PARTIAL": "partial", "FAILED": "failed"}[res]] += 1
            pct = imp_cf if imp_cf is not None else imp
            self.log("VERIFY", self.agent_name(a["domain"]), f"{a['title']}: {res}" + (f" ({pct:+.1f}% vs {'counterfactual' if imp_cf is not None else 'before'})" if pct is not None else "") + f". {note}", a["id"])
            if res == "FAILED":
                f = self.issues.get(a["issue"])
                if f is not None:
                    f.setdefault("failed_types", []).append(a["type"])
                    self.log("INVESTIGATE", "City Orchestrator", f"Action {a['id']} failed → re-investigating {f['title']} and re-planning with alternatives.", a["id"])
            self.store.action(a)
        # coordinated plans complete when their road reopens
        for pl in self.plans:
            if pl["status"] == "ACTIVE" and not self.tw.roads[pl["road"]].closed:
                pl["status"] = "COMPLETED"
                self.log("VERIFY", "City Orchestrator", f"{pl['id']} complete — {self.tw.roads[pl['road']].name} reopened.", pl["id"])

    def _expire_approvals(self):
        tw = self.tw
        for a in self.actions.values():
            if a["status"] == "PENDING_APPROVAL" and tw.tick - a["created_tick"] > settings.approval_timeout_ticks:
                a["status"] = "EXPIRED"
                self.log("DECIDE", "Safety Layer", f"{a['id']} authorisation window expired — will re-plan if still needed.", a["id"])
                self.store.action(a)
        if len(self.actions) > 600:
            keep = set(list(self.action_order)[:300]) | {k for k, v in self.actions.items() if v["status"] in ("PENDING_APPROVAL", "ACTIVE")}
            self.actions = {k: v for k, v in self.actions.items() if k in keep}

    def approve(self, aid: str, approver: str = "operator") -> dict:
        with self.lock:
            a = self.actions.get(aid)
            if not a or a["status"] != "PENDING_APPROVAL":
                return dict(ok=False, reason="not pending")
            sc = self.safety_check(dict(type=a["type"], params=a["params"], target=a["target"]))
            if not sc["passed"]:
                a["status"] = "BLOCKED"
                a["safety"] = sc
                self.log("DECIDE", "Safety Layer", f"{aid} re-validated at approval time and blocked: {'; '.join(sc['blocked'])}.", aid)
                self.store.action(a)
                return dict(ok=False, reason="; ".join(sc["blocked"]))
            a["authorized_by"] = approver
            a["authorized_at"] = self.tw.t.isoformat(timespec="minutes")
            self.log("ACT", "Safety Layer", f"{aid} authorised by {approver}.", aid)
            self._act(a)
            self.store.action(a)
            return dict(ok=True, action=a)

    def reject(self, aid: str, approver: str = "operator", reason: str = "") -> dict:
        with self.lock:
            a = self.actions.get(aid)
            if not a or a["status"] != "PENDING_APPROVAL":
                return dict(ok=False, reason="not pending")
            a["status"] = "REJECTED"
            a["rejected_by"], a["reject_reason"] = approver, reason
            self.log("DECIDE", "Safety Layer", f"{aid} rejected by {approver}{': ' + reason if reason else ''}.", aid)
            f = self.issues.get(a["issue"])
            if f:
                f.setdefault("failed_types", []).append(a["type"])
            self.store.action(a)
            return dict(ok=True)

    # =============================================================== ADAPT
    def _adapt(self):
        tw = self.tw
        for entry in list(self.ml.adaptation_log)[-3:]:
            if not entry.get("_logged"):
                entry["_logged"] = True
                if "predicted" in entry:
                    self.log("ADAPT", "ML Engine", f"{entry['model']}: predicted {entry['predicted']:,} vs actual {entry['actual']:,} "
                                                   f"(error {entry['error_pct']:+.1f}%) → bias {entry['old_bias']} → {entry['new_bias']}.")
                else:
                    self.log("ADAPT", "ML Engine", f"{entry['model']}: {entry['note']}.")
        if tw.tick % 12 == 0:
            self._count_misses()
            for grp, (thr, step, lo, hi) in (("safety", ("z", 0.1, 2.2, 4.5)), ("energy", ("z", 0.1, 2.2, 4.5)),
                                              ("infrastructure", ("failure_prob", 0.03, 0.25, 0.65))):
                d = self.det[grp]
                n = d["tp"] + d["fp"]
                key = (grp, d["tp"], d["fp"], d["fn"])
                if n >= 12 and self._prev_obs.get(("adapt", grp)) != key:
                    self._prev_obs[("adapt", grp)] = key
                    prec = d["tp"] / n
                    rec = d["tp"] / max(1, d["tp"] + d["fn"])
                    if prec < 0.4 and self.thresholds[thr] < hi:
                        self.thresholds[thr] = round(self.thresholds[thr] + step, 3)
                        self.log("ADAPT", "City Orchestrator", f"{grp} detection precision {prec:.0%} → raise {thr} threshold to {self.thresholds[thr]}.")
                    elif rec < 0.5 and prec > 0.8 and self.thresholds[thr] > lo:
                        self.thresholds[thr] = round(self.thresholds[thr] - step, 3)
                        self.log("ADAPT", "City Orchestrator", f"{grp} detection recall {rec:.0%} → lower {thr} threshold to {self.thresholds[thr]}.")
            # congestion threshold: if signal/reroute actions keep failing, act earlier
            e = self.effectiveness["reroute_traffic"]
            if e["total"] >= 6 and e["failed"] / e["total"] > 0.5 and self.thresholds["congestion_vc"] > 0.75:
                self.thresholds["congestion_vc"] = round(self.thresholds["congestion_vc"] - 0.02, 2)
                self.log("ADAPT", "Traffic Agent", f"Reroutes often too late → congestion trigger lowered to v/c {self.thresholds['congestion_vc']}.")
                e["failed"] = e["total"] = e["success"] = e["partial"] = 0
        # monitoring frequency: refresh forecasts each tick for escalated domains
        for d, f in self.monitor.items():
            if f == 2 and d in ("energy", "waste"):
                self.memory[d].pop("fc" if d == "energy" else "bin_fc", None)
        # periodic retraining (every simulated day) on history + live data
        if tw.tick % 288 == 0 and tw.tick > 0:
            h = load_history()
            if h:
                live_t = pd.DataFrame(self.live_rows["traffic"], columns=list(h["traffic"].columns))
                live_e = pd.DataFrame(self.live_rows["energy"], columns=list(h["energy"].columns[:8]))
                hist_e = h["energy"][list(h["energy"].columns[:8])]
                frames = dict(traffic=pd.concat([h["traffic"].iloc[-len(h["traffic"]) // 2:], live_t]),
                              energy=pd.concat([hist_e, live_e]))
                if self.ml.retrain_async(frames):
                    self.log("ADAPT", "ML Engine", "Daily retraining started (history + live observations).")

    # =============================================================== HEALTH SCORE
    WEIGHTS = dict(traffic=0.2, safety=0.2, waste=0.15, energy=0.15, infrastructure=0.2, environment=0.05, reliability=0.05)

    def _health(self):
        tw = self.tw
        dom = {d: r["health"] for d, r in self.reports.items()}
        thr = self.thresholds["peak_kw"]
        mean_vc = self.reports["traffic"]["kpis"]["mean_vc"]
        over = self.reports["waste"]["kpis"]["overflowing_bins"]
        env = 100 - 60 * max(0, mean_vc - 0.5) - 4 * over - 30 * max(0, tw.grid_kw / thr - 0.85) - 10 * (1 if any(a.curtail for a in tw.energy_assets.values()) else 0)
        ver = [a for a in self.actions.values() if a.get("verification") and a["verification"].get("result") in ("SUCCESS", "PARTIAL", "FAILED")][-40:]
        succ = sum(1 for a in ver if a["verification"]["result"] == "SUCCESS") + 0.5 * sum(1 for a in ver if a["verification"]["result"] == "PARTIAL")
        sensor_ok = 1 - len(self.proc.sensor_issues) / max(1, len(self.proc.stats)) * 10
        rel = 100 * (0.6 * (succ / len(ver) if ver else 0.85) + 0.4 * max(0, sensor_ok))
        dom["environment"], dom["reliability"] = max(0, min(100, env)), max(0, min(100, rel))
        score = sum(self.WEIGHTS[d] * v for d, v in dom.items())
        prev = next((h for h in reversed(self.health_hist) if tw.tick - h["tick"] >= 12), self.health_hist[0] if self.health_hist else None)
        contrib = {d: round(self.WEIGHTS[d] * (v - 100), 1) for d, v in dom.items()}
        delta = {d: round(v - prev["domains"][d], 1) for d, v in dom.items()} if prev else {d: 0.0 for d in dom}
        self.health_hist.append(dict(tick=tw.tick, ts=tw.t.isoformat(timespec="minutes"), score=round(score, 1), domains={d: round(v, 1) for d, v in dom.items()},
                                     contributions=contrib, delta_1h=delta))

    # =============================================================== TOOLS (LLM + API)
    def call_tool(self, name: str, args: dict) -> dict:
        with self.lock:
            tw = self.tw
            r = self.reports
            if name == "get_traffic_status":
                return dict(kpis=r["traffic"]["kpis"], issues=[self._brief(f) for f in r["traffic"]["findings"]][:12])
            if name == "predict_congestion":
                e = tw.roads.get(args.get("road_id", ""))
                if not e:
                    return dict(error="unknown road_id", roads=list(tw.roads))
                fc = self.ml.predict_flows([e], tw.t, [15, 30, 60, 120], tw.weather)
                cap = e.capacity * e.cap_factor
                return dict(road=e.id, name=e.name, current_vc=round(e.vc, 3),
                            forecast={f"+{h}min": dict(flow=round(fc[(e.id, h)]), vc=round(fc[(e.id, h)] / cap, 3),
                                                       p_congestion=round(self.ml.congestion_probability(fc[(e.id, h)] / cap), 2)) for h in (15, 30, 60, 120)},
                            model_validation=self.ml.metrics.get("traffic_flow"))
            if name == "detect_safety_incident":
                return dict(kpis=r["safety"]["kpis"], zones=r["safety"]["extra"]["zones"], coverage=r["safety"]["extra"]["coverage"],
                            incidents=[self._brief(f) for f in r["safety"]["findings"] if f["kind"] == "incident"])
            if name == "get_bin_status":
                z = args.get("zone")
                return {b.id: dict(zone=b.zone, fill_pct=round(b.fill * 100, 1), type=b.waste_type, weight_kg=round(b.weight_kg))
                        for b in tw.bins.values() if not z or b.zone == z}
            if name == "predict_waste_overflow":
                b = tw.bins.get(args.get("bin_id", ""))
                return self.ml.forecast_bin(b, tw.t, tw.weather) if b else dict(error="unknown bin_id")
            if name == "get_energy_status":
                return dict(kpis=r["energy"]["kpis"], issues=[self._brief(f) for f in r["energy"]["findings"]])
            if name == "predict_energy_demand":
                fc = self.ml.forecast_energy(tw.t, tw.load_kw + tw.ev_kw, tw.weather)
                net = [f["demand_kw"] - f["solar_kw"] for f in fc]
                return dict(hourly=[dict(t=f["time"][11:16], demand_kw=round(f["demand_kw"]), solar_kw=round(f["solar_kw"])) for f in fc],
                            expected_peak_kw=round(max(net)), expected_cost=r["energy"]["kpis"]["expected_cost_24h"],
                            expected_co2_kg=r["energy"]["kpis"]["expected_co2_24h_kg"], validation=self.ml.metrics.get("energy_demand"))
            if name == "detect_infrastructure_failure":
                return dict(issues=[self._brief(f) for f in r["infrastructure"]["findings"]], kpis=r["infrastructure"]["kpis"])
            if name == "predict_maintenance":
                return dict(priority_queue=r["infrastructure"]["extra"]["priority_queue"], validation=self.ml.metrics.get("infra_failure"))
            if name == "optimize_route":
                if args.get("road_id") in tw.roads:
                    return dict(road=args["road_id"], alternatives=[dict(path=p, share=round(s, 2)) for p, s in opt.alternative_paths(tw, args["road_id"])])
                if args.get("from_node") in tw.nodes and args.get("to_node") in tw.nodes:
                    p, tt, km = tw.route(args["from_node"], args["to_node"], emergency=True)
                    return dict(path=p, eta_min=round(tt, 1), km=round(km, 2))
                return dict(error="provide road_id or from_node/to_node")
            if name == "optimize_signal_timing":
                n = args.get("node_id")
                return opt.webster_plan(tw, n) if n in tw.nodes else dict(error="unknown node_id")
            if name == "optimize_resources":
                return dict(units={u.id: dict(kind=u.kind, status=u.status, node=u.node) for u in tw.units.values()},
                            trucks={t.id: dict(status=t.status, stops=len(t.stops)) for t in tw.trucks.values()})
            if name == "calculate_cost":
                return opt.calculate_cost(tw)
            if name == "calculate_carbon":
                return opt.calculate_carbon(tw)
            if name == "simulate_action":
                pid = args.get("proposal_id", "")
                for f in self.issues.values():
                    for p in f["proposals"]:
                        if p["id"] == pid:
                            m = p["measure"]
                            if m["metric"] in ("none", "asset_status", "incident_response", "unit_arrival", "coverage", "contamination"):
                                return dict(proposal_id=pid, expected=p.get("expected"), note="effect verified by status checks, not a numeric metric")
                            fake = dict(type=p["type"], params=json.loads(json.dumps(p["params"], default=list)))
                            res = opt.simulate(tw, lambda t: self._apply_to(t, fake), max(1, m["ticks"]), lambda t: self.measure(m, t))
                            return dict(proposal_id=pid, metric=m["metric"], horizon_min=m["ticks"] * tw.dt_min, **res)
                return dict(error="unknown proposal_id")
            if name == "verify_action":
                a = self.actions.get(args.get("action_id", ""))
                return dict(status=a["status"], verification=a.get("verification")) if a else dict(error="unknown action_id")
            return dict(error=f"unknown tool {name}")

    def _apply_to(self, twin, a):
        real = self.tw
        self.tw = twin
        try:
            self.apply(a)
        finally:
            self.tw = real

    @staticmethod
    def _brief(f):
        return dict(key=f["key"], title=f["title"], detail=f["detail"], priority=f.get("priority"), category=f.get("category"))

    # =============================================================== WHAT-IF SIMULATION
    SCENARIOS = {
        "population_increase": dict(label="Population +20%", params=dict(population_mult=1.2)),
        "festival": dict(label="City festival (CBD)", params=dict(zone="Z2")),
        "road_closure": dict(label="Bridge closure (3rd/4th St Bridge)", params=dict(road=None)),
        "extreme_weather": dict(label="Heatwave + storm", params=dict(heat=7, storm=0.8)),
        "ev_adoption": dict(label="EV adoption ×1.8", params=dict(ev_mult=1.8)),
        "waste_increase": dict(label="Waste generation +40%", params=dict(waste_mult=1.4)),
        "energy_demand_increase": dict(label="Energy demand +25%", params=dict(energy_mult=1.25)),
        "infrastructure_failure": dict(label="Transformer failure (CBD)", params=dict(asset="TX-1")),
        "fleet_reduction": dict(label="Fleet reduction (1 truck, 1 ambulance)", params=dict()),
    }

    def _apply_scenario(self, t, name: str, params: dict, duration_ticks: int = 288):
        sc = t.scenario
        if name == "population_increase":
            sc["population_mult"] = params.get("population_mult", 1.2)
        elif name == "festival":
            t.start_event(params.get("zone", "Z2"), "festival", 3.8, min(duration_ticks, 60))
        elif name == "road_closure":
            rid = params.get("road") or next(e.id for e in t.roads.values() if e.is_bridge and e.u.endswith("3"))
            t.close_road(rid, "what-if closure")
        elif name == "extreme_weather":
            t.weather.heatwave, t.weather.storm = params.get("heat", 7), params.get("storm", 0.8)
            sc["accident_mult"], sc["infra_stress"] = 1.6, 1.8
        elif name == "ev_adoption":
            sc["ev_mult"] = params.get("ev_mult", 1.8)
        elif name == "waste_increase":
            sc["waste_mult"] = params.get("waste_mult", 1.4)
        elif name == "energy_demand_increase":
            sc["energy_mult"] = params.get("energy_mult", 1.25)
        elif name == "infrastructure_failure":
            a = t.assets[params.get("asset", "TX-1")]
            a.fault, a.fault_rate = 1.0, 0.0
            t._fail(a)
        elif name == "fleet_reduction":
            t.trucks.pop("WT-3", None)
            t.fixed_rounds = {k: v for k, v in t.fixed_rounds.items() if k in t.trucks}
            ids = sorted(t.bins)
            t.fixed_rounds = {k: ids[i::len(t.trucks)] for i, k in enumerate(t.trucks)}
            t.units.pop("AMB-2", None)

    def _kpi_run(self, t, ticks: int) -> dict:
        acc = defaultdict(float)
        e0 = dict(t.energy_totals)
        w0, d0 = dict(t.waste_totals), t.traffic_totals["delay_veh_h"]
        inc0 = len(t.incidents)
        peak = 0.0
        fails0 = sum(1 for a in t.assets.values() if a.status == "FAILED")
        dens = 0.0
        gen = 0.0
        for _ in range(ticks):
            fills = {b.id: b.fill for b in t.bins.values()}
            t.step()
            acc["vkt"] += sum(e.flow * e.length_km for e in t.roads.values()) * t.dt_min / 60
            acc["vc_max"] = max(acc["vc_max"], max(e.vc for e in t.roads.values()))
            acc["congested_road_h"] += sum(1 for e in t.roads.values() if e.vc > 0.85) * t.dt_min / 60
            peak = max(peak, t.grid_kw)
            dens = max(dens, max(z.density for z in t.zones.values()))
            for b in t.bins.values():
                dfill = b.fill - fills[b.id]
                if dfill > 0:
                    gen += dfill * b.capacity_l * {"general": 0.11, "recycling": 0.06, "organic": 0.25}[b.waste_type]
        kwh = t.energy_totals["kwh"] - e0["kwh"]
        fuel = t.waste_totals["fuel_l"] - w0["fuel_l"]
        delay_h = t.traffic_totals["delay_veh_h"] - d0
        co2 = (t.energy_totals["co2"] - e0["co2"]) + fuel * settings.diesel_kgco2_per_l + delay_h * settings.car_idle_l_per_h * settings.petrol_kgco2_per_l \
            + (t.waste_totals["landfill_kg"] - w0["landfill_kg"]) * settings.landfill_kgco2e_per_kg
        return dict(vehicle_km=round(acc["vkt"]), max_vc=round(acc["vc_max"], 2), congested_road_hours=round(acc["congested_road_h"], 1),
                    delay_vehicle_hours=round(delay_h), waste_generated_kg=round(gen),
                    overflow_bin_hours=round((t.waste_totals["overflow_bin_ticks"] - w0["overflow_bin_ticks"]) * t.dt_min / 60, 1),
                    energy_kwh=round(kwh), peak_kw=round(peak), energy_cost=round(t.energy_totals["cost"] - e0["cost"]),
                    max_crowd_density=round(dens, 2), new_incidents=len(t.incidents) - inc0 + 0,
                    asset_failures=sum(1 for a in t.assets.values() if a.status == "FAILED") - fails0,
                    carbon_kgco2e=round(co2))

    def run_scenario(self, name: str, params: dict | None = None, hours: int = 24) -> dict:
        if name not in self.SCENARIOS:
            return dict(error=f"unknown scenario; choose one of {list(self.SCENARIOS)}")
        params = {**self.SCENARIOS[name]["params"], **(params or {})}
        with self.lock:
            base, scen = self.tw.clone(), self.tw.clone()
        ticks = int(hours * 60 / self.tw.dt_min)
        self._apply_scenario(scen, name, params, ticks)
        b, s = self._kpi_run(base, ticks), self._kpi_run(scen, ticks)
        delta = {k: dict(baseline=b[k], scenario=s[k], change=round(s[k] - b[k], 2),
                         change_pct=round((s[k] - b[k]) / abs(b[k]) * 100, 1) if b[k] else None) for k in b}
        extra_trucks = max(0, math.ceil((s["waste_generated_kg"] - b["waste_generated_kg"]) / 9000 / max(hours / 24, 0.1)))
        extra_crews = max(0, s["asset_failures"] - b["asset_failures"])
        batt_kwh = max(0, s["peak_kw"] - b["peak_kw"]) * 3
        recs = []
        if delta["congested_road_hours"]["change"] > 2:
            recs.append("Pre-load adaptive signal plans and publish diversion routes before the peak.")
        if delta["overflow_bin_hours"]["change"] > 2 or extra_trucks:
            recs.append(f"Add {max(1, extra_trucks)} collection run(s) per day using predictive routing.")
        if delta["peak_kw"]["change"] > 150:
            recs.append(f"Reserve ≈{batt_kwh:,.0f} kWh of storage / flexible load for peak shaving.")
        if delta["asset_failures"]["change"] > 0:
            recs.append(f"Stand by {extra_crews} extra maintenance crew(s); prioritise critical assets.")
        if delta["max_crowd_density"]["change"] > 0.3:
            recs.append("Deploy crowd stewards and protect emergency access routes in affected zones.")
        result = dict(scenario=name, label=self.SCENARIOS[name]["label"], params=params, horizon_hours=hours, comparison=delta,
                      additional_resources=dict(collection_trucks=extra_trucks, maintenance_crews=extra_crews, storage_kwh=round(batt_kwh)),
                      recommendations=recs or ["No significant additional resources required."],
                      method="Paired digital-twin simulation (baseline vs scenario, identical random seed, traditional policies, no AI actions)")
        return result

    def inject(self, name: str, params: dict | None = None) -> dict:
        if name not in self.SCENARIOS:
            return dict(ok=False, error="unknown scenario")
        with self.lock:
            p = {**self.SCENARIOS[name]["params"], **(params or {})}
            self._apply_scenario(self.tw, name, p)
            self.log("OBSERVE", "City Orchestrator", f"Live scenario injected: {self.SCENARIOS[name]['label']}.")
        return dict(ok=True, scenario=name)

    def clear_scenario(self):
        with self.lock:
            t = self.tw
            t.scenario.clear()
            t.weather.heatwave = t.weather.storm = 0.0
            for e in t.roads.values():
                if e.closed and e.closed_reason in ("what-if closure", "precautionary (authorised)"):
                    t.reopen_road(e.id)
            self.log("OBSERVE", "City Orchestrator", "Live scenario modifiers cleared.")

    # =============================================================== SNAPSHOT
    def metrics(self) -> dict:
        det = {}
        for g, d in self.det.items():
            p = d["tp"] / (d["tp"] + d["fp"]) if d["tp"] + d["fp"] else None
            r = d["tp"] / (d["tp"] + d["fn"]) if d["tp"] + d["fn"] else None
            det[g] = dict(**d, precision=None if p is None else round(p, 3), recall=None if r is None else round(r, 3),
                          f1=None if not p or not r else round(2 * p * r / (p + r), 3))
        eff = {k: dict(v) for k, v in self.effectiveness.items()}

        def clean(x):
            if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
                return None
            if isinstance(x, dict):
                return {k: clean(v) for k, v in x.items()}
            return x
        return clean(dict(offline_validation=self.ml.metrics, live_forecast_errors=self.ml.live_metrics(), detection=det,
                          action_effectiveness=eff, thresholds=self.thresholds, model_version=self.ml.version, trained_at=self.ml.trained_at,
                          retraining=self.ml.retraining, adaptation=[{k: v for k, v in e.items() if k != "_logged"} for e in list(self.ml.adaptation_log)[-10:]],
                          processing=self.proc.summary()["totals"], cycle_ms=round(self.last_cycle_ms, 1)))

    def snapshot(self, full: bool = True) -> dict:
        with self.lock:
            tw = self.tw
            issues = sorted(self.issues.values(), key=lambda f: -f["priority"])
            acts = [self.actions[a] for a in list(self.action_order)[:60] if a in self.actions]
            pend = [a for a in self.actions.values() if a["status"] == "PENDING_APPROVAL"]
            h = self.health_hist[-1] if self.health_hist else None
            agents = {}
            for d, r in self.reports.items():
                agents[d] = dict(agent=r["agent"], subtitle=r["subtitle"], level=r["level"], label=r["label"], health=r["health"], headline=r["headline"],
                                 kpis=r["kpis"], extra=r["extra"], n_findings=len(r["findings"]))
            crit = sum(1 for f in issues if f["category"] == "CRITICAL")
            active = sum(1 for a in self.actions.values() if a["status"] == "ACTIVE")
            orch_head = (f"City Health {h['score']:.0f}/100 — {crit} critical issue{'s' * (crit != 1)} · {active} action{'s' * (active != 1)} in progress · "
                         f"{len(pend)} awaiting authorisation." if h else "Initialising…")
            snap = dict(
                city=tw.name, currency=settings.currency, tick=tw.tick, time=tw.t.isoformat(timespec="minutes"), weather=tw.weather.snapshot(), running=self.running, speed=self.speed,
                health=h, health_trend=[dict(t=x["ts"][11:16], s=x["score"]) for x in list(self.health_hist)[-144::2]],
                state=self.state_engine, agents=agents, orchestrator=dict(headline=orch_head, llm=self.llm.status(), llm_summary=self.llm_summary),
                issues=[self._issue_view(f) for f in issues[:60]], approvals=[self._action_view(a) for a in pend],
                actions=[self._action_view(a) for a in acts], plans=list(self.plans)[:10],
                activity=list(self.activity)[-120:], messages=self.bus.messages[-40:], events=list(self.events)[-40:],
                live=tw.live(), cycle_ms=round(self.last_cycle_ms, 1),
            )
            if full:
                snap["metrics"] = self.metrics()
            return snap

    @staticmethod
    def _issue_view(f):
        return {k: f.get(k) for k in ("key", "domain", "kind", "entity", "entity_name", "zone", "x", "y", "title", "detail", "severity", "probability",
                                      "impact", "urgency", "priority", "category", "first_seen", "root_causes", "forecast", "evidence", "status",
                                      "actions", "plan", "llm")} | dict(proposals=[dict(id=p["id"], title=p["title"], type=p["type"], risk=p["risk"],
                                                                                         expected=p.get("expected")) for p in f["proposals"]])

    @staticmethod
    def _action_view(a):
        v = {k: a.get(k) for k in ("id", "issue", "issue_title", "created", "domain", "type", "target", "title", "expected", "risk", "safety",
                                   "execution_mode", "status", "verification", "decision", "result", "before", "after", "counterfactual",
                                   "authorized_by", "tool")}
        if v.get("result"):
            v["result"] = {k: x for k, x in v["result"].items() if k in ("summary", "eta_min", "work_order", "km")}
        return v
