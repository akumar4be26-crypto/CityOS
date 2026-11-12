"""
Specialised domain agents. Each runs OBSERVE → ANALYZE → DETECT → INVESTIGATE → PREDICT → PLAN for its
domain and returns a DomainReport (findings with root causes, forecasts and candidate actions).
Agents communicate through a message bus; the City Orchestrator coordinates their outputs.

All numbers come from processed sensor data, ML models and optimisation tools – never from the LLM.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from . import optimize as opt
from .config import settings
from .ml import VIB_NORMAL, TEMP_OFFSET

LEVELS = ["NORMAL", "WATCH", "WARNING", "CRITICAL"]
CROWD_T = (0.9, 1.3, 1.8)  # p/m² thresholds WATCH / WARNING / CRITICAL


def clip01(v) -> float:
    return float(min(1.0, max(0.0, v)))


@dataclass
class Bus:
    messages: list = field(default_factory=list)

    def publish(self, tick: int, sender: str, to: str, kind: str, text: str, payload: dict | None = None):
        m = dict(tick=tick, sender=sender, to=to, kind=kind, text=text, payload=payload or {})
        self.messages.append(m)
        if len(self.messages) > 400:
            self.messages = self.messages[-300:]
        return m

    def inbox(self, tick: int, agent: str, kinds: tuple = ()) -> list[dict]:
        return [m for m in self.messages if m["tick"] == tick and m["to"] in (agent, "all") and (not kinds or m["kind"] in kinds)]


@dataclass
class Ctx:
    tw: object
    ml: object
    proc: object
    bus: Bus
    tick: int
    thresholds: dict
    busy: set                    # finding keys with an active / pending action
    effectiveness: dict          # action type → {"success": n, "total": n}
    memory: dict                 # per-agent scratch memory across ticks


def finding(domain, kind, entity, title, detail, S, P, I, U, *, name=None, zone=None, x=None, y=None, evidence=None,
            root_causes=None, forecast=None, proposals=None) -> dict:
    return dict(key=f"{domain}:{kind}:{entity}", domain=domain, kind=kind, entity=entity, entity_name=name or entity, zone=zone,
                x=x, y=y, title=title, detail=detail, severity=clip01(S), probability=clip01(P), impact=clip01(I), urgency=clip01(U),
                evidence=evidence or {}, root_causes=root_causes or [], forecast=forecast or {}, proposals=proposals or [])


def proposal(type_, domain, target, title, params, expected, measure, risk="LOW", tool=None) -> dict:
    return dict(type=type_, domain=domain, target=target, title=title, params=params, expected=expected, measure=measure, risk=risk,
                tool=tool or type_)


def rank_causes(cands: list[tuple[str, float, str]]) -> list[dict]:
    cands = [(f, s, e) for f, s, e in cands if s > 0.05]
    tot = sum(s for _, s, _ in cands) or 1
    return [dict(factor=f, contribution=round(s / tot * 100), evidence=e) for f, s, e in sorted(cands, key=lambda c: -c[1])]


class DomainAgent:
    name = "Agent"
    domain = "generic"
    title = ""

    def report(self, level, label, health, headline, findings, kpis, extra=None) -> dict:
        return dict(agent=self.name, domain=self.domain, subtitle=self.title, level=level, label=label, health=round(health, 1),
                    headline=headline, findings=findings, kpis=kpis, extra=extra or {})


# ============================================================================================
class TrafficAgent(DomainAgent):
    name, domain, title = "Traffic Agent", "traffic", "Mobility Intelligence"

    def run(self, c: Ctx) -> dict:
        tw, ml, P = c.tw, c.ml, c.proc
        now, wx = tw.t, tw.weather
        dt = tw.dt_min
        we, hr = int(now.weekday() >= 5), now.hour
        thr = c.thresholds["congestion_vc"]
        cap_w = 1 - min(0.18, 0.02 * wx.rain)
        F: list[dict] = []
        obs = {}
        for e in tw.roads.values():
            es = P.entity_state.get(e.id, {})
            cnt = es.get("vehicle_count")
            flow = cnt * 60 / dt if cnt is not None else e.flow
            speed = es.get("speed", e.speed)
            cap = e.capacity * e.cap_factor * cap_w
            obs[e.id] = dict(flow=flow, speed=speed, vc=0.0 if e.closed else flow / max(cap, 1), occ=es.get("occupancy", 0),
                             queue=es.get("queue_length", 0), suspect=any(k.startswith(f"S-{e.id}-") for k in P.sensor_issues))
        # ---- PREDICT (batched ML forecast)
        horizons = [15, 30, 60]
        fc = ml.predict_flows(list(tw.roads.values()), now, horizons, wx)
        if c.tick % 6 == 0:
            for e in tw.roads.values():
                if not e.closed:
                    ml.record("traffic_flow", e.id, c.tick + tw.ticks_for(30), fc[(e.id, 30)], 30, now_tick=c.tick)
        forecasts = {}
        for e in tw.roads.values():
            cap = e.capacity * e.cap_factor * cap_w
            vcs = {h: (0 if e.closed else fc[(e.id, h)] / max(cap, 1)) for h in horizons}
            probs = {h: ml.congestion_probability(v) for h, v in vcs.items()}
            onset = next((h for h in horizons if vcs[h] > thr), None)
            tt0 = e.length_km / e.free_speed * 60
            tt60 = tt0 * (1 + 0.15 * min(vcs[60], 1.6) ** 4)
            forecasts[e.id] = dict(vc=vcs, probability=probs, onset_min=onset, flow_60=fc[(e.id, 60)],
                                   travel_time_increase_pct=round((tt60 / max(e.travel_min or tt0, 0.01) - 1) * 100, 1))
        busy = c.busy
        # ---- closures (from infrastructure bus or road status)
        for e in tw.roads.values():
            if not e.closed:
                continue
            key = f"traffic:closure_impact:{e.id}"
            mem = c.memory.setdefault("closure_impact", {})
            if e.id not in mem or c.tick - mem[e.id]["tick"] > 12:
                org = opt.predict_closure_impact(tw, e.id, ai_diversion=False)
                ai = opt.predict_closure_impact(tw, e.id, ai_diversion=True)
                mem[e.id] = dict(tick=c.tick, org=org, ai=ai)
            org, ai = mem[e.id]["org"], mem[e.id]["ai"]
            props = []
            if e.id not in tw.diversions:
                paths = opt.alternative_paths(tw, e.id)
                props.append(proposal("traffic_diversion", "traffic", e.id, f"Coordinated diversion around {e.name}",
                                      dict(road=e.id, paths=paths), dict(metric="peak v/c near closure", before=org["near_max_vc"],
                                                                          after=ai["near_max_vc"],
                                                                          improvement_pct=round((org["near_max_vc"] - ai["near_max_vc"]) / max(org["near_max_vc"], 1e-6) * 100, 1)),
                                      dict(metric="max_vc", entities=org["near_roads"], direction="decrease", ticks=settings.verification_ticks), "MEDIUM",
                                      tool="optimize_route"))
            ux, uy = tw.nodes[e.u].x, tw.nodes[e.u].y
            F.append(finding("traffic", "closure_impact", e.id, f"Road closure — {e.name}",
                             f"Closure ({e.closed_reason}) predicted to raise congestion on adjacent roads by {org['congestion_increase_pct']}%.",
                             0.75, 1.0, clip01(0.4 + org["max_vc_after"] / 2), 0.9, name=e.name, zone=e.zone, x=(ux + tw.nodes[e.v].x) / 2,
                             y=(uy + tw.nodes[e.v].y) / 2, evidence=dict(organic=org, ai_diversion=ai),
                             forecast=dict(congestion_increase_pct=org["congestion_increase_pct"]), proposals=props))
            if key not in busy and c.memory.setdefault("closure_published", set()).isdisjoint({e.id}):
                c.memory["closure_published"].add(e.id)
                c.bus.publish(c.tick, self.name, "orchestrator", "congestion_forecast",
                              f"Closure of {e.name} → {org['congestion_increase_pct']}% congestion increase predicted", dict(road=e.id, **org))
        # ---- congestion, risk, abnormal flow, incidents
        sig_ops = {p["node"]: p for p in opt.signal_opportunities(tw, min_gain_pct=8)}
        open_ids = [e.id for e in tw.roads.values() if not e.closed]
        if_all = dict(zip(open_ids, ml.if_scores("traffic", [[obs[i]["flow"] / tw.roads[i].base_flow, obs[i]["speed"] / 50, obs[i]["occ"] / 100,
                                                              obs[i]["queue"] / 300, obs[i]["vc"]] for i in open_ids])))
        for e in tw.roads.values():
            if e.closed:
                continue
            o, f = obs[e.id], forecasts[e.id]
            ex = ml.expected("traffic", (e.id, we, hr)) or {}
            mu, sd = ex.get("flow_m", e.flow), ex.get("flow_s", 0) or 0
            z = (o["flow"] - mu) / max(sd, 0.08 * mu, 1)
            ifs = float(if_all[e.id])
            ux, uy = (tw.nodes[e.u].x + tw.nodes[e.v].x) / 2, (tw.nodes[e.u].y + tw.nodes[e.v].y) / 2
            causes = self._root_causes(tw, e, o, mu, z, sig_ops)
            props = self._proposals(tw, e, sig_ops, c)
            if o["vc"] > thr:
                F.append(finding("traffic", "congestion", e.id, f"Congestion on {e.name}",
                                 f"Volume/capacity {o['vc']:.2f} (threshold {thr:.2f}); speed {o['speed']:.0f} km/h; queue {o['queue']:.0f} m.",
                                 (o["vc"] - 0.6) / 0.6, 1.0, 0.35 + o["flow"] / 4000, 0.6 + 0.4 * f["probability"][30],
                                 name=e.name, zone=e.zone, x=ux, y=uy, root_causes=causes,
                                 evidence=dict(vc=round(o["vc"], 3), flow=round(o["flow"]), expected_flow=round(mu), z=round(z, 2), if_score=round(ifs, 3),
                                               speed=round(o["speed"], 1)),
                                 forecast=dict(vc_30=round(f["vc"][30], 3), vc_60=round(f["vc"][60], 3), p_congestion_30=round(f["probability"][30], 2),
                                               duration_min=60 if f["vc"][60] > thr else (30 if f["vc"][30] > thr else 15),
                                               travel_time_increase_pct=f["travel_time_increase_pct"]), proposals=props))
            elif f["onset_min"] and f["probability"][f["onset_min"]] > 0.6 and o["vc"] > 0.55:
                F.append(finding("traffic", "congestion_risk", e.id, f"Congestion predicted on {e.name} in {f['onset_min']} min",
                                 f"Forecast v/c {f['vc'][f['onset_min']]:.2f} at +{f['onset_min']} min (p={f['probability'][f['onset_min']]:.0%}).",
                                 0.45, f["probability"][f["onset_min"]], 0.3 + o["flow"] / 5000, 0.5, name=e.name, zone=e.zone, x=ux, y=uy,
                                 root_causes=causes, forecast=dict(onset_min=f["onset_min"], vc=f["vc"], probability=f["probability"],
                                                                   travel_time_increase_pct=f["travel_time_increase_pct"]),
                                 proposals=[p for p in props if p["type"] == "optimize_signal_timing"]))
            if abs(z) > c.thresholds["z"] and ifs > 0 and e.cap_reason is None:
                kind = "sensor_anomaly" if o["suspect"] else "abnormal_traffic"
                F.append(finding("traffic", kind, e.id, f"{'Suspect sensor' if o['suspect'] else 'Abnormal traffic pattern'} on {e.name}",
                                 f"Flow {o['flow']:.0f} veh/h vs expected {mu:.0f} (z={z:+.1f}); Isolation Forest score {ifs:+.2f}.",
                                 0.35, clip01(abs(z) / 6), 0.3, 0.4, name=e.name, zone=e.zone, x=ux, y=uy, root_causes=causes,
                                 evidence=dict(z=round(z, 2), if_score=round(ifs, 3), expected=round(mu), observed=round(o["flow"]))))
        for inc in tw.incidents.values():
            if inc.type == "traffic_collision" and inc.status != "CLEARED" and inc.road:
                e = tw.roads[inc.road]
                props = [p for p in self._proposals(tw, e, sig_ops, c, force_reroute=True) if p["type"] == "reroute_traffic"]
                F.append(finding("traffic", "incident", inc.id, f"Collision — {e.name}", f"Severity {inc.severity}/4; lane capacity reduced to {e.cap_factor:.0%}.",
                                 inc.severity / 4, 1.0, 0.35 + obs[e.id]["flow"] / 4000, 0.8, name=e.name, zone=inc.zone, x=inc.x, y=inc.y,
                                 evidence=dict(severity=inc.severity, capacity_factor=e.cap_factor, status=inc.status), proposals=props))
        # ---- bottlenecks + optimization opportunities
        for nid, p in list(sig_ops.items())[:4]:
            n = tw.nodes[nid]
            if f"traffic:signal_opportunity:{nid}" in busy:
                continue
            bott = p["delay_before_s"] > 40
            F.append(finding("traffic", "bottleneck" if bott else "signal_opportunity", nid,
                             f"{'Bottleneck at' if bott else 'Signal optimisation at'} intersection {nid}",
                             f"Webster plan cuts average delay {p['delay_before_s']}s → {p['delay_after_s']}s ({p['improvement_pct']}%) "
                             f"for {p['volume_vph']} veh/h.", 0.5 if bott else 0.25, 0.9, clip01(p["volume_vph"] / 4000), 0.5 if bott else 0.3,
                             name=f"Intersection {nid}", zone=n.zone, x=n.x, y=n.y, evidence=p,
                             proposals=[self._signal_prop(tw, [nid], {nid: p})]))
        # ---- KPIs / level
        open_roads = [e for e in tw.roads.values() if not e.closed]
        avg_delay = tw.avg_intersection_delay()
        congested = sum(1 for e in open_roads if obs[e.id]["vc"] > thr)
        risks = sum(1 for f_ in F if f_["kind"] == "congestion_risk")
        events = sum(1 for f_ in F if f_["kind"] in ("incident", "closure_impact", "congestion", "abnormal_traffic"))
        opps = sum(1 for f_ in F if f_["kind"] in ("signal_opportunity", "bottleneck"))
        mean_vc = float(np.mean([obs[e.id]["vc"] for e in open_roads]))
        health = 100 - 120 * max(0, mean_vc - 0.45) - 6 * congested - 8 * sum(1 for e in tw.roads.values() if e.closed) - 4 * sum(
            1 for f_ in F if f_["kind"] == "incident") - max(0, avg_delay - 20)
        lvl = 3 if congested >= 4 or health < 45 else 2 if congested >= 2 or health < 65 else 1 if congested or events else 0
        label = ["NORMAL", "MODERATE", "HIGH", "SEVERE"][lvl]
        mood = ["operating normally", "busier than usual", "under pressure", "severely congested"][lvl]
        headline = f"City mobility is {mood} — {events} active event{'s' * (events != 1)} · {risks} congestion risk{'s' * (risks != 1)} · {opps} optimization opportunit{'ies' if opps != 1 else 'y'}."
        kpis = dict(avg_intersection_delay_s=round(avg_delay, 1), mean_vc=round(mean_vc, 3), congested_roads=congested,
                    avg_speed_kmh=round(float(np.mean([obs[e.id]["speed"] for e in open_roads])), 1),
                    closed_roads=sum(1 for e in tw.roads.values() if e.closed), active_reroutes=len(tw.reroutes) + len(tw.diversions),
                    ai_signal_plans=sum(1 for n in tw.nodes.values() if n.plan_source != "fixed-time"))
        extra = dict(forecast_top=sorted(([eid, round(f["vc"][60], 3), round(f["probability"][60], 2)] for eid, f in forecasts.items()),
                                         key=lambda r: -r[1])[:8])
        return self.report(LEVELS[lvl], label, max(0, min(100, health)), headline, F, kpis, extra)

    # ---- root cause analysis
    def _root_causes(self, tw, e, o, mu, z, sig_ops) -> list[dict]:
        adj = [x for x in tw.incident_edges[e.u] + tw.incident_edges[e.v] if x != e.id]
        closed_adj = [tw.roads[x].name for x in adj if tw.roads[x].closed]
        acc = e.cap_reason == "collision" or any(tw.roads[x].cap_reason == "collision" for x in adj)
        peak = clip01((tw.traffic_profile(tw.hour) - 0.45) / 0.45)
        ev = e.event_mult - 1
        sig = max((sig_ops.get(n, {}).get("improvement_pct", 0) for n in (e.u, e.v)), default=0)
        diverted = 0.0
        if closed_adj:
            diverted = clip01((o["flow"] - mu) / max(mu, 1))
        cands = [
            ("Road closure / diverted traffic", 0.9 * len(closed_adj) + diverted, f"closed nearby: {', '.join(closed_adj) or 'none'}"),
            ("Accident / capacity reduction", 0.9 if acc else 0, f"capacity factor {e.cap_factor:.0%}"),
            ("Peak-hour demand", 0.6 * peak, f"time-of-day demand index {tw.traffic_profile(tw.hour):.2f}"),
            ("Event traffic", clip01(ev * 3), f"event demand multiplier {e.event_mult:.2f}"),
            ("Weather (rain)", clip01(tw.weather.rain / 10), f"rainfall {tw.weather.rain:.1f} mm/h"),
            ("Signal timing", clip01(sig / 40), f"Webster improvement potential {sig:.0f}%"),
            ("Above-normal demand", clip01(z / 4) if z > 1 else 0, f"flow z-score {z:+.1f} vs history"),
        ]
        return rank_causes(cands)

    def _signal_prop(self, tw, nodes, plans) -> dict:
        before = np.mean([plans[n]["delay_before_s"] for n in nodes])
        after = np.mean([plans[n]["delay_after_s"] for n in nodes])
        return proposal("optimize_signal_timing", "traffic", ",".join(nodes), f"Switch {', '.join(nodes)} to AI adaptive signal timing",
                        dict(plans={n: plans[n]["proposed"] for n in nodes}),
                        dict(metric="avg intersection delay (s/veh)", before=round(before, 1), after=round(after, 1),
                             improvement_pct=round((before - after) / max(before, 1e-6) * 100, 1)),
                        dict(metric="node_delay", entities=nodes, direction="decrease", ticks=settings.verification_ticks), "LOW",
                        tool="optimize_signal_timing")

    def _proposals(self, tw, e, sig_ops, c, force_reroute=False) -> list[dict]:
        props = []
        nodes = [n for n in (e.u, e.v) if n in sig_ops]
        if nodes:
            props.append(self._signal_prop(tw, nodes, sig_ops))
        if (force_reroute or e.vc > c.thresholds["congestion_vc"]) and e.id not in tw.reroutes:
            cache = c.memory.setdefault("reroute_cache", {})
            hit = cache.get(e.id)
            if hit and c.tick - hit["tick"] < 4:
                if hit["prop"]:
                    props.append(hit["prop"])
                return props
            paths = opt.alternative_paths(tw, e.id, k=2, max_vc=0.7)
            if paths:
                frac = 0.25 if e.vc > 1 else 0.18
                sim = tw.clone()
                sim.reroutes[e.id] = dict(fraction=frac, paths=paths, until=sim.tick + 99)
                sim._traffic()
                if max(sim.roads[x].vc for p, _ in paths for x in p) > 0.88 or sim.roads[e.id].vc >= e.vc:
                    cache[e.id] = dict(tick=c.tick, prop=None)
                    return props
                props.append(proposal("reroute_traffic", "traffic", e.id, f"Recommend alternate routes around {e.name} ({int(frac * 100)}% of trips)",
                                      dict(road=e.id, fraction=frac, paths=paths, duration_ticks=12),
                                      dict(metric="v/c on road", before=round(e.vc, 3), after=round(sim.roads[e.id].vc, 3),
                                           improvement_pct=round((e.vc - sim.roads[e.id].vc) / max(e.vc, 1e-6) * 100, 1),
                                           max_alt_vc=round(max(sim.roads[x].vc for p, _ in paths for x in p), 3)),
                                      dict(metric="road_vc", entities=[e.id], direction="decrease", ticks=settings.verification_ticks),
                                      "MEDIUM", tool="optimize_route"))
                cache[e.id] = dict(tick=c.tick, prop=props[-1])
        return props


# ============================================================================================
class SafetyAgent(DomainAgent):
    name, domain, title = "Safety Agent", "safety", "Public Safety Intelligence"

    def run(self, c: Ctx) -> dict:
        tw, ml, P = c.tw, c.ml, c.proc
        now = tw.t
        we, hr = int(now.weekday() >= 5), now.hour
        night = hr >= 22 or hr < 5
        F = []
        zone_levels = {}
        zfeat = {}
        for z in tw.zones.values():
            d = P.get(z.id, "crowd_density", z.density)
            ex = (ml.expected("safety", (z.id, we, hr)) or {}).get("m", d)
            zfeat[z.id] = (d, ex, d / max(ex, 0.02))
        zif = dict(zip(zfeat, ml.if_scores("safety", [[r, int(night), d] for d, _, r in zfeat.values()])))
        for z in tw.zones.values():
            d, ex, ratio = zfeat[z.id]
            st = P.stats.get(f"S-{z.id}-crowd_density")
            d_diff = (st.hist[-1] - st.hist[-4]) if st and len(st.hist) >= 4 else 0.0
            ifs = float(zif[z.id])
            risk = ml.crowding_risk(z.kind, now, d, z.event is not None, tw.weather.rain, d_diff)
            lvl = 3 if d >= CROWD_T[2] else 2 if d >= CROWD_T[1] else 1 if d >= CROWD_T[0] or risk > 0.6 else 0
            zone_levels[z.id] = dict(level=LEVELS[lvl], density=round(d, 3), expected=round(ex, 3), crowding_risk_1h=round(risk, 2),
                                     dark=z.id in getattr(tw, "dark_zones", set()), event=z.event)
            bx = z.bbox
            cx, cy = (bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2
            if lvl >= 1:
                props = [proposal("safety_alert", "safety", z.id, f"Issue crowding advisory for {z.name}", dict(zone=z.id, message="crowding"),
                                  dict(metric="awareness", note="notify field teams & public channels"),
                                  dict(metric="none", entities=[z.id], direction="none", ticks=1), "LOW", tool="detect_safety_incident")]
                if lvl >= 2 and z.crowd_mgmt_until < tw.tick:
                    sc = c.memory.setdefault("crowd_sim", {})
                    if z.id not in sc or c.tick - sc[z.id][0] >= 4:
                        sc[z.id] = (c.tick, opt.simulate(tw, lambda t: self._crowd_apply(t, z.id), 6, lambda t: t.zones[z.id].density))
                    sim = sc[z.id][1]
                    props.insert(0, proposal("crowd_management", "safety", z.id, f"Deploy stewards & open additional exits — {z.name}",
                                             dict(zone=z.id, factor=0.78, duration_ticks=12),
                                             dict(metric="crowd density (p/m²)", before=round(sim["baseline"], 3), after=round(sim["with_action"], 3),
                                                  improvement_pct=round((sim["baseline"] - sim["with_action"]) / max(sim["baseline"], 1e-6) * 100, 1)),
                                             dict(metric="zone_density", entities=[z.id], direction="decrease", ticks=settings.verification_ticks),
                                             "MEDIUM", tool="simulate_action"))
                F.append(finding("safety", "crowding", z.id, f"Crowding {LEVELS[lvl]} — {z.name}",
                                 f"Density {d:.2f} p/m² (normal {ex:.2f}); crowding risk next hour {risk:.0%}{'; ' + z.event if z.event else ''}.",
                                 [0.2, 0.45, 0.7, 0.95][lvl], max(risk, 0.6 if lvl >= 2 else 0.4), clip01(z.population / 50000), 0.4 + 0.2 * lvl,
                                 name=z.name, zone=z.id, x=cx, y=cy, evidence=dict(density=round(d, 3), expected=round(ex, 3), ratio=round(ratio, 2)),
                                 root_causes=rank_causes([("Scheduled / public event", 0.9 if z.event else 0, z.event or "none"),
                                                          ("Time-of-day pattern", 0.4, f"expected {ex:.2f} p/m² at {hr}:00"),
                                                          ("Unusual gathering", clip01((ratio - 2) / 3), f"{ratio:.1f}× normal")]),
                                 forecast=dict(crowding_risk_1h=round(risk, 2)), proposals=props))
            if ratio > 2.5 and ifs > 0 and not z.event:
                pol = opt.best_unit(tw, tw.nearest_node(cx, cy), ("police",))
                props = [proposal("safety_alert", "safety", z.id, f"Alert control room: abnormal activity in {z.name}", dict(zone=z.id),
                                  dict(metric="awareness"), dict(metric="none", entities=[z.id], direction="none", ticks=1), "LOW")]
                if pol:
                    props.append(proposal("patrol_dispatch", "safety", z.id, f"Send {pol['unit']} to verify (ETA {pol['eta_min']} min)",
                                          dict(unit=pol["unit"], node=tw.nearest_node(cx, cy), zone=z.id),
                                          dict(metric="verification ETA (min)", after=pol["eta_min"]),
                                          dict(metric="unit_arrival", entities=[pol["unit"]], direction="none", ticks=tw.ticks_for(pol["eta_min"]) + 1),
                                          "HIGH", tool="optimize_resources"))
                F.append(finding("safety", "abnormal_activity", z.id, f"Abnormal activity — {z.name}",
                                 f"Density {ratio:.1f}× the {hr}:00 baseline {'at night ' if night else ''}(Isolation Forest {ifs:+.2f}).",
                                 0.55, clip01(0.5 + ifs * 2), clip01(z.population / 60000), 0.6 if night else 0.4, name=z.name, zone=z.id, x=cx, y=cy,
                                 evidence=dict(ratio=round(ratio, 2), if_score=round(ifs, 3), night=night), proposals=props))
            if zone_levels[z.id]["dark"]:
                F.append(finding("safety", "unsafe_zone", z.id, f"Unsafe zone — streetlight outage in {z.name}",
                                 "Lighting below 60% after dark raises incident risk; repair requested from Infrastructure Agent.",
                                 0.45, 0.8, clip01(z.population / 50000), 0.5, name=z.name, zone=z.id, x=cx, y=cy,
                                 proposals=[proposal("safety_alert", "safety", z.id, f"Increase patrol visibility in {z.name}", dict(zone=z.id),
                                                     dict(metric="awareness"), dict(metric="none", entities=[z.id], direction="none", ticks=1), "LOW")]))
        # ---- incidents: Detect → Classify → Locate → Assess → Prioritise → Plan
        for inc in tw.incidents.values():
            if inc.status == "CLEARED":
                continue
            kinds = {"medical": ("ambulance",), "traffic_collision": ("ambulance", "police"), "disturbance": ("police",),
                     "fire": ("fire",)}.get(inc.type, ("police",))
            if inc.type == "traffic_collision" and inc.severity <= 1:
                kinds = ("police",)
            props = []
            plan = None
            if inc.status == "OPEN":
                plan = opt.best_unit(tw, inc.node, kinds)
                if plan:
                    path_edges = tw.path_edges(plan["path"])
                    cong = max((tw.roads[x].vc for x in path_edges), default=0)
                    corridor = cong > 0.7 or inc.severity >= 3
                    props.append(proposal("emergency_dispatch", "safety", inc.id,
                                          f"Dispatch {plan['unit']} to {inc.description}{' with signal-priority corridor' if corridor else ''}",
                                          dict(unit=plan["unit"], incident=inc.id, node=inc.node, corridor=corridor),
                                          dict(metric="response time (min)", before=plan["eta_min"], after=plan["corridor_eta_min"] if corridor else plan["eta_min"],
                                               alternate_route_eta=plan["alternate_eta_min"], improvement_pct=22.0 if corridor else 0.0),
                                          dict(metric="incident_response", entities=[inc.id], direction="none",
                                               ticks=tw.ticks_for(plan["eta_min"]) + 2), "HIGH", tool="optimize_route"))
                else:
                    props.append(proposal("incident_escalation", "safety", inc.id, f"Escalate {inc.id}: no unit available — request mutual aid",
                                          dict(incident=inc.id), dict(metric="awareness"), dict(metric="none", entities=[inc.id], direction="none", ticks=1),
                                          "MEDIUM"))
            rt = plan["eta_min"] if plan else 20
            F.append(finding("safety", "incident", inc.id, f"{inc.description.capitalize()} — {tw.zones[inc.zone].name}",
                             f"Severity {inc.severity}/4 · {inc.people_affected} people nearby · status {inc.status}"
                             + (f" · nearest unit {plan['unit']} ETA {plan['eta_min']} min" if plan else ""),
                             0.25 + inc.severity / 5, 0.95, clip01(0.3 + inc.people_affected / 1500), clip01(0.6 + rt / 15), name=inc.id, zone=inc.zone,
                             x=inc.x, y=inc.y, evidence=dict(type=inc.type, severity=inc.severity, status=inc.status, unit=inc.unit,
                                                             age_min=(tw.tick - inc.start_tick) * tw.dt_min), proposals=props))
        # ---- cross-domain: road closure → emergency routes
        for m in c.bus.inbox(c.tick, self.name, ("road_closure",)):
            rid = m["payload"]["road"]
            base = tw.clone()
            base.roads[rid].closed = False
            cov0, cov1 = opt.coverage(base), opt.coverage(tw)
            worse = {z: round(cov1[z] - cov0[z], 1) for z in cov1 if cov1[z] - cov0[z] > 0.5}
            if worse:
                zmax = max(worse, key=worse.get)
                c.memory.setdefault("route_affected", {})[rid] = dict(tick=c.tick, worse=worse)
                c.bus.publish(c.tick, self.name, "orchestrator", "emergency_route_affected",
                              f"Emergency response to {tw.zones[zmax].name} +{worse[zmax]} min after closure of {tw.roads[rid].name}",
                              dict(road=rid, zones=worse))
        for rid, info in list(c.memory.get("route_affected", {}).items()):
            if not tw.roads[rid].closed:
                c.memory["route_affected"].pop(rid)
                continue
            zmax = max(info["worse"], key=info["worse"].get)
            z = tw.zones[zmax]
            zn = [n for n in tw.nodes.values() if n.zone == zmax]
            plan = opt.best_unit(tw, zn[len(zn) // 2].id, ("ambulance",))
            props = []
            if plan:
                props.append(proposal("emergency_preposition", "safety", zmax,
                                      f"Pre-position {plan['unit']} and protect emergency corridor into {z.name}",
                                      dict(unit=plan["unit"], node=zn[len(zn) // 2].id, zone=zmax),
                                      dict(metric="response time to zone (min)", before=round(opt.coverage(tw)[zmax], 1), after=1.0),
                                      dict(metric="coverage", entities=[zmax], direction="decrease", ticks=settings.verification_ticks), "HIGH",
                                      tool="optimize_resources"))
            F.append(finding("safety", "emergency_route_affected", rid, f"Emergency route affected — {tw.roads[rid].name}",
                             "Response times increase: " + ", ".join(f"{tw.zones[k].name} +{v} min" for k, v in info["worse"].items()),
                             0.7, 0.9, clip01(sum(tw.zones[k].population for k in info["worse"]) / 60000), 0.8, name=tw.roads[rid].name,
                             zone=zmax, x=(tw.nodes[tw.roads[rid].u].x + tw.nodes[tw.roads[rid].v].x) / 2,
                             y=(tw.nodes[tw.roads[rid].u].y + tw.nodes[tw.roads[rid].v].y) / 2, evidence=info, proposals=props))
        # hot bins → fire risk
        for m in c.bus.inbox(c.tick, self.name, ("fire_risk",)):
            b = tw.bins[m["payload"]["bin"]]
            F.append(finding("safety", "fire_risk", b.id, f"Fire risk — bin {b.id}", m["text"], 0.6, 0.6, 0.3, 0.7, zone=b.zone, x=b.x, y=b.y,
                             proposals=[proposal("safety_alert", "safety", b.id, "Alert fire service for inspection", dict(bin=b.id),
                                                 dict(metric="awareness"), dict(metric="none", entities=[b.id], direction="none", ticks=1), "LOW")]))
        open_inc = [i for i in tw.incidents.values() if i.status != "CLEARED"]
        crowd = sum(1 for f_ in F if f_["kind"] == "crowding")
        unsafe = sum(1 for f_ in F if f_["kind"] in ("unsafe_zone", "abnormal_activity"))
        worst = max([LEVELS.index(v["level"]) for v in zone_levels.values()] + [min(3, max([i.severity - 1 for i in open_inc], default=0))])
        health = 100 - 7 * len(open_inc) - 4 * sum(i.severity for i in open_inc) - 10 * crowd - 8 * unsafe
        mood = ["stable", "being watched closely", "under elevated risk", "in a critical state"][worst]
        headline = (f"Public safety is {mood} — {len(open_inc)} open incident{'s' * (len(open_inc) != 1)} · {crowd} crowding alert"
                    f"{'s' * (crowd != 1)} · {unsafe} unsafe-zone signal{'s' * (unsafe != 1)}.")
        if "cov" not in c.memory or c.tick - c.memory["cov"][0] >= 3:
            c.memory["cov"] = (c.tick, opt.coverage(tw))
        cov = c.memory["cov"][1]
        kpis = dict(open_incidents=len(open_inc), units_available=sum(1 for u in tw.units.values() if u.kind != "crew" and u.status == "available"),
                    max_density=round(max(v["density"] for v in zone_levels.values()), 2),
                    avg_response_coverage_min=round(float(np.mean([v for v in cov.values() if v < 1e9])), 1))
        return self.report(LEVELS[worst], LEVELS[worst], max(0, min(100, health)), headline, F, kpis,
                           dict(zones=zone_levels, coverage={k: (round(v, 1) if v < 1e9 else None) for k, v in cov.items()}))

    @staticmethod
    def _crowd_apply(t, zid):
        t.zones[zid].crowd_factor, t.zones[zid].crowd_mgmt_until = 0.78, t.tick + 12


# ============================================================================================
class WasteAgent(DomainAgent):
    name, domain, title = "Waste Agent", "waste", "Waste & Collection Intelligence"

    def run(self, c: Ctx) -> dict:
        tw, ml, P = c.tw, c.ml, c.proc
        mem = c.memory
        if c.tick % 3 == 0 or "bin_fc" not in mem:
            mem["bin_fc"] = {b.id: ml.forecast_bin(b, tw.t, tw.weather, tw.zones[b.zone].event_mult if tw.zones[b.zone].event else 1.0)
                             for b in tw.bins.values()}
            if c.tick % 12 == 0:
                for b in tw.bins.values():
                    f2 = mem["bin_fc"][b.id]["trajectory"].get("+2h")
                    if f2 is not None and b.fill < 0.95:
                        ml.record("waste_fill", b.id, c.tick + tw.ticks_for(120), f2, 120, now_tick=c.tick)
        fc = mem["bin_fc"]
        F = []
        cand = []
        in_route = {s[0] for t in tw.trucks.values() for s in t.stops}
        for b in tw.bins.values():
            f = fc[b.id]
            fill = P.get(b.id, "fill_level", b.fill * 100) / 100
            if b.id in in_route:
                continue
            if f["overflow_probability"] >= 0.5 or fill >= c.thresholds["overflow_fill"]:
                over = f["overflow_hours"]
                cand.append(dict(bin=b.id, priority=f["overflow_probability"] + fill, weight=b.weight_kg + 60))
                F.append(finding("waste", "overflow_risk", b.id, f"{'Overflowing' if fill >= 1 else 'Overflow risk'} — bin {b.id}",
                                 f"Fill {fill:.0%}; predicted {', '.join(f'{k} → {min(v, 1.2):.0%}' for k, v in list(f['trajectory'].items())[:4])}; "
                                 f"overflow {'now' if over == 0 else (f'in ~{over:g} h' if over else 'beyond 12 h')}.",
                                 clip01(fill), f["overflow_probability"], clip01(tw.zones[b.zone].population / 50000 + (0.2 if b.waste_type == "organic" else 0)),
                                 1.0 if (over is not None and over <= 2) else 0.6, name=f"{b.id} ({b.waste_type})", zone=b.zone, x=b.x, y=b.y,
                                 forecast=f, evidence=dict(fill=round(fill, 3), weight_kg=round(b.weight_kg))))
            # illegal dumping: sudden mass + fill jump without collection
            fs, ws = P.stats.get(f"S-{b.id}-fill_level"), P.stats.get(f"S-{b.id}-weight")
            if fs and ws and len(fs.hist) >= 2 and len(ws.hist) >= 2:
                jf, jw = fs.hist[-1] - fs.hist[-2], ws.hist[-1] - ws.hist[-2]
                if jf > 22 and jw > 150:
                    mem.setdefault("dumping", {})[b.id] = dict(tick=c.tick, jump_fill=round(jf, 1), jump_kg=round(jw))
            dmp = mem.get("dumping", {}).get(b.id)
            if dmp and c.tick - dmp["tick"] <= 12:
                F.append(finding("waste", "illegal_dumping", b.id, f"Suspected illegal dumping at {b.id}",
                                 f"+{dmp['jump_kg']} kg and +{dmp['jump_fill']} pts fill in one interval — inconsistent with normal deposit rate.",
                                 0.5, 0.75, 0.35, 0.5, zone=b.zone, x=b.x, y=b.y, evidence=dmp,
                                 proposals=[proposal("inspection_request", "waste", b.id, f"Request enforcement inspection + CCTV review at {b.id}",
                                                     dict(bin=b.id), dict(metric="enforcement"), dict(metric="none", entities=[b.id], direction="none", ticks=1),
                                                     "LOW")]))
            if b.waste_type == "recycling" and b.contamination > 0.25:
                F.append(finding("waste", "poor_segregation", b.id, f"Poor segregation — recycling bin {b.id}",
                                 f"Contamination {b.contamination:.0%} (target < 15%) sends the load to landfill.",
                                 0.3, 0.9, 0.3, 0.3, zone=b.zone, x=b.x, y=b.y,
                                 proposals=[proposal("segregation_advisory", "waste", b.id, f"Send segregation advisory to premises around {b.id}",
                                                     dict(bin=b.id), dict(metric="contamination (%)", before=round(b.contamination * 100), after=15),
                                                     dict(metric="contamination", entities=[b.id], direction="decrease", ticks=24), "LOW")]))
            if b.temperature > 60:
                c.bus.publish(c.tick, self.name, SafetyAgent.name, "fire_risk", f"Bin {b.id} at {b.temperature:.0f} °C", dict(bin=b.id))
        for t in tw.trucks.values():
            for bid in t.missed:
                b = tw.bins[bid]
                F.append(finding("waste", "missed_collection", bid, f"Missed collection — {bid}", f"{t.id} could not service {bid}.",
                                 0.35, 1.0, 0.3, 0.6, zone=b.zone, x=b.x, y=b.y))
                cand.append(dict(bin=bid, priority=1.5, weight=b.weight_kg + 60))
            t.missed = []
        # ---- plan: predict → prioritise → optimise → dispatch
        idle = [t.id for t in tw.trucks.values() if t.status == "idle"]
        if cand and idle and "waste:collection_plan:city" not in c.busy:
            ids = {x["bin"] for x in cand}
            for b in tw.bins.values():       # opportunistic pick-ups near planned stops
                if b.id not in ids and b.id not in in_route and b.fill >= 0.6 and any(tw.bins[i].zone == b.zone for i in ids):
                    cand.append(dict(bin=b.id, priority=b.fill, weight=b.weight_kg + 60))
            trucks = idle[:max(1, min(len(idle), math.ceil(len(cand) / 6)))]
            plan = opt.optimize_collection(tw, cand, trucks)
            over_now = sum(1 for x in cand if fc.get(x["bin"], {}).get("overflow_hours") is not None and fc[x["bin"]]["overflow_hours"] <= settings.overflow_horizon_h)
            fixed_km = mem.get("fixed_km") or opt.fixed_schedule_km(tw)
            mem["fixed_km"] = fixed_km
            F.append(finding("waste", "collection_plan", "city", f"Predictive collection — {len(cand)} bins, {len(plan['routes'])} truck(s)",
                             f"{over_now} bins would overflow within {settings.overflow_horizon_h:g} h. Optimised route {plan['total_km']} km "
                             f"({plan['solver']}) vs fixed daily round {fixed_km} km.",
                             clip01(0.3 + over_now / 8), 0.85, clip01(len(cand) / 10), 0.7 if over_now else 0.4, name="Collection plan", zone=None,
                             evidence=dict(candidates=[x["bin"] for x in cand], solver=plan["solver"]),
                             proposals=[proposal("dispatch_collection", "waste", "city", f"Dispatch {', '.join(plan['routes'])} on optimised routes",
                                                 dict(routes={k: v["bins"] for k, v in plan["routes"].items()}),
                                                 dict(metric="bins overflowing / at risk", before=over_now, after=0, route_km=plan["total_km"], fixed_km=fixed_km),
                                                 dict(metric="bins_overflow", entities=[x["bin"] for x in cand], direction="decrease",
                                                      ticks=max(6, settings.verification_ticks * 3)), "LOW", tool="optimize_route")]))
        # ---- cross-domain: closures → collection delay
        for m in c.bus.inbox(c.tick, self.name, ("road_closure",)):
            rid = m["payload"]["road"]
            for t in tw.trucks.values():
                if t.status == "collecting" and t.stops:
                    base = tw.clone(); base.roads[rid].closed = False
                    _, tt0, km0 = base.route(t.node or t.depot, tw.bins[t.stops[0][0]].node)
                    _, tt1, km1 = tw.route(t.node or t.depot, tw.bins[t.stops[0][0]].node)
                    if km1 - km0 > 0.05:
                        c.bus.publish(c.tick, self.name, "orchestrator", "collection_delay",
                                      f"{t.id} route crosses closure — +{(km1 - km0):.1f} km detour", dict(truck=t.id, road=rid, extra_km=round(km1 - km0, 2)))
                        mem.setdefault("delays", {})[t.id] = dict(tick=c.tick, road=rid, extra_km=round(km1 - km0, 2))
        for tid, d in list(mem.get("delays", {}).items()):
            t = tw.trucks[tid]
            if t.status != "collecting" or c.tick - d["tick"] > 24:
                mem["delays"].pop(tid)
                continue
            remaining = [s[0] for s in t.stops]
            F.append(finding("waste", "collection_delay", tid, f"Collection delay — {tid} detour around closure",
                             f"Route crosses {tw.roads[d['road']].name}; +{d['extra_km']} km. Re-optimising remaining {len(remaining)} stops.",
                             0.35, 0.9, 0.3, 0.5, name=tid, evidence=d,
                             proposals=[proposal("reoptimize_collection", "waste", tid, f"Re-optimise {tid} route avoiding closure",
                                                 dict(truck=tid, bins=remaining), dict(metric="route km", extra_km=d["extra_km"]),
                                                 dict(metric="none", entities=[tid], direction="none", ticks=1), "LOW", tool="optimize_route")]))
        fills = [tw.bins[b].fill for b in tw.bins]
        overflowing = sum(1 for f in fills if f >= 1)
        at_risk = sum(1 for f_ in F if f_["kind"] == "overflow_risk")
        dumping = sum(1 for f_ in F if f_["kind"] == "illegal_dumping")
        health = 100 - 12 * overflowing - 3 * at_risk - 5 * dumping - 4 * sum(1 for f_ in F if f_["kind"] == "poor_segregation")
        lvl = 3 if overflowing >= 4 else 2 if overflowing >= 1 else 1 if at_risk >= 2 else 0
        mood = ["on track", "tightening", "behind — bins overflowing", "critical — widespread overflow"][lvl]
        headline = f"Waste collection is {mood} — {at_risk} bin{'s' * (at_risk != 1)} at overflow risk · {dumping} dumping alert{'s' * (dumping != 1)} · {len(idle)} truck{'s' * (len(idle) != 1)} available."
        kpis = dict(avg_fill_pct=round(float(np.mean(fills)) * 100, 1), overflowing_bins=overflowing, at_risk=at_risk,
                    trucks_active=sum(1 for t in tw.trucks.values() if t.status != "idle"), collected_kg=round(tw.waste_totals["collected_kg"]),
                    truck_km=round(tw.waste_totals["truck_km"], 1))
        return self.report(LEVELS[lvl], ["NORMAL", "WATCH", "WARNING", "CRITICAL"][lvl], max(0, min(100, health)), headline, F, kpis,
                           dict(forecasts={k: dict(overflow_hours=v["overflow_hours"], p=round(v["overflow_probability"], 2),
                                                   traj=v["trajectory"]) for k, v in fc.items()}))


# ============================================================================================
class EnergyAgent(DomainAgent):
    name, domain, title = "Energy Agent", "energy", "Energy & Efficiency Intelligence"

    def run(self, c: Ctx) -> dict:
        tw, ml, P = c.tw, c.ml, c.proc
        now = tw.t
        we, hr = int(now.weekday() >= 5), now.hour
        night = hr >= 22 or hr < 6
        F = []
        mem = c.memory
        demand_now = tw.load_kw + tw.ev_kw
        if c.tick % 6 == 0 or "fc" not in mem:
            mem["fc"] = ml.forecast_energy(now, demand_now, tw.weather)
            if c.tick % 12 == 0:
                ml.record("energy_demand", "CITY", c.tick + tw.ticks_for(60), mem["fc"][0]["demand_kw"], 60, now_tick=c.tick)
        fc = mem["fc"]
        exp_city = (ml.expected("energy", ("CITY", we, hr)) or {}).get("m", demand_now)
        vs_expected = (demand_now - exp_city) / max(exp_city, 1) * 100
        bfeat = {}
        for b in tw.buildings.values():
            pw = P.get(b.id, "power", b.power_kw)
            ex = ml.expected("energy", (b.id, we, hr)) or {}
            bfeat[b.id] = (pw, ex.get("m", pw), ex.get("s", 0) or 0)
        bif = dict(zip(bfeat, ml.if_scores("energy", [[pw / max(mu, 1), int(night)] for pw, mu, _ in bfeat.values()])))
        simc = mem.setdefault("sim_cache", {})

        def cached_sim(key, fn, ticks, meas):
            if key not in simc or c.tick - simc[key][0] >= 6:
                simc[key] = (c.tick, opt.simulate(tw, fn, ticks, meas))
            return simc[key][1]
        for b in tw.buildings.values():
            pw, mu, sd = bfeat[b.id]
            ratio = pw / max(mu, 1)
            z = (pw - mu) / max(sd, 0.06 * mu, 1)
            ifs = float(bif[b.id])
            hv, lt, pf = P.get(b.id, "hvac_power", b.hvac_kw), P.get(b.id, "lighting_power", b.lighting_kw), P.get(b.id, "power_factor", b.pf)
            if (z > c.thresholds["z"] - 0.5 or ratio > 1.25) and ifs > -0.02:
                hv_share = hv / max(pw, 1)
                cause = "hvac" if pf < 0.925 else ("lighting" if night and lt >= 0.97 * b.light_kw else "plug")
                causes = rank_causes([("HVAC inefficiency / fault", 0.9 if cause == "hvac" else 0.15, f"power factor {pf:.3f}, HVAC share {hv_share:.0%}"),
                                      ("Excessive lighting", 0.9 if cause == "lighting" else 0.05, f"lighting {lt:.0f} kW at {hr}:00"),
                                      ("Unexpected equipment load", 0.9 if cause == "plug" else 0.1, f"non-HVAC load {(pw - hv - lt):.0f} kW"),
                                      ("Weather (cooling demand)", clip01((tw.weather.temp - 30) / 8), f"outdoor {tw.weather.temp:.1f} °C")])
                props = []
                if cause == "hvac":
                    props.append(proposal("maintenance_request", "energy", b.id, f"Raise HVAC service work order for {b.name}",
                                          dict(target=b.id, kind="building"), dict(metric="building power (kW)", before=round(pw), after=round(mu)),
                                          dict(metric="building_power", entities=[b.id], direction="decrease", ticks=30), "MEDIUM", tool="predict_maintenance"))
                    if b.kind != "health" and b.hvac_offset == 0:
                        if ("hvac", b.id) not in simc or c.tick - simc[("hvac", b.id)][0] >= 6:
                            simc[("hvac", b.id)] = (c.tick, self._hvac_prop(tw, b))
                        props.append(simc[("hvac", b.id)][1])
                elif cause == "lighting":
                    sim = cached_sim(("light", b.id), lambda t, bid=b.id: setattr(t.buildings[bid], "lighting_fault", False), 2,
                                     lambda t, bid=b.id: t.buildings[bid].power_kw)
                    props.append(proposal("lighting_schedule", "energy", b.id, f"Restore lighting schedule at {b.name}", dict(building=b.id),
                                          dict(metric="building power (kW)", before=round(sim["baseline"]), after=round(sim["with_action"]),
                                               improvement_pct=round((sim["baseline"] - sim["with_action"]) / max(sim["baseline"], 1) * 100, 1)),
                                          dict(metric="building_power", entities=[b.id], direction="decrease", ticks=3), "LOW", tool="simulate_action"))
                else:
                    sim = cached_sim(("plug", b.id), lambda t, bid=b.id: setattr(t.buildings[bid], "plug_fault", 0.0), 2,
                                     lambda t, bid=b.id: t.buildings[bid].power_kw)
                    props.append(proposal("shutdown_idle_equipment", "energy", b.id, f"Schedule idle-equipment shutdown via BMS at {b.name}",
                                          dict(building=b.id), dict(metric="building power (kW)", before=round(sim["baseline"]), after=round(sim["with_action"]),
                                                                    improvement_pct=round((sim["baseline"] - sim["with_action"]) / max(sim["baseline"], 1) * 100, 1)),
                                          dict(metric="building_power", entities=[b.id], direction="decrease", ticks=3), "MEDIUM", tool="simulate_action"))
                kind = {"hvac": "hvac_inefficiency", "lighting": "excessive_lighting", "plug": "unexpected_consumption"}[cause]
                F.append(finding("energy", kind, b.id, f"{kind.replace('_', ' ').capitalize()} — {b.name}",
                                 f"{pw:.0f} kW vs expected {mu:.0f} kW (+{(ratio - 1) * 100:.0f}%, z={z:+.1f}, IF {ifs:+.2f}).",
                                 clip01((ratio - 1) * 1.5), clip01(0.5 + ifs * 3 + z / 10), clip01(pw / 1500), 0.4, name=b.name, zone=b.zone, x=b.x, y=b.y,
                                 root_causes=causes, evidence=dict(power=round(pw), expected=round(mu), z=round(z, 2), pf=round(pf, 3), if_score=round(ifs, 3)),
                                 proposals=props))
        # ---- peak demand + battery + EV
        thr = c.thresholds["peak_kw"]
        net = [f["demand_kw"] - f["solar_kw"] for f in fc]
        pk_i = int(np.argmax(net))
        cost24 = sum(max(0, n) * tw.tariff(f["hour"] + 0.5) for n, f in zip(net, fc))
        co2_24 = sum(max(0, n) * tw.grid_intensity(f["hour"] + 0.5) for n, f in zip(net, fc))
        bat = tw.energy_assets["BAT-1"]
        if max(net) > thr * 0.95 and pk_i <= 12 and "energy:peak_demand_risk:CITY" not in c.busy:
            lp = opt.battery_schedule(fc[:24], bat.soc, bat.capacity, bat.max_kw, tw)
            props = []
            if lp.get("ok"):
                props.append(proposal("battery_dispatch", "energy", "BAT-1", f"Optimise battery dispatch for the next 24 h (peak {lp['peak_before_kw']:.0f} → {lp['peak_after_kw']:.0f} kW)",
                                      dict(schedule=lp["schedule"], peak_guard_kw=round(lp["peak_after_kw"] * 0.88)), dict(metric="forecast peak (kW)", before=lp["peak_before_kw"], after=lp["peak_after_kw"],
                                                                          improvement_pct=round((lp["peak_before_kw"] - lp["peak_after_kw"]) / lp["peak_before_kw"] * 100, 1),
                                                                          cost_before=lp["cost_before"], cost_after=lp["cost_after"], co2_before=lp["co2_before_kg"],
                                                                          co2_after=lp["co2_after_kg"], solver=lp["solver"]),
                                      dict(metric="grid_peak_window", entities=["CITY"], direction="decrease", ticks=tw.ticks_for(12 * 60)),
                                      "MEDIUM", tool="optimize_resources"))
            if 15 <= hr < 21:
                sim = cached_sim("ev", self._ev_apply, 6, lambda t: t.grid_kw)
                props.append(proposal("ev_charging_shift", "energy", "EV", "Shift 40% of flexible EV charging out of the 17–21 h peak",
                                      dict(fraction=0.4), dict(metric="grid import in 30 min (kW)", before=round(sim["baseline"]), after=round(sim["with_action"]),
                                                               improvement_pct=round((sim["baseline"] - sim["with_action"]) / max(sim["baseline"], 1) * 100, 1)),
                                      dict(metric="grid_kw", entities=["CITY"], direction="decrease", ticks=6), "MEDIUM", tool="simulate_action"))
            F.append(finding("energy", "peak_demand_risk", "CITY", f"Peak demand risk at {fc[pk_i]['time'][11:16]}",
                             f"Forecast net demand {max(net):.0f} kW vs threshold {thr:.0f} kW; battery {'idle' if not bat.schedule else 'scheduled'} at {bat.soc:.0%} SoC.",
                             clip01((max(net) - thr * 0.9) / (thr * 0.3)), 0.8, 0.6, clip01(1 - pk_i / 24), name="City grid", zone="Z5",
                             x=bat.x, y=bat.y, forecast=dict(peak_kw=round(max(net)), peak_time=fc[pk_i]["time"], expected_cost_24h=round(cost24),
                                                             expected_co2_24h_kg=round(co2_24)),
                             root_causes=rank_causes([("Weather-driven cooling", clip01((tw.weather.temp - 28) / 8), f"{tw.weather.temp:.1f} °C"),
                                                      ("EV charging evening peak", clip01(tw.ev_kw / 900), f"EV load {tw.ev_kw:.0f} kW"),
                                                      ("Battery not dispatched", 0.5 if not bat.schedule else 0, f"SoC {bat.soc:.0%}"),
                                                      ("Low solar output", clip01(tw.weather.cloud), f"cloud {tw.weather.cloud:.0%}")]),
                             proposals=props))
        if any(a.curtail > 0 for a in tw.energy_assets.values() if a.kind == "solar"):
            F.append(finding("energy", "poor_solar_utilization", "PV", "Solar curtailed — poor utilisation",
                             "PV output curtailed while the battery is near full; storage schedule should absorb midday surplus.",
                             0.3, 0.9, 0.3, 0.3, name="Solar fleet", zone="Z5"))
        if tw.ev_kw > 0.75 * sum(a.max_kw for a in tw.energy_assets.values() if a.kind == "ev_hub") * 0.7 and "energy:ev_charging_peak:EV" not in c.busy:
            F.append(finding("energy", "ev_charging_peak", "EV", "EV charging peak", f"EV hubs drawing {tw.ev_kw:.0f} kW during the evening peak.",
                             0.4, 0.9, 0.4, 0.5, name="EV hubs", zone="Z2",
                             proposals=[proposal("ev_charging_shift", "energy", "EV", "Shift 40% of flexible EV charging after 22:00", dict(fraction=0.4),
                                                 dict(metric="EV load (kW)", before=round(tw.ev_kw), after=round(tw.ev_kw * 0.6), improvement_pct=40),
                                                 dict(metric="ev_kw", entities=["EV"], direction="decrease", ticks=3), "MEDIUM")]))
        anomalies = sum(1 for f_ in F if f_["kind"] in ("hvac_inefficiency", "excessive_lighting", "unexpected_consumption"))
        risks = sum(1 for f_ in F if f_["kind"] == "peak_demand_risk")
        opps = sum(len(f_["proposals"]) for f_ in F)
        health = 100 - 6 * anomalies - 12 * risks - max(0, vs_expected) * 0.8 - (8 if tw.grid_kw > thr else 0)
        lvl = 3 if tw.grid_kw > thr * 1.05 else 2 if risks or anomalies >= 3 else 1 if anomalies or vs_expected > 5 else 0
        label = ["NORMAL", "ELEVATED", "HIGH DEMAND", "PEAK EXCEEDED"][lvl]
        direction = "above" if vs_expected >= 0 else "below"
        headline = (f"City energy demand is {abs(vs_expected):.0f}% {direction} expected — {anomalies} anomal{'ies' if anomalies != 1 else 'y'} · "
                    f"{risks} forecast risk{'s' * (risks != 1)} · {opps} optimization opportunit{'ies' if opps != 1 else 'y'}.")
        kpis = dict(grid_kw=round(tw.grid_kw), demand_kw=round(demand_now), solar_kw=round(tw.solar_kw), ev_kw=round(tw.ev_kw),
                    battery_soc_pct=round(bat.soc * 100), day_peak_kw=round(tw.energy_totals["day_peak_kw"]), peak_threshold_kw=round(thr),
                    expected_cost_24h=round(cost24), expected_co2_24h_kg=round(co2_24), vs_expected_pct=round(vs_expected, 1))
        return self.report(LEVELS[lvl], label, max(0, min(100, health)), headline, F, kpis,
                           dict(forecast=[dict(t=f["time"][11:16], demand=round(f["demand_kw"]), solar=round(f["solar_kw"])) for f in fc]))

    @staticmethod
    def _ev_apply(t):
        for a in t.energy_assets.values():
            if a.kind == "ev_hub":
                a.ev_shift, a.ev_shift_until = 0.4, t.tick + t.ticks_for(max(30, (21 - t.hour) * 60))

    @staticmethod
    def _hvac_prop(tw, b):
        def ap(t):
            t.buildings[b.id].hvac_offset, t.buildings[b.id].hvac_offset_until = 1.5, t.tick + 36
        sim = opt.simulate(tw, ap, 2, lambda t: t.buildings[b.id].power_kw)
        return proposal("hvac_adjustment", "energy", b.id, f"Relax HVAC set-point +1.5 °C at {b.name} (within comfort band)",
                        dict(building=b.id, offset=1.5, duration_ticks=36),
                        dict(metric="building power (kW)", before=round(sim["baseline"]), after=round(sim["with_action"]),
                             improvement_pct=round((sim["baseline"] - sim["with_action"]) / max(sim["baseline"], 1) * 100, 1)),
                        dict(metric="building_power", entities=[b.id], direction="decrease", ticks=3), "LOW", tool="simulate_action")


# ============================================================================================
class InfrastructureAgent(DomainAgent):
    name, domain, title = "Infrastructure Agent", "infrastructure", "Asset & Maintenance Intelligence"

    def run(self, c: Ctx) -> dict:
        tw, ml, P = c.tw, c.ml, c.proc
        F = []
        rows, assets = [], []
        for a in tw.assets.values():
            vib = P.get(a.id, "vibration", a.vibration)
            st = P.stats.get(f"S-{a.id}-vibration")
            slope = (st.hist[-1] - st.hist[-4]) if st and len(st.hist) >= 4 else 0.0
            rows.append(ml.infra_features(a.kind, P.get(a.id, "structural_condition", a.condition), vib, P.get(a.id, "temperature", a.temperature),
                                          tw.weather.temp, P.get(a.id, "lamps_ok", a.lamps_ok * 100) / 100, P.get(a.id, "pressure", a.pressure),
                                          a.age_years, a.usage, a.criticality, slope))
            assets.append(a)
        pf, rul = ml.predict_failure(rows)
        ifa = ml.if_scores("infra", [x[2:6] for x in rows])
        queue = []
        for a, p, r, x, ifs in zip(assets, pf, rul, rows, ifa):
            if a.status == "FAILED":
                p, r = 1.0, 0.0
            pr = opt.maintenance_priority(a, float(p), float(r))
            pr["work_order"] = a.work_order
            queue.append(pr)
            if c.tick % 12 == 0 and a.status == "OPERATIONAL":
                ml.record("infra_failure_24h", a.id, c.tick + tw.ticks_for(24 * 60), float(p), 1440, now_tick=c.tick)
            ifs = float(ifs)
            props = []
            busy_crew = a.work_order is not None or a.status == "UNDER_REPAIR"
            crew = opt.best_unit(tw, tw.nearest_node(a.x, a.y), ("crew",))
            if not busy_crew and crew:
                kind = "maintenance_request" if (p >= 0.55 or a.status == "FAILED") else "inspection_request"
                props.append(proposal(kind, "infrastructure", a.id,
                                      f"{'Emergency repair' if a.status == 'FAILED' else 'Preventive maintenance' if kind == 'maintenance_request' else 'Inspection'} — {a.name} ({crew['unit']}, ETA {crew['eta_min']} min)",
                                      dict(target=a.id, kind="asset", unit=crew["unit"]),
                                      dict(metric="failure probability (24 h)", before=round(float(p), 2), after=0.05, rul_hours=round(float(r), 1)),
                                      dict(metric="asset_status", entities=[a.id], direction="restore", ticks=tw.ticks_for(crew["eta_min"] + 300)),
                                      "MEDIUM" if kind == "maintenance_request" else "LOW", tool="predict_maintenance"))
            if a.kind == "bridge" and a.road and not tw.roads[a.road].closed and p >= 0.75 and x[2] > 2.5:
                props.append(proposal("road_closure", "infrastructure", a.road, f"Precautionary closure of {tw.roads[a.road].name}",
                                      dict(road=a.road, reason=f"{a.name} vibration {x[2]:.1f}× normal"), dict(metric="public safety risk", note="prevents use of a bridge at risk"),
                                      dict(metric="none", entities=[a.road], direction="none", ticks=1), "HIGH", tool="detect_infrastructure_failure"))
            common = dict(name=a.name, zone=a.zone, x=a.x, y=a.y,
                          evidence=dict(failure_probability=round(float(p), 3), rul_h=round(float(r), 1), condition=round(a.condition, 1),
                                        vib_ratio=round(x[2], 2), temp_excess=round(x[3], 1), if_score=round(ifs, 3), priority=pr["score"]))
            crit = a.criticality / 5
            if a.status == "FAILED":
                F.append(finding("infrastructure", "asset_failure", a.id, f"Asset failure — {a.name}", f"{a.kind.replace('_', ' ').title()} out of service.",
                                 1.0, 1.0, clip01(a.population_served / 50000), 1.0, proposals=props, **common))
            elif p >= c.thresholds["failure_prob"]:
                causes = rank_causes([("Abnormal vibration", clip01((x[2] - 1.2) / 2), f"{x[2]:.1f}× normal"),
                                      ("Thermal stress", clip01(x[3] / 15), f"+{x[3]:.1f} °C over expected"),
                                      ("Leakage / pressure loss", clip01(x[5] / 0.5), f"pressure deficit {x[5]:.2f} bar"),
                                      ("Lamp failures", clip01((1 - x[4]) * 2), f"{x[4]:.0%} lamps working"),
                                      ("Age & condition", clip01((100 - a.condition) / 80), f"condition {a.condition:.0f}, {a.age_years:.0f} yrs"),
                                      ("Usage / load", clip01(a.usage - 0.8), f"usage index {a.usage:.2f}")])
                F.append(finding("infrastructure", "failure_risk", a.id, f"Failure risk {p:.0%} — {a.name}",
                                 f"Predicted failure probability (24 h) {p:.0%}; remaining useful life ≈ {r:.0f} h.",
                                 0.5 + 0.5 * crit, float(p), clip01(a.population_served / 50000), clip01(1 - r / 72), root_causes=causes,
                                 forecast=dict(failure_probability_24h=round(float(p), 3), rul_hours=round(float(r), 1)), proposals=props, **common))
            elif a.kind == "water_main" and x[5] > 0.15:
                F.append(finding("infrastructure", "water_leakage", a.id, f"Possible leakage — {a.name}", f"Pressure deficit {x[5]:.2f} bar with flow {a.flow:.0f} L/s.",
                                 0.5, 0.7, clip01(a.population_served / 50000), 0.6, proposals=props, **common))
            elif a.kind == "streetlight" and x[4] < 0.7:
                F.append(finding("infrastructure", "streetlight_failure", a.id, f"Streetlight outage — {a.name}", f"Only {x[4]:.0%} of lamps working.",
                                 0.4, 0.95, clip01(a.population_served / 60000), 0.5, proposals=props, **common))
            elif ifs > 0.04 and (x[2] > (1.7 if a.kind == "bridge" else 1.35) or x[3] > 4 or x[5] > 0.1 or x[4] < 0.9):
                # only deviations in the degrading direction matter (quieter-than-usual after a repair is not a fault)
                F.append(finding("infrastructure", "infrastructure_anomaly", a.id, f"Sensor pattern anomaly — {a.name}",
                                 f"Isolation Forest score {ifs:+.2f} on vibration/thermal/pressure features.", 0.3, 0.5, 0.3, 0.3, proposals=props, **common))
            elif a.condition < 40:
                F.append(finding("infrastructure", "degradation", a.id, f"Degradation — {a.name}", f"Condition index {a.condition:.0f}/100.",
                                 0.35, 0.6, 0.3, 0.3, proposals=props, **common))
        # ---- publish road closures (cross-domain trigger)
        known = c.memory.setdefault("closures", set())
        for e in tw.roads.values():
            if e.closed and e.id not in known:
                known.add(e.id)
                for to in ("Traffic Agent", "Safety Agent", "Waste Agent", "orchestrator"):
                    c.bus.publish(c.tick, self.name, to, "road_closure", f"Road closure detected: {e.name} ({e.closed_reason})", dict(road=e.id))
            elif not e.closed and e.id in known:
                known.discard(e.id)
        queue.sort(key=lambda q: -q["score"])
        failed = sum(1 for a in assets if a.status == "FAILED")
        at_risk = sum(1 for f_ in F if f_["kind"] == "failure_risk")
        wos = sum(1 for a in assets if a.work_order)
        avg_cond = float(np.mean([a.condition for a in assets]))
        health = avg_cond + 10 - 15 * failed - 4 * at_risk
        lvl = 3 if any(a.status == "FAILED" and a.criticality >= 4 for a in assets) else 2 if failed or at_risk >= 3 else 1 if at_risk else 0
        mood = ["in good shape", "showing early warning signs", "degraded", "critical"][lvl]
        headline = f"Infrastructure is {mood} — {failed} failed asset{'s' * (failed != 1)} · {at_risk} at risk · {wos} work order{'s' * (wos != 1)} active."
        kpis = dict(avg_condition=round(avg_cond, 1), failed=failed, at_risk=at_risk, work_orders=wos,
                    crews_available=sum(1 for u in tw.units.values() if u.kind == "crew" and u.status == "available"))
        return self.report(LEVELS[lvl], LEVELS[lvl], max(0, min(100, health)), headline, F, kpis, dict(priority_queue=queue[:12]))
