"""
Optimisation + calculation tools (deterministic; the LLM only selects and interprets them).

  * Signal timing      – Webster optimal cycle + critical-flow-ratio green splits, evaluated with HCM delay
  * Routing            – k-shortest alternative paths (NetworkX) weighted by spare capacity
  * Emergency response – nearest unit by emergency travel time, alternate route, signal-preemption corridor
  * Waste collection   – predict → prioritise → capacitated routing (OR-Tools if present, else NN + 2-opt)
  * Energy             – battery dispatch LP (SciPy linprog) minimising cost + peak + carbon; EV / HVAC shifting
  * Maintenance        – priority queue: Risk × Failure probability × Impact × Criticality
  * Counterfactual sim – clone the twin, apply an action, roll forward, compare to the no-action twin
  * Cost / carbon      – transparent formulas with editable assumptions in config
"""
from __future__ import annotations

import itertools
import math

import networkx as nx
import numpy as np
from scipy.optimize import linprog

from .config import settings

try:
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2  # type: ignore
    HAVE_ORTOOLS = True
except Exception:  # pragma: no cover
    HAVE_ORTOOLS = False


# ------------------------------------------------------------------ traffic
def webster_plan(tw, node_id: str) -> dict:
    n = tw.nodes[node_id]
    cap_w = 1 - min(0.18, 0.02 * tw.weather.rain)
    ln = sum(tw.roads[x].lanes / 2 for x in tw.incident_edges[node_id] if tw.roads[x].axis == "NS" and not tw.roads[x].closed) or 1
    le = sum(tw.roads[x].lanes / 2 for x in tw.incident_edges[node_id] if tw.roads[x].axis == "EW" and not tw.roads[x].closed) or 1
    s_ns, s_ew = 1800 * ln * cap_w, 1800 * le * cap_w
    g_ns, C = tw.webster(n.q_ns, n.q_ew, s_ns, s_ew)
    before = (n.q_ns * n.d_ns + n.q_ew * n.d_ew) / max(1, n.q_ns + n.q_ew)
    d_ns = tw._hcm_delay(n.q_ns, s_ns, g_ns, C, 0.25)
    d_ew = tw._hcm_delay(n.q_ew, s_ew, 1 - g_ns, C, 0.25)
    after = (n.q_ns * d_ns + n.q_ew * d_ew) / max(1, n.q_ns + n.q_ew)
    return dict(node=node_id, current=dict(g_ns=round(n.g_ns, 2), cycle=round(n.cycle)), proposed=dict(g_ns=round(g_ns, 2), cycle=round(C)),
                delay_before_s=round(before, 1), delay_after_s=round(after, 1),
                improvement_pct=round((before - after) / before * 100, 1) if before > 0 else 0.0,
                volume_vph=round(n.q_ns + n.q_ew))


def signal_opportunities(tw, min_gain_pct: float = 8.0, min_volume: float = 600) -> list[dict]:
    out = []
    for nid in tw.nodes:
        if tw.nodes[nid].plan_source == "AI-adaptive":
            continue
        p = webster_plan(tw, nid)
        if p["improvement_pct"] >= min_gain_pct and p["volume_vph"] >= min_volume and p["delay_before_s"] - p["delay_after_s"] >= 2:
            out.append(p)
    return sorted(out, key=lambda p: -(p["delay_before_s"] - p["delay_after_s"]) * p["volume_vph"])


def alternative_paths(tw, road_id: str, k: int = 3, max_vc: float | None = None) -> list[tuple[list[str], float]]:
    """Alternate paths between the road's end nodes. Roads that are themselves being rerouted, or already near
    capacity, are excluded so concurrent reroutes do not pile traffic onto each other."""
    e = tw.roads[road_id]
    avoid = {road_id} | set(tw.reroutes)
    if max_vc is not None:
        avoid |= {x.id for x in tw.roads.values() if x.vc > max_vc}
    w = tw.weight_fn(avoid=avoid)
    H = nx.Graph()
    for a, b, d in tw.G.edges(data=True):
        ww = w(a, b, d)
        if ww is not None:
            H.add_edge(a, b, w=ww, id=d["id"])
    paths = []
    try:
        for p in itertools.islice(nx.shortest_simple_paths(H, e.u, e.v, weight="w"), 6):
            edges = tw.path_edges(p)
            if len(edges) > 7:
                continue
            spare = min(max(1.0, tw.roads[x].capacity * tw.roads[x].cap_factor - tw.roads[x].flow) for x in edges)
            paths.append((edges, spare))
            if len(paths) >= k:
                break
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []
    tot = sum(s for _, s in paths) or 1
    return [(edges, s / tot) for edges, s in paths]


def predict_closure_impact(tw, road_id: str, ai_diversion: bool) -> dict:
    """Counterfactual: closure with organic diversion vs AI diversion, measured on a cloned twin."""
    from .twin import CityTwin  # noqa
    base = tw.clone()
    base.reroutes, base.diversions = {}, {}
    if tw.roads[road_id].closed:
        base.roads[road_id].closed = False
    base._traffic()
    vc0 = {k: e.vc for k, e in base.roads.items()}
    d0 = base.avg_intersection_delay()
    clo = tw.clone()
    clo.roads[road_id].closed, clo.roads[road_id].closed_reason = True, "what-if"
    if ai_diversion:
        clo.diversions[road_id] = dict(paths=alternative_paths(clo, road_id))
    else:
        clo.diversions.pop(road_id, None)
    clo._traffic()
    affected = sorted(((k, clo.roads[k].vc - vc0[k]) for k in clo.roads if k != road_id), key=lambda x: -x[1])[:5]
    d1 = clo.avg_intersection_delay()
    e = tw.roads[road_id]
    near = {x for n in (e.u, e.v) for x in tw.incident_edges[n]}
    near |= {y for x in list(near) for n in (tw.roads[x].u, tw.roads[x].v) for y in tw.incident_edges[n]}
    near.discard(road_id)
    near_max = max((clo.roads[k].vc for k in near), default=0)
    neigh = [k for k, dv in affected if dv > 0.02]
    vc_before = np.mean([vc0[k] for k in neigh]) if neigh else 0
    vc_after = np.mean([clo.roads[k].vc for k in neigh]) if neigh else 0
    return dict(delay_before_s=round(d0, 1), delay_after_s=round(d1, 1),
                congestion_increase_pct=round((vc_after - vc_before) / max(vc_before, 1e-6) * 100, 1) if neigh else 0.0,
                max_vc_after=round(max((clo.roads[k].vc for k in clo.roads if k != road_id), default=0), 3),
                near_max_vc=round(near_max, 3), near_roads=sorted(near),
                most_affected=[dict(road=k, name=clo.roads[k].name, vc_increase=round(dv, 3)) for k, dv in affected])


# ------------------------------------------------------------------ emergency
def best_unit(tw, target_node: str, kinds: tuple[str, ...], avoid_closed: bool = True) -> dict | None:
    best = None
    for u in tw.units.values():
        if u.kind not in kinds or u.status != "available":
            continue
        p, tt, km = tw.route(u.node, target_node, emergency=True)
        if p is None:
            continue
        if best is None or tt < best["eta_min"]:
            best = dict(unit=u.id, kind=u.kind, eta_min=round(tt, 1), km=round(km, 2), path=p)
    if best:
        alt_p, alt_tt, _ = tw.route(tw.units[best["unit"]].node, target_node, avoid=set(tw.path_edges(best["path"])), emergency=True)
        best["alternate_path"] = alt_p
        best["alternate_eta_min"] = round(alt_tt, 1) if alt_p else None
        best["corridor_eta_min"] = round(best["eta_min"] * 0.78, 1)
    return best


def coverage(tw) -> dict:
    """Emergency response time from nearest available ambulance/fire unit to every zone centroid node."""
    out = {}
    for z in tw.zones.values():
        zn = [n for n in tw.nodes.values() if n.zone == z.id]
        c = zn[len(zn) // 2].id
        b = best_unit(tw, c, ("ambulance", "fire"))
        out[z.id] = b["eta_min"] if b else math.inf
    return out


# ------------------------------------------------------------------ waste
def _dist_matrix(tw, nodes: list[str]) -> np.ndarray:
    w = tw.weight_fn()
    H = nx.Graph()
    for a, b, d in tw.G.edges(data=True):
        if w(a, b, d) is not None:
            H.add_edge(a, b, km=tw.roads[d["id"]].length_km)
    n = len(nodes)
    D = np.full((n, n), 99.0)
    for i, a in enumerate(nodes):
        try:
            lens = nx.single_source_dijkstra_path_length(H, a, weight="km")
        except nx.NodeNotFound:
            continue
        for j, b in enumerate(nodes):
            D[i, j] = lens.get(b, 99.0)
    return D + 0.12 * (1 - np.eye(n))   # stop access overhead


def _tour_len(order, D):
    return sum(D[a, b] for a, b in zip(order, order[1:]))


def _two_opt(order, D):
    best = order[:]
    improved = True
    while improved:
        improved = False
        for i in range(1, len(best) - 2):
            for j in range(i + 1, len(best) - 1):
                new = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                if _tour_len(new, D) + 1e-9 < _tour_len(best, D):
                    best, improved = new, True
    return best


def optimize_collection(tw, candidates: list[dict], trucks: list[str]) -> dict:
    """candidates: [{bin, priority, weight}] → per-truck ordered routes (capacitated)."""
    if not candidates or not trucks:
        return dict(routes={}, total_km=0.0, solver="none")
    nodes = [tw.depot] + [tw.bins[c["bin"]].node for c in candidates]
    D = _dist_matrix(tw, nodes)
    cap = tw.trucks[trucks[0]].capacity_kg
    routes: dict[str, list[int]] = {}
    solver = "nearest-neighbour + 2-opt"
    if HAVE_ORTOOLS and len(candidates) > 2:  # pragma: no cover - optional
        solver = "OR-Tools CVRP"
        mgr = pywrapcp.RoutingIndexManager(len(nodes), len(trucks), 0)
        rt = pywrapcp.RoutingModel(mgr)
        cb = rt.RegisterTransitCallback(lambda i, j: int(D[mgr.IndexToNode(i), mgr.IndexToNode(j)] * 1000))
        rt.SetArcCostEvaluatorOfAllVehicles(cb)
        dem = [0] + [int(c["weight"]) for c in candidates]
        dcb = rt.RegisterUnaryTransitCallback(lambda i: dem[mgr.IndexToNode(i)])
        rt.AddDimensionWithVehicleCapacity(dcb, 0, [int(cap)] * len(trucks), True, "cap")
        prm = pywrapcp.DefaultRoutingSearchParameters()
        prm.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        prm.time_limit.seconds = 2
        sol = rt.SolveWithParameters(prm)
        if sol:
            for v, tid in enumerate(trucks):
                idx, seq = rt.Start(v), []
                while not rt.IsEnd(idx):
                    nd = mgr.IndexToNode(idx)
                    if nd:
                        seq.append(nd)
                    idx = sol.Value(rt.NextVar(idx))
                if seq:
                    routes[tid] = seq
    if not routes:
        remaining = sorted(range(1, len(nodes)), key=lambda i: -candidates[i - 1]["priority"])
        per = math.ceil(len(remaining) / len(trucks))
        # cluster by angle around depot, then NN + 2-opt within each truck
        dep = tw.nodes[tw.depot]
        ang = sorted(remaining, key=lambda i: math.atan2(tw.bins[candidates[i - 1]["bin"]].y - dep.y, tw.bins[candidates[i - 1]["bin"]].x - dep.x))
        for k, tid in enumerate(trucks):
            grp = ang[k * per:(k + 1) * per]
            if not grp:
                continue
            load, seq, cur, left = 0.0, [], 0, set(grp)
            while left:
                nxt = min(left, key=lambda j: D[cur, j])
                if load + candidates[nxt - 1]["weight"] > cap:
                    break
                seq.append(nxt); load += candidates[nxt - 1]["weight"]; left.remove(nxt); cur = nxt
            tour = _two_opt([0] + seq + [0], D)
            routes[tid] = [i for i in tour if i != 0]
    out, total = {}, 0.0
    for tid, seq in routes.items():
        km = _tour_len([0] + seq + [0], D)
        total += km
        out[tid] = dict(bins=[candidates[i - 1]["bin"] for i in seq], km=round(km, 2))
    return dict(routes=out, total_km=round(total, 2), solver=solver)


def fixed_schedule_km(tw) -> float:
    total = 0.0
    for tid, ids in tw.fixed_rounds.items():
        nodes = [tw.depot] + [tw.bins[b].node for b in ids]
        D = _dist_matrix(tw, nodes)
        total += _tour_len(list(range(len(nodes))) + [0], D)
    return round(total, 2)


# ------------------------------------------------------------------ energy
def battery_schedule(forecast: list[dict], soc: float, cap_kwh: float, max_kw: float, tw) -> dict:
    """LP over the next N hours. Decision: charge c_t, discharge d_t, grid g_t, peak P.
    min Σ (tariff_t + carbon_price·ci_t)·g_t + demand_charge·P   s.t. g_t ≥ L_t − S_t + c_t − d_t, g_t ≤ P, SoC bounds."""
    T = len(forecast)
    L = np.array([f["demand_kw"] for f in forecast])
    S = np.array([f["solar_kw"] for f in forecast])
    tar = np.array([tw.tariff(f["hour"] + 0.5) for f in forecast])
    ci = np.array([tw.grid_intensity(f["hour"] + 0.5) for f in forecast])
    carbon_price = settings.carbon_shadow_price  # currency per kg CO2, internal shadow price
    eta = 0.92
    n = 3 * T + 1                    # c, d, g, P
    cost = np.concatenate([np.full(T, 0.002), np.full(T, 0.002), tar + carbon_price * ci, [settings.demand_charge_per_kw / 30]])
    A, b = [], []
    for t in range(T):               # L - S + c - d - g <= 0
        row = np.zeros(n); row[t] = 1; row[T + t] = -1; row[2 * T + t] = -1
        A.append(row); b.append(-(L[t] - S[t]))
        row = np.zeros(n); row[2 * T + t] = 1; row[3 * T] = -1     # g_t <= P
        A.append(row); b.append(0)
    for t in range(T):               # SoC bounds: soc0 + Σ(eta c - d)/cap ∈ [0.1, 0.95]
        row = np.zeros(n); row[:t + 1] = eta / cap_kwh; row[T:T + t + 1] = -1 / cap_kwh
        A.append(row); b.append(0.95 - soc)
        A.append(-row); b.append(soc - 0.1)
    bounds = [(0, max_kw)] * T + [(0, max_kw)] * T + [(0, None)] * T + [(0, None)]
    res = linprog(cost, A_ub=np.array(A), b_ub=np.array(b), bounds=bounds, method="highs")
    if not res.success:
        return dict(ok=False, reason=res.message)
    c, d, gg, P = res.x[:T], res.x[T:2 * T], res.x[2 * T:3 * T], res.x[3 * T]
    g0 = np.maximum(0, L - S)
    sched = {int(f["hour"]): round(float(d[i] - c[i]), 1) for i, f in enumerate(forecast)}
    return dict(ok=True, schedule=sched, peak_before_kw=round(float(g0.max()), 1), peak_after_kw=round(float(P), 1),
                cost_before=round(float((g0 * tar).sum()), 2), cost_after=round(float((gg * tar).sum()), 2),
                co2_before_kg=round(float((g0 * ci).sum()), 1), co2_after_kg=round(float((gg * ci).sum()), 1),
                solver="scipy-linprog/HiGHS")


# ------------------------------------------------------------------ maintenance priority
def maintenance_priority(asset, p_fail: float, rul_h: float) -> dict:
    risk = 1 - asset.condition / 100
    impact = min(1.0, asset.population_served / 60000) * 0.6 + (0.4 if asset.kind in ("bridge", "transformer", "water_main") else 0.2)
    crit = asset.criticality / 5
    urgency = 1.0 if rul_h < 12 else (0.7 if rul_h < 48 else 0.4)
    score = 100 * (0.3 + 0.7 * risk) * max(p_fail, 0.02) * impact * crit * (0.5 + 0.5 * urgency) * 4
    return dict(asset=asset.id, name=asset.name, kind=asset.kind, zone=asset.zone, condition=round(asset.condition, 1),
                failure_probability=round(p_fail, 3), rul_hours=round(rul_h, 1), criticality=asset.criticality,
                population=asset.population_served, repair_cost=asset.repair_cost, score=round(min(100, score), 2), status=asset.status,
                safety_impact="HIGH" if asset.kind in ("bridge", "streetlight", "transformer") else "MEDIUM")


# ------------------------------------------------------------------ counterfactual simulation
def simulate(tw, apply_fn, ticks: int, measure_fn, also_baseline: bool = True) -> dict:
    """Clone twin; run with & without the action for `ticks`; return the measured metric for both (paired RNG)."""
    a = tw.clone()
    apply_fn(a)
    for _ in range(ticks):
        a.step()
    with_action = measure_fn(a)
    out = dict(with_action=with_action)
    if also_baseline:
        b = tw.clone()
        for _ in range(ticks):
            b.step()
        out["baseline"] = measure_fn(b)
    return out


# ------------------------------------------------------------------ cost & carbon
def calculate_cost(tw) -> dict:
    hours_delay = tw.traffic_totals["delay_veh_h"]
    return dict(
        traffic_delay_cost=round(hours_delay * settings.value_of_time_per_h, 0),
        fleet_fuel_cost=round(tw.waste_totals["fuel_l"] * settings.fuel_cost_per_l, 0),
        waste_collection_cost=round(tw.waste_totals["truck_km"] * settings.truck_cost_per_km + tw.waste_totals["fuel_l"] * settings.fuel_cost_per_l, 0),
        energy_cost=round(tw.energy_totals["cost"] + tw.energy_totals["peak_kw"] * settings.demand_charge_per_kw / 30, 0),
        assumptions=dict(value_of_time_per_h=settings.value_of_time_per_h, fuel_cost_per_l=settings.fuel_cost_per_l,
                         demand_charge_per_kw=settings.demand_charge_per_kw),
    )


def calculate_carbon(tw) -> dict:
    traffic = tw.traffic_totals["delay_veh_h"] * settings.car_idle_l_per_h * settings.petrol_kgco2_per_l
    fleet = tw.waste_totals["fuel_l"] * settings.diesel_kgco2_per_l
    landfill = tw.waste_totals["landfill_kg"] * settings.landfill_kgco2e_per_kg
    elec = tw.energy_totals["co2"]
    return dict(traffic_idling_kg=round(traffic), fleet_kg=round(fleet), landfill_kgco2e=round(landfill), electricity_kg=round(elec),
                total_kgco2e=round(traffic + fleet + landfill + elec))
