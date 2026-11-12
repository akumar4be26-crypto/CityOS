"""
CityOS — Autonomous Agentic AI for Smart Cities & Intelligent Infrastructure
FastAPI entry point:  python main.py   (or: uvicorn main:app --port 8000)

Frontend → FastAPI → City Orchestrator → Domain Agents → ML / Tools → Database
"""
from __future__ import annotations

import asyncio
import json
import math
from contextlib import asynccontextmanager
from datetime import datetime

import numpy as np
from fastapi import Body, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from cityos.config import settings
from cityos.llm import TOOL_SPECS
from cityos.orchestrator import Orchestrator

ORCH: Orchestrator | None = None
CLIENTS: set[WebSocket] = set()


# ------------------------------------------------------------------ JSON helpers (numpy / NaN safe)
def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        f = float(o)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, datetime):
        return o.isoformat()
    return o


def J(obj, status: int = 200) -> Response:
    return Response(content=json.dumps(_clean(obj), default=str), media_type="application/json", status_code=status)


def orch() -> Orchestrator:
    if ORCH is None:
        raise HTTPException(503, "CityOS is starting — models are loading or training")
    return ORCH


# ------------------------------------------------------------------ background closed loop
async def run_loop():
    global ORCH
    ORCH = await asyncio.to_thread(Orchestrator)
    n = 0
    while True:
        o = ORCH
        if o.running:
            try:
                await asyncio.to_thread(o.cycle)
            except Exception as e:  # keep the loop alive; surface in the activity log
                o.log("OBSERVE", "orchestrator", f"Cycle error: {type(e).__name__}: {e}")
            n += 1
            if CLIENTS:
                snap = await asyncio.to_thread(o.snapshot, n % 5 == 0)
                msg = json.dumps(_clean(dict(type="snapshot", data=snap)), default=str)
                for ws in list(CLIENTS):
                    try:
                        await ws.send_text(msg)
                    except Exception:
                        CLIENTS.discard(ws)
        await asyncio.sleep(max(0.15, settings.tick_seconds / max(0.1, o.speed)))


@asynccontextmanager
async def lifespan(_app: FastAPI):
    task = asyncio.create_task(run_loop())
    yield
    task.cancel()


app = FastAPI(title="CityOS — Autonomous City Operations Agent", version="1.0.0", lifespan=lifespan,
              description="Agentic AI for traffic, safety, waste, energy and infrastructure. All actions execute in a simulated digital twin.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=str(settings.frontend_dir)), name="static")


class SimulationRequest(BaseModel):
    scenario: str = Field(..., description="population_increase | festival | road_closure | extreme_weather | ev_adoption | "
                                           "waste_increase | energy_demand_increase | infrastructure_failure | fleet_reduction")
    params: dict = Field(default_factory=dict)
    hours: int = Field(24, ge=1, le=72)
    mode: str = Field("whatif", description="whatif (paired twin simulation) | live (inject into the running city)")


class Decision(BaseModel):
    approver: str = "operator"
    reason: str = ""


class Control(BaseModel):
    running: bool | None = None
    speed: float | None = Field(None, ge=0.25, le=8)


# ------------------------------------------------------------------ frontend
@app.get("/", include_in_schema=False)
def index():
    return FileResponse(settings.frontend_dir / "index.html")


@app.get("/api/health", tags=["system"])
def health():
    return J(dict(ready=ORCH is not None, boot=ORCH.boot_log if ORCH else ["starting"], llm=ORCH.llm.status() if ORCH else None))


# ------------------------------------------------------------------ city
@app.get("/api/city/status", tags=["city"])
def city_status():
    o = orch()
    s = o.snapshot(full=False)
    return J({k: s[k] for k in ("city", "tick", "time", "weather", "health", "health_trend", "state", "orchestrator", "running", "speed")}
             | dict(agents={d: {k: v for k, v in a.items() if k != "extra"} for d, a in s["agents"].items()}))


@app.get("/api/city/snapshot", tags=["city"])
def city_snapshot():
    return J(orch().snapshot(full=True))


@app.get("/api/city/twin", tags=["city"])
def city_twin():
    o = orch()
    with o.lock:
        return J(o.tw.geometry())


@app.post("/api/city/control", tags=["city"])
def city_control(c: Control):
    o = orch()
    if c.running is not None:
        o.running = c.running
    if c.speed is not None:
        o.speed = c.speed
    return J(dict(running=o.running, speed=o.speed))


@app.get("/api/city/simulation", tags=["simulation"])
def list_scenarios():
    return J({k: v["label"] for k, v in Orchestrator.SCENARIOS.items()})


@app.post("/api/city/simulation", tags=["simulation"])
async def simulation(req: SimulationRequest):
    o = orch()
    if req.mode == "live":
        return J(o.inject(req.scenario, req.params))
    res = await asyncio.to_thread(o.run_scenario, req.scenario, req.params, req.hours)
    return J(res, 400 if "error" in res else 200)


@app.post("/api/city/simulation/clear", tags=["simulation"])
def simulation_clear():
    orch().clear_scenario()
    return J(dict(ok=True))


# ------------------------------------------------------------------ domain endpoints
def _domain(d: str) -> dict:
    o = orch()
    with o.lock:
        r = o.reports.get(d)
        if not r:
            raise HTTPException(404, "domain not ready")
        return dict(r)


@app.get("/api/traffic/current", tags=["traffic"])
def traffic_current():
    o = orch()
    r = _domain("traffic")
    live = o.tw.live()
    return J(dict(kpis=r["kpis"], level=r["level"], label=r["label"], headline=r["headline"], roads=live["roads"], intersections=live["nodes"],
                  findings=[o._issue_view(f) for f in r["findings"] if f["kind"] in ("congestion", "closure_impact", "bottleneck", "abnormal_traffic")]))


@app.get("/api/traffic/forecast", tags=["traffic"])
def traffic_forecast():
    o = orch()
    with o.lock:
        tw = o.tw
        fc = o.ml.predict_flows(list(tw.roads.values()), tw.t, [15, 30, 60, 120], tw.weather)
        out = {}
        for e in tw.roads.values():
            cap = max(1, e.capacity * e.cap_factor)
            out[e.id] = dict(name=e.name, current_vc=round(e.vc, 3),
                             horizons={f"+{h}min": dict(flow=round(fc[(e.id, h)]), vc=round(fc[(e.id, h)] / cap, 3),
                                                        p_congestion=round(o.ml.congestion_probability(fc[(e.id, h)] / cap), 3)) for h in (15, 30, 60, 120)})
        return J(dict(time=tw.t.isoformat(timespec="minutes"), roads=out, model=o.ml.metrics.get("traffic_flow"),
                      live_error=o.ml.live_metrics().get("traffic_flow")))


@app.get("/api/traffic/incidents", tags=["traffic"])
def traffic_incidents():
    o = orch()
    return J([i for i in o.tw.live()["incidents"] if i["type"] == "traffic_collision"] +
             [dict(id=e.id, type="road_closure", road=e.id, description=f"{e.name} closed — {e.closed_reason}") for e in o.tw.roads.values() if e.closed])


@app.get("/api/safety/status", tags=["safety"])
def safety_status():
    r = _domain("safety")
    return J(dict(level=r["level"], headline=r["headline"], kpis=r["kpis"], zones=r["extra"]["zones"], coverage_min=r["extra"]["coverage"]))


@app.get("/api/safety/incidents", tags=["safety"])
def safety_incidents():
    o = orch()
    r = _domain("safety")
    return J(dict(open=o.tw.live()["incidents"], findings=[o._issue_view(f) for f in r["findings"]]))


@app.get("/api/waste/current", tags=["waste"])
def waste_current():
    o = orch()
    r = _domain("waste")
    return J(dict(level=r["level"], headline=r["headline"], kpis=r["kpis"], trucks=o.tw.live()["trucks"],
                  findings=[o._issue_view(f) for f in r["findings"]]))


@app.get("/api/waste/forecast", tags=["waste"])
def waste_forecast():
    r = _domain("waste")
    return J(dict(bins=r["extra"]["forecasts"], horizon_hours=settings.overflow_horizon_h))


@app.get("/api/waste/bins", tags=["waste"])
def waste_bins():
    o = orch()
    return J(o.tw.live()["bins"])


@app.get("/api/energy/current", tags=["energy"])
def energy_current():
    o = orch()
    r = _domain("energy")
    live = o.tw.live()
    return J(dict(level=r["level"], label=r["label"], headline=r["headline"], kpis=r["kpis"], grid=live["grid"], buildings=live["buildings"],
                  assets=live["energy_assets"], findings=[o._issue_view(f) for f in r["findings"]]))


@app.get("/api/energy/forecast", tags=["energy"])
def energy_forecast():
    o = orch()
    r = _domain("energy")
    return J(dict(hourly=r["extra"]["forecast"], expected_cost_24h=r["kpis"]["expected_cost_24h"],
                  expected_co2_24h_kg=r["kpis"]["expected_co2_24h_kg"], peak_threshold_kw=r["kpis"]["peak_threshold_kw"],
                  model=o.ml.metrics.get("energy_demand"), solar_model=o.ml.metrics.get("solar")))


@app.get("/api/infrastructure/assets", tags=["infrastructure"])
def infra_assets():
    o = orch()
    with o.lock:
        geo = {a["id"]: a for a in o.tw.geometry()["assets"]}
        live = o.tw.live()["assets"]
        return J({k: geo[k] | v for k, v in live.items()})


@app.get("/api/infrastructure/maintenance", tags=["infrastructure"])
def infra_maintenance():
    o = orch()
    r = _domain("infrastructure")
    return J(dict(priority_queue=r["extra"]["priority_queue"], formula="Risk × Failure probability × Impact × Criticality",
                  work_orders=[a for a in o.snapshot(False)["actions"] if a["type"] in ("maintenance_request", "inspection_request")],
                  model=o.ml.metrics.get("infra_failure")))


# ------------------------------------------------------------------ agents & actions
@app.get("/api/agents/status", tags=["agents"])
def agents_status():
    o = orch()
    s = o.snapshot(full=False)
    return J(dict(orchestrator=s["orchestrator"], agents={d: {k: v for k, v in a.items() if k != "extra"} for d, a in s["agents"].items()},
                  state=s["state"], llm=o.llm.status()))


@app.get("/api/agents/decisions", tags=["agents"])
def agents_decisions(limit: int = 50):
    o = orch()
    s = o.snapshot(full=False)
    return J([dict(action=a["id"], issue=a["issue_title"], decision=a["decision"], status=a["status"], risk=a["risk"]) for a in s["actions"][:limit]])


@app.get("/api/agents/log", tags=["agents"])
def agents_log(limit: int = 200, phase: str | None = None):
    o = orch()
    with o.lock:
        rows = [e for e in o.activity if not phase or e["phase"] == phase.upper()]
    return J(rows[-limit:])


@app.get("/api/agents/messages", tags=["agents"])
def agents_messages():
    return J(orch().bus.messages[-80:])


@app.get("/api/issues", tags=["agents"])
def issues():
    return J(orch().snapshot(False)["issues"])


@app.get("/api/actions", tags=["actions"])
def actions(status: str | None = None):
    o = orch()
    with o.lock:
        rows = [o._action_view(o.actions[a]) for a in o.action_order if a in o.actions]
    return J([a for a in rows if not status or a["status"] == status.upper()])


@app.get("/api/actions/{aid}", tags=["actions"])
def action(aid: str):
    o = orch()
    a = o.actions.get(aid)
    if not a:
        raise HTTPException(404, "unknown action")
    return J(o._action_view(a) | dict(params=a.get("params"), measure=a.get("measure")))


@app.get("/api/actions/{aid}/verify", tags=["actions"])
def verify(aid: str):
    o = orch()
    a = o.actions.get(aid)
    if not a:
        raise HTTPException(404, "unknown action")
    return J(dict(id=aid, status=a["status"], verification=a.get("verification"), before=a.get("before"), after=a.get("after"),
                  counterfactual=a.get("counterfactual"), execution_mode=a["execution_mode"]))


@app.post("/api/actions/{aid}/approve", tags=["actions"])
def approve(aid: str, d: Decision = Body(default=Decision())):
    r = orch().approve(aid, d.approver)
    return J(r, 200 if r.get("ok") else 409)


@app.post("/api/actions/{aid}/reject", tags=["actions"])
def reject(aid: str, d: Decision = Body(default=Decision())):
    r = orch().reject(aid, d.approver, d.reason)
    return J(r, 200 if r.get("ok") else 409)


# ------------------------------------------------------------------ ML / tools / reports / sensors
@app.get("/api/ml/metrics", tags=["ml"])
def ml_metrics():
    return J(orch().metrics())


@app.get("/api/tools", tags=["tools"])
def tools():
    return J([dict(name=n, description=d, params=p) for n, d, p in TOOL_SPECS])


@app.post("/api/tools/{name}", tags=["tools"])
def run_tool(name: str, args: dict = Body(default={})):
    return J(orch().call_tool(name, args))


@app.get("/api/sensors/{entity}/{metric}", tags=["data"])
def sensor_series(entity: str, metric: str, limit: int = 288):
    return J(orch().store.sensor_series(entity, metric, limit))


@app.get("/api/data/quality", tags=["data"])
def data_quality():
    return J(orch().proc.summary())


@app.get("/api/reports/briefing", tags=["reports"])
async def briefing():
    o = orch()
    s = o.snapshot(full=True)
    data = dict(time=s["time"], health=s["health"], state=s["state"], headlines={d: a["headline"] for d, a in s["agents"].items()},
                top_issues=[dict(title=i["title"], category=i["category"], detail=i["detail"]) for i in s["issues"][:8]],
                recent_actions=[dict(title=a["title"], status=a["status"], verification=(a.get("verification") or {}).get("result"))
                                for a in s["actions"][:12]], pending_authorisation=len(s["approvals"]),
                detection=s["metrics"]["detection"], forecast_errors=s["metrics"]["live_forecast_errors"])
    if o.llm.enabled:
        try:
            text, trace = await asyncio.to_thread(
                o.llm.run, "Write a concise operational shift briefing (markdown, ≤ 220 words) for the city operations manager from this "
                           "data. Quote only numbers present in the data or tool results. Data:\n" + json.dumps(_clean(data), default=str),
                "You are CityOS's reporting layer. Never invent numbers. All actions are simulated in a digital twin.", True)
            return J(dict(source=f"LLM ({o.llm.model})", text=text, tools_used=[t["tool"] for t in trace], data=data))
        except Exception as e:
            data["llm_error"] = str(e)
    h = s["health"] or {}
    lines = [f"**CityOS briefing — {s['time'].replace('T', ' ')}**", "",
             f"City Health **{h.get('score', 0):.0f}/100**. " + " ".join(a["headline"] for a in s["agents"].values()), ""]
    if s["issues"]:
        lines.append("**Top issues**")
        lines += [f"- {i['category']}: {i['title']} — {i['detail']}" for i in s["issues"][:5]]
    done = [a for a in s["actions"] if a.get("verification")][:6]
    if done:
        lines += ["", "**Recent verified actions (simulated)**"]
        lines += [f"- {a['title']}: {a['verification']['result']} — {a['verification'].get('note', '')}" for a in done]
    if s["approvals"]:
        lines += ["", f"**{len(s['approvals'])} action(s) awaiting human authorisation.**"]
    return J(dict(source="Deterministic report generator (no LLM configured)", text="\n".join(lines), data=data))


# ------------------------------------------------------------------ websocket
@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    CLIENTS.add(websocket)
    try:
        if ORCH is not None:
            await websocket.send_text(json.dumps(_clean(dict(type="snapshot", data=ORCH.snapshot(True))), default=str))
        while True:
            await websocket.receive_text()   # keep-alive / ignore client messages
    except WebSocketDisconnect:
        pass
    finally:
        CLIENTS.discard(websocket)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(__import__("os").getenv("PORT", "8000")), reload=False)
