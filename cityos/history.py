"""Synthetic historical dataset generator → data/*.csv (traffic, waste, energy, safety, infrastructure)."""
from __future__ import annotations

from datetime import timedelta

import pandas as pd

from .config import settings
from .twin import CityTwin


def generate_history(days: int | None = None, start=None, seed: int = 1234) -> dict[str, pd.DataFrame]:
    days = days or settings.history_days
    from datetime import datetime
    start = start or (datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=days))
    tw = CityTwin(seed=seed, start=start, light=True)
    tw.event_rate = settings.event_rate * 1.5          # richer history for training
    traffic, waste, energy, safety, infra = [], [], [], [], []
    n = int(days * 24 * 60 / tw.dt_min)
    for _ in range(n):
        tw.step()
        ts = tw.t
        if ts.hour == 0 and ts.minute == 0:
            # fleet diversity: re-baseline some assets' condition so condition alone is not a proxy for "recently failed"
            for a in tw.assets.values():
                if a.status == "OPERATIONAL" and a.fault == 0 and tw.rng.random() < 0.35:
                    a.condition = float(tw.rng.uniform(30, 96))
        w = tw.weather
        for e in tw.roads.values():
            traffic.append((ts, e.id, e.zone, e.flow, e.speed, e.vc, e.density, e.occupancy, e.queue_m, e.capacity * e.cap_factor,
                            e.base_flow, int(e.closed), e.event_mult, w.temp, w.rain))
        if tw.t.minute % 15 == 0:
            for b in tw.bins.values():
                waste.append((ts, b.id, b.zone, b.waste_type, b.fill, b.weight_kg, b.temperature, b.contamination, w.temp,
                              tw.zones[b.zone].event_mult))
            for z in tw.zones.values():
                safety.append((ts, z.id, z.kind, z.density, int(z.event is not None), int(z.anomaly_until >= tw.tick), w.rain,
                               sum(1 for i in tw.incidents.values() if i.zone == z.id and i.status != "CLEARED")))
        energy.append((ts, tw.grid_kw, tw.load_kw, tw.ev_kw, tw.solar_kw, w.temp, w.cloud, w.rain,
                       *[b.power_kw for b in tw.buildings.values()]))
        if tw.t.minute % 30 == 0:
            for a in tw.assets.values():
                infra.append((ts, a.id, a.kind, a.zone, a.condition, a.vibration, a.temperature, a.lamps_ok, a.pressure, a.flow,
                              a.age_years, a.usage, a.criticality, a.fault, a.status, w.temp))
            # traditional maintenance in history: failed assets are repaired after a delay (reactive)
            for a in tw.assets.values():
                if a.status == "FAILED" and tw.rng.random() < 0.08:
                    tw._repair(a, None)
    bcols = [f"bld_{b}" for b in tw.buildings]
    dfs = dict(
        traffic=pd.DataFrame(traffic, columns=["timestamp", "road", "zone", "flow", "speed", "vc", "density", "occupancy", "queue_m",
                                               "capacity", "base_flow", "closed", "event_mult", "temp", "rain"]),
        waste=pd.DataFrame(waste, columns=["timestamp", "bin", "zone", "waste_type", "fill", "weight", "temperature", "contamination",
                                           "temp", "event_mult"]),
        energy=pd.DataFrame(energy, columns=["timestamp", "grid_kw", "load_kw", "ev_kw", "solar_kw", "temp", "cloud", "rain", *bcols]),
        safety=pd.DataFrame(safety, columns=["timestamp", "zone", "kind", "density", "event", "abnormal", "rain", "open_incidents"]),
        infrastructure=pd.DataFrame(infra, columns=["timestamp", "asset", "kind", "zone", "condition", "vibration", "temperature",
                                                    "lamps_ok", "pressure", "flow", "age_years", "usage", "criticality", "fault",
                                                    "status", "air_temp"]),
    )
    for k, df in dfs.items():
        df.to_csv(settings.data_dir / f"{k}.csv", index=False)
    return dfs


def load_history() -> dict[str, pd.DataFrame] | None:
    out = {}
    for k in ("traffic", "waste", "energy", "safety", "infrastructure"):
        p = settings.data_dir / f"{k}.csv"
        if not p.exists():
            return None
        out[k] = pd.read_csv(p, parse_dates=["timestamp"])
    return out
