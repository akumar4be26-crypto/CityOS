"""
Data ingestion + processing pipeline.

  raw records → de-duplication → timestamp normalisation → unit conversion → range validation
  → missing-value imputation → sensor-health checks (stuck / dropout / uncorroborated spike)
  → rolling statistics → entity state → spatial (zone) aggregation → domain features

Every operational record follows: Timestamp + Location + Entity + Measurement + Status + Source.
Also contains configurable dataset connectors (CSV / JSON / REST) for public or municipal data.
"""
from __future__ import annotations

import json
import math
import urllib.request
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ROOT, settings

UNIT_CONVERSIONS = {
    ("speed", "mph"): (lambda v: v * 1.609344, "km/h"),
    ("power", "W"): (lambda v: v / 1000.0, "kW"),
    ("temperature", "F"): (lambda v: (v - 32) * 5 / 9, "C"),
    ("rainfall", "in/h"): (lambda v: v * 25.4, "mm/h"),
}

VALID_RANGES = {
    "vehicle_count": (0, 1500), "speed": (0, 140), "density": (0, 2500), "occupancy": (0, 100), "queue_length": (0, 5000),
    "crowd_density": (0, 8), "incident_count": (0, 100), "fill_level": (0, 150), "weight": (0, 3000), "temperature": (-30, 130),
    "contamination": (0, 100), "power": (0, 30000), "voltage": (150, 280), "current": (0, 80000), "power_factor": (0, 1.0001),
    "energy_consumption": (0, 1e12), "hvac_power": (0, 30000), "lighting_power": (0, 5000), "soc": (0, 100), "battery_power": (-5000, 5000),
    "vibration": (0, 60), "lamps_ok": (0, 100), "flow": (0, 1000), "pressure": (0, 12), "structural_condition": (0, 100),
    "operating_status": (0, 1), "rainfall": (0, 300), "humidity": (0, 100), "wind": (0, 250),
}
# metrics expected to change tick-to-tick (used for stuck-sensor detection)
VARYING = {"speed", "power", "vibration", "flow", "pressure", "voltage", "current", "crowd_density"}
# continuous physical signals where an isolated, uncorroborated jump indicates a sensor fault (step changes in
# crowd density, fill level, condition indices etc. are real events and are left to the domain agents)
SPIKE_CHECK = {"speed", "power", "vibration", "voltage", "current", "flow", "pressure", "temperature", "vehicle_count", "occupancy"}


class SensorStats:
    """Online per-sensor statistics (EWMA mean/var) + short history window."""
    __slots__ = ("mean", "var", "n", "hist", "last_good", "missing", "stuck", "suspect", "last_ts")

    def __init__(self):
        self.mean, self.var, self.n = 0.0, 0.0, 0
        self.hist: deque = deque(maxlen=48)
        self.last_good: float | None = None
        self.missing = 0
        self.stuck = 0
        self.suspect = 0
        self.last_ts = ""

    def update(self, v: float, alpha: float = 0.08):
        if self.n == 0:
            self.mean, self.var = v, 0.0
        else:
            d = v - self.mean
            self.mean += alpha * d
            self.var = (1 - alpha) * (self.var + alpha * d * d)
        self.n += 1
        self.hist.append(v)

    def z(self, v: float) -> float:
        sd = math.sqrt(self.var) if self.var > 0 else 0.0
        floor = max(abs(self.mean) * 0.05, 1e-3)
        return (v - self.mean) / max(sd, floor)


class Processor:
    def __init__(self):
        self.stats: dict[str, SensorStats] = defaultdict(SensorStats)
        self.entity_state: dict[str, dict] = {}            # entity -> {metric: value, _meta…}
        self.zone_state: dict[str, dict] = {}
        self.totals = defaultdict(int)
        self.last_batch: dict = {}
        self.sensor_issues: dict[str, dict] = {}            # sensor_id -> issue (live)
        self.seen_keys: deque = deque(maxlen=5000)

    # -------------------------------------------------------------- main entry
    def process(self, records: list[dict]) -> dict:
        b = defaultdict(int)
        b["received"] = len(records)
        seen = set()
        clean: list[dict] = []
        for r in records:
            key = (r["sensor_id"], r["ts"])
            if key in seen:
                b["duplicates_removed"] += 1
                continue
            seen.add(key)
            r = dict(r)
            r["ts"] = self._norm_ts(r["ts"])
            conv = UNIT_CONVERSIONS.get((r["metric"], r["unit"]))
            if conv and r["value"] is not None:
                r["value"], r["unit"] = conv[0](r["value"]), conv[1]
                b["unit_conversions"] += 1
            clean.append(r)

        by_entity: dict[str, list[dict]] = defaultdict(list)
        for r in clean:
            st = self.stats[r["sensor_id"]]
            v = r["value"]
            issue = None
            if v is None or (isinstance(v, float) and math.isnan(v)):
                st.missing += 1
                issue = "dropout" if st.missing >= 3 else None
                v = st.last_good
                r["status"] = "IMPUTED" if v is not None else "MISSING"
                b["imputed"] += 1
            else:
                st.missing = 0
                lo, hi = VALID_RANGES.get(r["metric"], (-1e12, 1e12))
                if not (lo <= v <= hi):
                    b["invalid"] += 1
                    issue = "out_of_range"
                    v = st.last_good
                    r["status"] = "INVALID"
                elif r["metric"] in VARYING and st.hist and abs(v) > 1e-9 and v == st.hist[-1]:
                    st.stuck += 1
                    if st.stuck >= 5:
                        issue = "stuck"
                        r["status"] = "SUSPECT"
                else:
                    st.stuck = 0
            r["z"] = st.z(v) if (v is not None and st.n > 12) else 0.0
            r["clean_value"] = v
            by_entity[r["entity_id"]].append(r)
            if issue:
                self.sensor_issues[r["sensor_id"]] = dict(sensor_id=r["sensor_id"], entity=r["entity_id"], metric=r["metric"],
                                                          issue=issue, since=self.sensor_issues.get(r["sensor_id"], {}).get("since", r["ts"]),
                                                          ts=r["ts"], domain=r["domain"])
            elif r["sensor_id"] in self.sensor_issues and self.sensor_issues[r["sensor_id"]]["issue"] != "spike":
                self.sensor_issues.pop(r["sensor_id"], None)

        # uncorroborated spike check: a sensor jumping > 6σ while its sibling metrics stay calm ⇒ sensor fault
        for ent, recs in by_entity.items():
            for r in recs:
                if abs(r["z"]) > 6 and r["status"] == "OK" and r["metric"] in SPIKE_CHECK:
                    siblings = [abs(o["z"]) for o in recs if o is not r and o["metric"] in SPIKE_CHECK | {"density", "queue_length", "hvac_power"}]
                    if len(siblings) >= 2 and max(siblings) < 1.5:
                        b["outliers_flagged"] += 1
                        r["status"] = "SUSPECT"
                        self.sensor_issues[r["sensor_id"]] = dict(sensor_id=r["sensor_id"], entity=ent, metric=r["metric"], issue="spike",
                                                                  since=r["ts"], ts=r["ts"], domain=r["domain"])
                        r["clean_value"] = self.stats[r["sensor_id"]].last_good
                elif r["sensor_id"] in self.sensor_issues and self.sensor_issues[r["sensor_id"]]["issue"] == "spike" and abs(r["z"]) < 3:
                    self.sensor_issues.pop(r["sensor_id"], None)

        # update statistics + entity state
        for ent, recs in by_entity.items():
            es = self.entity_state.setdefault(ent, {})
            for r in recs:
                v = r["clean_value"]
                st = self.stats[r["sensor_id"]]
                if v is not None and r["status"] in ("OK",):
                    st.update(v)
                    st.last_good = v
                if v is not None:
                    es[r["metric"]] = v
                    es[r["metric"] + "_avg"] = st.mean
                es["_type"], es["_domain"], es["_zone"], es["_ts"] = r["entity_type"], r["domain"], r["zone"], r["ts"]
                es["_x"], es["_y"] = r["x"], r["y"]

        # stale issue cleanup
        live_sensors = {r["sensor_id"] for r in clean}
        for sid in [s for s in self.sensor_issues if s not in live_sensors]:
            self.sensor_issues.pop(sid, None)

        self._spatial_aggregate()
        for k, v in b.items():
            self.totals[k] += v
        self.last_batch = dict(b)
        return dict(batch=dict(b), records=clean)

    @staticmethod
    def _norm_ts(ts) -> str:
        if isinstance(ts, (int, float)):
            return datetime.fromtimestamp(ts).isoformat(timespec="seconds")
        try:
            return pd.Timestamp(ts).tz_localize(None).isoformat(timespec="seconds") if "T" not in str(ts) else str(ts)[:19]
        except Exception:
            return str(ts)

    def _spatial_aggregate(self):
        zs: dict[str, dict] = defaultdict(lambda: defaultdict(list))
        for ent, es in self.entity_state.items():
            z = es.get("_zone")
            if not z:
                continue
            t = es.get("_type")
            if t == "road":
                zs[z]["road_speed"].append(es.get("speed", 0))
                zs[z]["road_occupancy"].append(es.get("occupancy", 0))
            elif t == "bin":
                zs[z]["bin_fill"].append(es.get("fill_level", 0))
            elif t == "building":
                zs[z]["building_kw"].append(es.get("power", 0))
            elif t == "zone":
                zs[z]["crowd_density"].append(es.get("crowd_density", 0))
        self.zone_state = {z: {k: (float(np.mean(v)) if k != "building_kw" else float(np.sum(v))) for k, v in d.items()} for z, d in zs.items()}

    def get(self, entity: str, metric: str, default=None):
        return self.entity_state.get(entity, {}).get(metric, default)

    def summary(self) -> dict:
        return dict(totals=dict(self.totals), last_batch=self.last_batch, sensors_tracked=len(self.stats),
                    sensor_issues=list(self.sensor_issues.values())[:50], zone_state=self.zone_state)


# ------------------------------------------------------------------------------------------
# Offline (history) processing used by the ML pipeline
# ------------------------------------------------------------------------------------------
def clean_frame(df: pd.DataFrame, ts_col: str = "timestamp", key_cols: tuple = ()) -> pd.DataFrame:
    df = df.drop_duplicates()
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.dropna(subset=[ts_col]).sort_values(list(key_cols) + [ts_col])
    num = df.select_dtypes("number").columns
    if key_cols:
        df[num] = df.groupby(list(key_cols))[num].transform(lambda s: s.interpolate(limit=6).ffill().bfill())
    else:
        df[num] = df[num].interpolate(limit=6).ffill().bfill()
    return df


def time_features(ts: pd.Series) -> pd.DataFrame:
    h = ts.dt.hour + ts.dt.minute / 60
    return pd.DataFrame({"hour_sin": np.sin(2 * np.pi * h / 24), "hour_cos": np.cos(2 * np.pi * h / 24),
                         "weekend": (ts.dt.weekday >= 5).astype(int), "hour": h})


def add_rolling(df: pd.DataFrame, col: str, by: str, windows=(3, 12)) -> pd.DataFrame:
    for w in windows:
        df[f"{col}_roll{w}"] = df.groupby(by)[col].transform(lambda s: s.rolling(w, min_periods=1).mean())
    df[f"{col}_diff"] = df.groupby(by)[col].diff().fillna(0)
    return df


# ------------------------------------------------------------------------------------------
# Configurable dataset connectors (CSV / JSON / REST). Normalise to the common record schema.
#   DATASET_CONNECTORS="traffic=csv:data/my_counts.csv;energy=json:data/meters.json;weather=rest:https://…"
#   Column mapping is read from data/connectors.json when present.
# ------------------------------------------------------------------------------------------
SCHEMA = ["timestamp", "location", "entity", "measurement", "value", "status", "source"]


class DatasetConnector:
    def __init__(self, domain: str, kind: str, target: str, mapping: dict | None = None):
        self.domain, self.kind, self.target = domain, kind, target
        self.mapping = mapping or {}

    def load(self) -> pd.DataFrame:
        if self.kind == "csv":
            df = pd.read_csv(self._path())
        elif self.kind == "json":
            df = pd.json_normalize(json.loads(Path(self._path()).read_text()))
        elif self.kind == "rest":
            req = urllib.request.Request(self.target, headers={"User-Agent": "CityOS/1.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:   # only public, authorised endpoints
                payload = json.loads(resp.read().decode())
            df = pd.json_normalize(payload if isinstance(payload, list) else payload.get(self.mapping.get("records_key", "data"), payload))
        else:
            raise ValueError(f"unsupported connector kind {self.kind}")
        df = df.rename(columns=self.mapping.get("columns", {}))
        for c in SCHEMA:
            if c not in df.columns:
                df[c] = {"status": "OK", "source": f"{self.kind}:{self.domain}"}.get(c)
        df = clean_frame(df[SCHEMA].copy(), "timestamp")
        return df

    def _path(self) -> str:
        p = Path(self.target)
        return str(p if p.is_absolute() else ROOT / p)


def configured_connectors() -> list[DatasetConnector]:
    out = []
    mapping_file = settings.data_dir / "connectors.json"
    mappings = json.loads(mapping_file.read_text()) if mapping_file.exists() else {}
    for part in filter(None, (s.strip() for s in settings.dataset_connectors.split(";"))):
        try:
            dom, spec = part.split("=", 1)
            kind, target = spec.split(":", 1)
            out.append(DatasetConnector(dom.strip(), kind.strip(), target.strip(), mappings.get(dom.strip())))
        except ValueError:
            continue
    return out


def fetch_open_meteo(lat: float, lon: float) -> dict | None:
    """Optional public weather connector (Open-Meteo, no key). Returns None when unreachable."""
    url = (f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
           "&current=temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,cloud_cover")
    try:
        with urllib.request.urlopen(url, timeout=8) as r:
            cur = json.loads(r.read().decode())["current"]
        return dict(temperature=cur["temperature_2m"], humidity=cur["relative_humidity_2m"], rainfall=cur["precipitation"],
                    wind=cur["wind_speed_10m"], cloud=cur["cloud_cover"] / 100)
    except Exception:
        return None
