"""
Machine-learning layer (quantitative intelligence – the LLM never computes these numbers).

  * Statistical baselines   – hour-of-day × weekday/weekend mean/std per entity (Z-score)
  * Rolling deviation       – EWMA residual per entity
  * Isolation Forest        – multivariate anomaly scoring per domain
  * Forecasters             – traffic flow (15–120 min), city energy demand + solar (1–24 h),
                              bin fill trajectory / overflow time, crowding risk (1 h),
                              infrastructure failure probability (24 h) + remaining useful life
  * Feedback loop           – prediction ledger → actual outcome → MAE / RMSE / MAPE / P / R / F1
                              → bias correction + periodic retraining
Gradient boosting uses XGBoost when installed, otherwise scikit-learn's HistGradientBoosting.
"""
from __future__ import annotations

import math
import threading
import time
from collections import defaultdict, deque

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, IsolationForest

from .config import settings

try:  # optional
    from xgboost import XGBClassifier, XGBRegressor  # type: ignore
    HAVE_XGB = True
except Exception:  # pragma: no cover
    HAVE_XGB = False

KIND_CODES = {"bridge": 0, "streetlight": 1, "water_main": 2, "pump": 3, "transformer": 4, "pavement": 5}
ZONE_KIND_CODES = {"campus": 0, "cbd": 1, "commercial": 2, "residential": 3, "civic": 4, "industrial": 5}
TYPE_CODES = {"general": 0, "recycling": 1, "organic": 2}
VIB_NORMAL = {"bridge": 2.0, "pump": 2.6, "transformer": 0.9}
TEMP_OFFSET = {"bridge": 0, "pump": 12, "transformer": 22}
WATCH_DENSITY = 0.9


def regressor():
    if HAVE_XGB:
        return XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.07, subsample=0.9, n_jobs=2, verbosity=0)
    return HistGradientBoostingRegressor(max_iter=250, learning_rate=0.08, max_leaf_nodes=31, random_state=0)


def classifier(balanced: bool = True):
    if HAVE_XGB:
        return XGBClassifier(n_estimators=250, max_depth=5, learning_rate=0.08, subsample=0.9, n_jobs=2, verbosity=0)
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, class_weight="balanced" if balanced else None,
                                          random_state=0, l2_regularization=1.0, min_samples_leaf=40)


def tfeat(ts) -> tuple[float, float, int]:
    h = ts.hour + ts.minute / 60
    return math.sin(2 * math.pi * h / 24), math.cos(2 * math.pi * h / 24), int(ts.weekday() >= 5)


def reg_metrics(y, p) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    if len(y) == 0:
        return dict(n=0)
    err = p - y
    nz = np.abs(y) > 1e-6
    mape = float(np.mean(np.abs(err[nz]) / np.abs(y[nz])) * 100) if nz.any() else float("nan")
    return dict(n=int(len(y)), mae=float(np.mean(np.abs(err))), rmse=float(np.sqrt(np.mean(err ** 2))), mape=mape)


def cls_metrics(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * p * r / (p + r) if (p == p and r == r and p + r) else float("nan")
    return dict(tp=tp, fp=fp, fn=fn, precision=p, recall=r, f1=f1)


class CityML:
    def __init__(self):
        self.models: dict = {}
        self.baselines: dict = {}
        self.metrics: dict = {}                  # offline validation metrics
        self.bias: dict[str, float] = defaultdict(lambda: 1.0)
        self.ledger: list[dict] = []
        self.live_errors: dict[str, deque] = defaultdict(lambda: deque(maxlen=400))
        self.adaptation_log: deque = deque(maxlen=200)
        self.trained_at: str | None = None
        self.retraining = False
        self.lock = threading.Lock()
        self.version = 0

    # ================================================================ TRAINING
    def train(self, h: dict[str, pd.DataFrame]):
        t0 = time.time()
        self._baselines(h)
        self._train_traffic(h["traffic"])
        self._train_energy(h["energy"])
        self._train_waste(h["waste"])
        self._train_infra(h["infrastructure"])
        self._train_safety(h["safety"])
        self._train_anomaly(h)
        self.trained_at = pd.Timestamp.now().isoformat(timespec="seconds")
        self.version += 1
        self.metrics["train_seconds"] = round(time.time() - t0, 1)
        self.metrics["engine"] = "xgboost" if HAVE_XGB else "sklearn-histgb"
        self.save()

    def _baselines(self, h):
        tr = h["traffic"].copy()
        tr["hr"] = tr.timestamp.dt.hour
        tr["we"] = (tr.timestamp.dt.weekday >= 5).astype(int)
        open_ = tr[tr.closed == 0]
        self.baselines["traffic"] = open_.groupby(["road", "we", "hr"]).agg(flow_m=("flow", "mean"), flow_s=("flow", "std"),
                                                                            speed_m=("speed", "mean"), speed_s=("speed", "std")).to_dict("index")
        en = h["energy"].copy()
        en["hr"], en["we"] = en.timestamp.dt.hour, (en.timestamp.dt.weekday >= 5).astype(int)
        bl = {}
        for c in [c for c in en.columns if c.startswith("bld_")]:
            gb = en.groupby(["we", "hr"])[c].agg(["mean", "std"])
            for (we, hr), row in gb.iterrows():
                bl[(c[4:], we, hr)] = dict(m=row["mean"], s=row["std"])
        gb = en.assign(d=en.load_kw + en.ev_kw).groupby(["we", "hr"])["d"].agg(["mean", "std"])
        for (we, hr), row in gb.iterrows():
            bl[("CITY", we, hr)] = dict(m=row["mean"], s=row["std"])
        self.baselines["energy"] = bl
        sf = h["safety"].copy()
        sf["hr"], sf["we"] = sf.timestamp.dt.hour, (sf.timestamp.dt.weekday >= 5).astype(int)
        normal = sf[(sf.event == 0) & (sf.abnormal == 0)]
        self.baselines["safety"] = normal.groupby(["zone", "we", "hr"])["density"].agg(["mean", "std"]).rename(
            columns={"mean": "m", "std": "s"}).to_dict("index")
        daily_peak = en.set_index("timestamp").grid_kw.resample("1D").max().dropna()
        self.baselines["peak_kw_p80"] = float(daily_peak.quantile(0.8)) if len(daily_peak) else 6000.0

    # ---------------------------------------------------------------- traffic
    def _traffic_xy(self, df: pd.DataFrame, horizons=(3, 6, 12, 24)):
        df = df.sort_values(["road", "timestamp"]).reset_index(drop=True)
        X, Y, VC = [], [], []
        ts = df.timestamp
        hs = ts.dt.hour + ts.dt.minute / 60
        for hzn in horizons:
            fut = df.groupby("road").flow.shift(-hzn)
            fut_cap = df.groupby("road").capacity.shift(-hzn)
            fut_ts = ts + pd.to_timedelta(hzn * settings.sim_minutes_per_tick, unit="m")
            fh = fut_ts.dt.hour + fut_ts.dt.minute / 60
            m = fut.notna() & (df.closed == 0)
            part = pd.DataFrame({
                "base": df.base_flow, "ratio": df.flow / df.base_flow, "h_sin": np.sin(2 * np.pi * hs / 24), "h_cos": np.cos(2 * np.pi * hs / 24),
                "fh_sin": np.sin(2 * np.pi * fh / 24), "fh_cos": np.cos(2 * np.pi * fh / 24), "f_we": (fut_ts.dt.weekday >= 5).astype(int),
                "horizon": hzn * settings.sim_minutes_per_tick, "temp": df.temp, "rain": df.rain, "event": df.event_mult,
                "cap_ratio": df.capacity / (df.base_flow / 0.74),
            })[m]
            X.append(part)
            Y.append((fut / df.base_flow)[m])
            VC.append((fut / fut_cap)[m])
        return pd.concat(X), pd.concat(Y), pd.concat(VC)

    def _train_traffic(self, df):
        X, y, vc = self._traffic_xy(df)
        if len(X) > 160_000:
            idx = np.random.default_rng(0).choice(len(X), 160_000, replace=False)
            X, y, vc = X.iloc[idx], y.iloc[idx], vc.iloc[idx]
        n = int(len(X) * 0.8)
        m = regressor().fit(X.iloc[:n], y.iloc[:n])
        p = m.predict(X.iloc[n:])
        flow_true = y.iloc[n:] * X.base.iloc[n:]
        flow_pred = p * X.base.iloc[n:]
        self.metrics["traffic_flow"] = reg_metrics(flow_true, flow_pred)
        resid_vc = (flow_pred - flow_true) / (X.base.iloc[n:] / 0.74 * X.cap_ratio.iloc[n:].clip(lower=0.1))
        self.metrics["traffic_flow"]["vc_sigma"] = float(np.std(resid_vc))
        self.models["traffic"] = m.fit(X, y)
        self.models["traffic_cols"] = list(X.columns)

    def predict_flow(self, road, now, horizon_min: float, wx) -> float:
        hs, hc, _ = tfeat(now)
        fut = now + pd.Timedelta(minutes=horizon_min)
        fs, fc, fwe = tfeat(fut)
        x = pd.DataFrame([[road.base_flow, road.flow / road.base_flow if not road.closed else road.demand / road.base_flow, hs, hc, fs, fc, fwe,
                           horizon_min, wx.temp, wx.rain, road.event_mult,
                           road.capacity * road.cap_factor / (road.base_flow / 0.74)]], columns=self.models["traffic_cols"])
        r = float(self.models["traffic"].predict(x)[0])
        return max(0.0, r * road.base_flow * self.bias["traffic_flow"])

    def predict_flows(self, roads: list, now, horizons: list[float], wx) -> dict[tuple[str, float], float]:
        """Batched multi-horizon flow forecast for many roads (one model call)."""
        hs, hc, _ = tfeat(now)
        rows, keys = [], []
        for e in roads:
            for hm in horizons:
                fut = now + pd.Timedelta(minutes=hm)
                fs, fc, fwe = tfeat(fut)
                ratio = (e.demand if e.closed else e.flow) / e.base_flow
                rows.append([e.base_flow, ratio, hs, hc, fs, fc, fwe, hm, wx.temp, wx.rain, e.event_mult,
                             e.capacity * e.cap_factor / (e.base_flow / 0.74)])
                keys.append((e.id, hm))
        with self.lock:
            pr = self.models["traffic"].predict(pd.DataFrame(rows, columns=self.models["traffic_cols"]))
        b = self.bias["traffic_flow"]
        return {k: max(0.0, float(r) * roads_by_id_base * b) for k, r, roads_by_id_base in
                zip(keys, pr, [e.base_flow for e in roads for _ in horizons])}

    def congestion_probability(self, pred_vc: float) -> float:
        sigma = max(0.04, self.metrics.get("traffic_flow", {}).get("vc_sigma", 0.08))
        return float(1 - norm.cdf((settings.congestion_vc - pred_vc) / sigma))

    # ---------------------------------------------------------------- energy
    def _train_energy(self, df):
        df = df.sort_values("timestamp").reset_index(drop=True)
        demand = df.load_kw + df.ev_kw
        ts = df.timestamp
        hs = ts.dt.hour + ts.dt.minute / 60
        step = int(60 / settings.sim_minutes_per_tick)
        X, Y = [], []
        for hh in range(1, 25):
            k = hh * step
            fut = demand.shift(-k)
            fts = ts + pd.Timedelta(hours=hh)
            fh = fts.dt.hour + fts.dt.minute / 60
            m = fut.notna()
            X.append(pd.DataFrame({"h_sin": np.sin(2 * np.pi * hs / 24), "h_cos": np.cos(2 * np.pi * hs / 24),
                                   "fh_sin": np.sin(2 * np.pi * fh / 24), "fh_cos": np.cos(2 * np.pi * fh / 24),
                                   "f_we": (fts.dt.weekday >= 5).astype(int), "horizon": hh, "demand": demand, "temp": df.temp,
                                   "cloud": df.cloud})[m][::3])
            Y.append(fut[m][::3])
        X, Y = pd.concat(X), pd.concat(Y)
        n = int(len(X) * 0.8)
        m = regressor().fit(X.iloc[:n], Y.iloc[:n])
        self.metrics["energy_demand"] = reg_metrics(Y.iloc[n:], m.predict(X.iloc[n:]))
        self.models["energy"] = m.fit(X, Y)
        self.models["energy_cols"] = list(X.columns)
        # solar
        Xs = pd.DataFrame({"h_sin": np.sin(2 * np.pi * hs / 24), "h_cos": np.cos(2 * np.pi * hs / 24), "cloud": df.cloud, "rain": df.rain})
        n = int(len(Xs) * 0.8)
        ms = regressor().fit(Xs.iloc[:n], df.solar_kw.iloc[:n])
        self.metrics["solar"] = reg_metrics(df.solar_kw.iloc[n:], ms.predict(Xs.iloc[n:]))
        self.models["solar"] = ms.fit(Xs, df.solar_kw)

    def forecast_energy(self, now, demand_now: float, wx, hours: int = 24) -> list[dict]:
        hs, hc, _ = tfeat(now)
        rows, srows, times = [], [], []
        for hh in range(1, hours + 1):
            fut = now + pd.Timedelta(hours=hh)
            fs, fc, fwe = tfeat(fut)
            rows.append([hs, hc, fs, fc, fwe, hh, demand_now, wx.temp, wx.cloud])
            srows.append([fs, fc, wx.cloud, wx.rain])
            times.append(fut)
        d = self.models["energy"].predict(pd.DataFrame(rows, columns=self.models["energy_cols"])) * self.bias["energy_demand"]
        s = np.clip(self.models["solar"].predict(pd.DataFrame(srows, columns=["h_sin", "h_cos", "cloud", "rain"])), 0, None)
        s = np.where([(6.2 <= t.hour + t.minute / 60 <= 18.8) for t in times], s, 0)
        return [dict(time=t.isoformat(timespec="minutes"), hour=t.hour, demand_kw=float(a), solar_kw=float(b)) for t, a, b in zip(times, d, s)]

    # ---------------------------------------------------------------- waste
    def _train_waste(self, df):
        df = df.sort_values(["bin", "timestamp"]).reset_index(drop=True)
        df["inc"] = df.groupby("bin").fill.diff()
        dt_h = df.groupby("bin").timestamp.diff().dt.total_seconds() / 3600
        df["rate"] = df.inc / dt_h
        ok = df.inc.between(0, 0.15) & df.rate.notna() & (df.fill < 1.3)
        d = df[ok]
        tsf = d.timestamp - pd.Timedelta(minutes=7.5)
        hh = tsf.dt.hour + tsf.dt.minute / 60
        X = pd.DataFrame({"h_sin": np.sin(2 * np.pi * hh / 24), "h_cos": np.cos(2 * np.pi * hh / 24), "we": (tsf.dt.weekday >= 5).astype(int),
                          "zone": d.zone.str[1].astype(int), "type": d.waste_type.map(TYPE_CODES), "event": d.event_mult, "temp": d.temp})
        # per-bin long-run rate captures the bin's own usage level
        bin_rate = d.groupby("bin").rate.mean()
        X["bin_rate"] = d.bin.map(bin_rate).values
        n = int(len(X) * 0.8)
        m = regressor().fit(X.iloc[:n], d.rate.iloc[:n])
        self.metrics["waste_fill_rate"] = reg_metrics(d.rate.iloc[n:], m.predict(X.iloc[n:]))
        self.models["waste"] = m.fit(X, d.rate)
        self.models["waste_cols"] = list(X.columns)
        self.models["waste_bin_rate"] = bin_rate.to_dict()

    def forecast_bin(self, b, now, wx, event_mult: float = 1.0, hours: int = 12) -> dict:
        rows = []
        times = [now + pd.Timedelta(minutes=30 * k + 15) for k in range(hours * 2)]
        br = self.models["waste_bin_rate"].get(b.id, b.rate)
        for t in times:
            s, c, we = tfeat(t)
            rows.append([s, c, we, int(b.zone[1]), TYPE_CODES[b.waste_type], event_mult, wx.temp, br])
        rates = np.clip(self.models["waste"].predict(pd.DataFrame(rows, columns=self.models["waste_cols"])), 0, None) * self.bias["waste_fill"]
        traj = b.fill + np.cumsum(rates * 0.5)
        over = next((0.5 * (k + 1) for k, f in enumerate(traj) if f >= 1.0), None)
        if b.fill >= 1.0:
            over = 0.0
        path = {f"+{h}h": float(traj[h * 2 - 1]) for h in (1, 2, 4, 6, 8, 12) if h * 2 - 1 < len(traj)}
        horizon = settings.overflow_horizon_h
        sigma_h = 0.18 * (over or hours)
        p_over = 1.0 if over == 0 else (float(norm.cdf((horizon - over) / max(sigma_h, 0.25))) if over is not None else
                                         float(1 - norm.cdf((1.0 - traj[min(len(traj) - 1, int(horizon * 2) - 1)]) / 0.12)))
        return dict(current=float(b.fill), trajectory=path, overflow_hours=over, overflow_probability=p_over,
                    rate_per_h=float(np.mean(rates[:4])))

    # ---------------------------------------------------------------- infrastructure
    @staticmethod
    def infra_features(kind, condition, vibration, temperature, air_temp, lamps_ok, pressure, age, usage, crit, vib_slope) -> list:
        vr = vibration / VIB_NORMAL.get(kind, 1.0) if kind in VIB_NORMAL else 1.0
        te = (temperature - air_temp - TEMP_OFFSET.get(kind, 0)) if kind in TEMP_OFFSET else 0.0
        pdef = max(0.0, 4.4 - pressure) if kind == "water_main" else 0.0
        lo = lamps_ok if kind == "streetlight" else 1.0
        # static per-asset attributes (age, criticality) are deliberately excluded: with few assets they let the model
        # memorise individual assets instead of learning degradation signatures. Criticality enters the priority score instead.
        return [KIND_CODES[kind], condition, vr, te, lo, pdef, usage, vib_slope]

    INFRA_COLS = ["kind", "condition", "vib_ratio", "temp_excess", "lamps_ok", "pressure_def", "usage", "vib_slope"]

    def _train_infra(self, df):
        df = df.sort_values(["asset", "timestamp"]).reset_index(drop=True)
        failed = df.status.eq("FAILED")
        df["fail_time"] = df.timestamp.where(failed)
        df["next_fail"] = df.groupby("asset").fail_time.transform(lambda s: s.bfill())
        ttf = (df.next_fail - df.timestamp).dt.total_seconds() / 3600
        df["ttf"] = ttf
        df["label"] = (ttf <= 24).astype(int)
        df["vib_slope"] = df.groupby("asset").vibration.diff(3).fillna(0)
        d = df[~failed & ~df.status.eq("UNDER_REPAIR")]
        X = pd.DataFrame([self.infra_features(r.kind, r.condition, r.vibration, r.temperature, r.air_temp, r.lamps_ok, r.pressure,
                                              r.age_years, r.usage, r.criticality, r.vib_slope) for r in d.itertuples()], columns=self.INFRA_COLS)
        y = d.label.values
        n = int(len(X) * 0.75)
        c = classifier(balanced=False).fit(X.iloc[:n], y[:n])
        pr = c.predict_proba(X.iloc[n:])[:, 1]
        pred = pr >= settings.failure_prob_threshold
        tp = int(((pred == 1) & (y[n:] == 1)).sum()); fp = int(((pred == 1) & (y[n:] == 0)).sum()); fn = int(((pred == 0) & (y[n:] == 1)).sum())
        self.metrics["infra_failure"] = cls_metrics(tp, fp, fn) | dict(n=int(len(y) - n), positives=int(y[n:].sum()))
        self.models["infra"] = classifier(balanced=False).fit(X, y)
        r = d.ttf.clip(upper=240).fillna(240).values
        self.models["infra_rul"] = regressor().fit(X, r)

    def predict_failure(self, rows: list[list]) -> tuple[np.ndarray, np.ndarray]:
        X = pd.DataFrame(rows, columns=self.INFRA_COLS)
        p = self.models["infra"].predict_proba(X)[:, 1]
        rul = np.clip(self.models["infra_rul"].predict(X), 0.5, 240)
        return p, rul

    # ---------------------------------------------------------------- safety
    def _train_safety(self, df):
        df = df.sort_values(["zone", "timestamp"]).reset_index(drop=True)
        fut = df.groupby("zone").density.shift(-4)   # +1 h (15-min rows)
        m = fut.notna()
        ts = df.timestamp
        hh = ts.dt.hour + ts.dt.minute / 60
        X = pd.DataFrame({"kind": df.kind.map(ZONE_KIND_CODES), "h_sin": np.sin(2 * np.pi * hh / 24), "h_cos": np.cos(2 * np.pi * hh / 24),
                          "we": (ts.dt.weekday >= 5).astype(int), "density": df.density, "event": df.event, "rain": df.rain,
                          "d_diff": df.groupby("zone").density.diff().fillna(0)})[m]
        fts = ts + pd.Timedelta(hours=1)
        exp_f = np.array([self.baselines["safety"].get((z, int(w), int(hr)), {}).get("m", np.nan)
                          for z, w, hr in zip(df.zone, fts.dt.weekday >= 5, fts.dt.hour)])
        crowd = (fut > WATCH_DENSITY) | (fut > 2.0 * np.nan_to_num(exp_f, nan=10))
        y = crowd[m].astype(int).to_numpy().copy()
        n = int(len(X) * 0.75)
        if y.sum() < 5:
            y[np.argsort(-X.density.values)[:10]] = 1
        c = classifier().fit(X.iloc[:n], y[:n])
        pred = c.predict_proba(X.iloc[n:])[:, 1] >= 0.5
        tp = int(((pred == 1) & (y[n:] == 1)).sum()); fp = int(((pred == 1) & (y[n:] == 0)).sum()); fn = int(((pred == 0) & (y[n:] == 1)).sum())
        self.metrics["crowding_risk"] = cls_metrics(tp, fp, fn) | dict(n=int(len(y) - n))
        self.models["safety"] = c.fit(X, y)
        self.models["safety_cols"] = list(X.columns)

    def crowding_risk(self, kind, now, density, event, rain, d_diff) -> float:
        s, c, we = tfeat(now)
        x = pd.DataFrame([[ZONE_KIND_CODES[kind], s, c, we, density, int(event), rain, d_diff]], columns=self.models["safety_cols"])
        return float(self.models["safety"].predict_proba(x)[:, 1][0])

    # ---------------------------------------------------------------- anomaly (Isolation Forest)
    def _train_anomaly(self, h):
        rng = np.random.default_rng(0)
        tr = h["traffic"]
        tr = tr[tr.closed == 0].sample(min(60000, len(tr)), random_state=0)
        Xt = np.column_stack([tr.flow / tr.base_flow, tr.speed / 50, tr.occupancy / 100, tr.queue_m / 300, tr.vc])
        en = h["energy"]
        bcols = [c for c in en.columns if c.startswith("bld_")]
        rows = []
        for c in bcols:
            bid = c[4:]
            hr, we = en.timestamp.dt.hour.values, (en.timestamp.dt.weekday >= 5).astype(int).values
            exp = np.array([self.baselines["energy"].get((bid, w, hh), {}).get("m", np.nan) for w, hh in zip(we, hr)])
            night = ((hr >= 22) | (hr < 6)).astype(int)
            rows.append(np.column_stack([en[c].values / exp, night]))
        Xe = np.vstack(rows)
        Xe = Xe[np.isfinite(Xe).all(1)]
        Xe = Xe[rng.choice(len(Xe), min(50000, len(Xe)), replace=False)]
        inf = h["infrastructure"]
        inf = inf[inf.fault == 0]
        Xi = np.array([self.infra_features(r.kind, r.condition, r.vibration, r.temperature, r.air_temp, r.lamps_ok, r.pressure, r.age_years,
                                           r.usage, r.criticality, 0)[2:6] for r in inf.itertuples()])
        sf = h["safety"]
        sf = sf[sf.abnormal == 0]
        hr, we = sf.timestamp.dt.hour.values, (sf.timestamp.dt.weekday >= 5).astype(int).values
        exp = np.array([self.baselines["safety"].get((z, w, hh), {}).get("m", np.nan) for z, w, hh in zip(sf.zone, we, hr)])
        Xs = np.column_stack([sf.density / exp, ((hr >= 22) | (hr < 5)).astype(int), sf.density])
        Xs = Xs[np.isfinite(Xs).all(1)]
        self.models["if_traffic"] = IsolationForest(n_estimators=100, contamination=0.01, random_state=0).fit(Xt)
        self.models["if_energy"] = IsolationForest(n_estimators=100, contamination=0.01, random_state=0).fit(Xe)
        self.models["if_infra"] = IsolationForest(n_estimators=100, contamination=0.01, random_state=0).fit(Xi)
        self.models["if_safety"] = IsolationForest(n_estimators=100, contamination=0.01, random_state=0).fit(Xs)

    def if_score(self, domain: str, x: list) -> float:
        """> 0 ⇒ anomalous (negated sklearn decision_function)."""
        return float(self.if_scores(domain, [x])[0])

    def if_scores(self, domain: str, rows: list[list]) -> np.ndarray:
        if not rows:
            return np.zeros(0)
        return -self.models[f"if_{domain}"].decision_function(np.asarray(rows, float))

    def expected(self, domain: str, key: tuple) -> dict | None:
        return self.baselines.get(domain, {}).get(key)

    # ================================================================ FEEDBACK LOOP
    def record(self, model: str, entity: str, target_tick: int, predicted: float, horizon_min: float, confidence: float | None = None,
               now_tick: int = 0) -> dict:
        p = dict(model=model, entity=entity, target_tick=target_tick, predicted=float(predicted), horizon_min=horizon_min,
                 confidence=confidence, made_tick=now_tick, actual=None)
        self.ledger.append(p)
        if len(self.ledger) > 6000:
            self.ledger = [x for x in self.ledger if x["actual"] is None][-3000:]
        return p

    def resolve(self, tick: int, actual_fn) -> list[dict]:
        done = []
        for p in self.ledger:
            if p["actual"] is None and p["target_tick"] <= tick:
                a = actual_fn(p)
                if a is None:
                    p["actual"] = float("nan")
                    continue
                p["actual"] = float(a)
                self.live_errors[p["model"]].append((p["predicted"], p["actual"]))
                if not hasattr(self, "_fresh"):
                    self._fresh = defaultdict(int)
                self._fresh[p["model"]] += 1
                done.append(p)
        self.ledger = [p for p in self.ledger if p["actual"] is None or p["target_tick"] > tick - 300]
        self._adapt_bias()
        return done

    def _adapt_bias(self):
        """Prediction → real outcome → error → model update. Recorded predictions already include the bias in force
        when they were made, so the correction is multiplicative and only uses outcomes resolved since the last update."""
        for model, errs in self.live_errors.items():
            fresh = self._fresh.get(model, 0) if hasattr(self, "_fresh") else 0
            if len(errs) < 30 or fresh < 30:
                continue
            arr = np.array(errs)[-fresh:]
            self._fresh[model] = 0
            m = (arr[:, 0] > 1e-3) & (arr[:, 1] > 1e-3)
            if m.sum() < 20:
                continue
            ratio = float(np.exp(np.median(np.log(arr[m, 1] / arr[m, 0]))))
            old = self.bias[model]
            new = float(np.clip(old * ratio ** 0.3, 0.75, 1.3))
            if abs(new - old) > 0.004:
                self.bias[model] = new
                pred_sum, act_sum = float(arr[m, 0].sum()), float(arr[m, 1].sum())
                self.adaptation_log.append(dict(model=model, predicted=round(pred_sum), actual=round(act_sum),
                                                error_pct=round((act_sum - pred_sum) / max(pred_sum, 1e-6) * 100, 1),
                                                old_bias=round(old, 4), new_bias=round(new, 4),
                                                note="Bias correction updated for future predictions"))

    def live_metrics(self) -> dict:
        out = {}
        for model, errs in self.live_errors.items():
            if errs:
                a = np.array(errs)
                out[model] = reg_metrics(a[:, 1], a[:, 0]) | dict(bias=round(self.bias[model], 4))
        return out

    def retrain_async(self, frames: dict[str, pd.DataFrame]):
        if self.retraining:
            return False

        def job():
            self.retraining = True
            try:
                with self.lock:
                    pass
                tmp = CityML()
                tmp.baselines = self.baselines
                tmp._train_traffic(frames["traffic"])
                tmp._train_energy(frames["energy"])
                with self.lock:
                    for k in ("traffic", "traffic_cols", "energy", "energy_cols", "solar"):
                        self.models[k] = tmp.models[k]
                    self.metrics["traffic_flow"] = tmp.metrics["traffic_flow"]
                    self.metrics["energy_demand"] = tmp.metrics["energy_demand"]
                    self.bias["traffic_flow"] = 1.0
                    self.bias["energy_demand"] = 1.0
                    self.version += 1
                    self.trained_at = pd.Timestamp.now().isoformat(timespec="seconds")
                    self.adaptation_log.append(dict(model="traffic+energy", note=f"Models retrained on history + live data (v{self.version})"))
                    self.save()
            finally:
                self.retraining = False
        threading.Thread(target=job, daemon=True).start()
        return True

    # ================================================================ persistence
    def save(self):
        d = settings.model_dir
        joblib.dump(dict(m=self.models.get("traffic"), cols=self.models.get("traffic_cols")), d / "traffic_model.pkl")
        joblib.dump(dict(m=self.models.get("waste"), cols=self.models.get("waste_cols"), bin_rate=self.models.get("waste_bin_rate")), d / "waste_model.pkl")
        joblib.dump(dict(m=self.models.get("energy"), cols=self.models.get("energy_cols"), solar=self.models.get("solar")), d / "energy_model.pkl")
        joblib.dump({k: v for k, v in self.models.items() if k.startswith("if_")}, d / "anomaly_model.pkl")
        joblib.dump(dict(infra=self.models.get("infra"), rul=self.models.get("infra_rul"), safety=self.models.get("safety"),
                         safety_cols=self.models.get("safety_cols"), baselines=self.baselines, metrics=self.metrics,
                         trained_at=self.trained_at, version=self.version), d / "aux_models.pkl")

    def load(self) -> bool:
        d = settings.model_dir
        try:
            t = joblib.load(d / "traffic_model.pkl"); w = joblib.load(d / "waste_model.pkl")
            e = joblib.load(d / "energy_model.pkl"); a = joblib.load(d / "anomaly_model.pkl"); x = joblib.load(d / "aux_models.pkl")
        except Exception:
            return False
        if t.get("m") is None or x.get("infra") is None:
            return False
        self.models.update(traffic=t["m"], traffic_cols=t["cols"], waste=w["m"], waste_cols=w["cols"], waste_bin_rate=w["bin_rate"],
                           energy=e["m"], energy_cols=e["cols"], solar=e["solar"], infra=x["infra"], infra_rul=x["rul"],
                           safety=x["safety"], safety_cols=x["safety_cols"], **a)
        self.baselines, self.metrics, self.trained_at, self.version = x["baselines"], x["metrics"], x["trained_at"], x["version"]
        return True
