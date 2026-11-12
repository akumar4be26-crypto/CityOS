"""
Digital twin + synthetic sensor simulator for Patiala, Punjab (schematic layout: real place names on a
simplified grid; Badi Nadi as the river; climate follows Patiala's monthly normals).

The twin is a small but physically-motivated model:
  * Traffic   – road graph, BPR link speeds, HCM/Webster intersection delay, signal plans,
                accidents, closures, diversion & rerouting, emergency corridors.
  * Safety    – zone crowd density, incidents, abnormal activity, emergency units.
  * Waste     – smart bins with fill dynamics, illegal dumping, contamination, collection trucks.
  * Energy    – buildings (HVAC / lighting / plug loads), solar, battery, EV charging, grid import.
  * Infra     – bridges, streetlights, water mains, pumps, transformers, pavement with degradation
                and fault progression; maintenance crews.
  * Weather   – synthetic diurnal temperature, rain episodes, wind, alerts.

Actions taken by the agent change the twin's state (signal splits, routes, dispatches, set-points…),
so verification measures real simulated effects instead of assumed ones.
Everything in here is SIMULATED – nothing touches physical infrastructure.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import networkx as nx
import numpy as np

from .config import settings

KM_PER_PX = 0.0048
NCOLS, NROWS = 6, 4
X0, DX, Y0, DY = 90, 164, 70, 150
RIVER_Y = Y0 + DY * 1.5            # river between row 1 and row 2
BRIDGE_COLS = (1, 3, 5)


def g(h: float, mu: float, s: float) -> float:
    d = min(abs(h - mu), 24 - abs(h - mu))
    return math.exp(-0.5 * (d / s) ** 2)


# --------------------------------------------------------------------------------------------
# Entities
# --------------------------------------------------------------------------------------------
@dataclass
class Zone:
    id: str
    name: str
    kind: str
    cols: tuple
    rows: tuple
    population: int
    public_area_m2: float
    traffic_w: float
    waste_w: float
    density: float = 0.2
    crowd_factor: float = 1.0          # crowd management action lowers this
    crowd_mgmt_until: int = -1
    event: str | None = None
    event_mult: float = 1.0
    event_until: int = -1
    anomaly_until: int = -1
    anomaly_mult: float = 1.0

    @property
    def bbox(self):
        x1 = X0 + DX * (self.cols[0] - 0.5)
        x2 = X0 + DX * (self.cols[1] + 0.5)
        y1 = Y0 + DY * (self.rows[0] - 0.5)
        y2 = Y0 + DY * (self.rows[1] + 0.5)
        return [x1, y1, x2, y2]


@dataclass
class Intersection:
    id: str
    r: int
    c: int
    x: float
    y: float
    zone: str
    g_ns: float = 0.5
    cycle: float = 90.0
    plan_source: str = "fixed-time"
    q_ns: float = 0.0
    q_ew: float = 0.0
    d_ns: float = 0.0
    d_ew: float = 0.0
    preempt_until: int = -1


@dataclass
class Road:
    id: str
    name: str
    u: str
    v: str
    axis: str                      # "EW" or "NS"
    length_km: float
    lanes: int
    capacity: float                # veh/h both directions
    free_speed: float              # km/h
    base_flow: float               # veh/h at profile 1.0
    zone: str
    is_bridge: bool = False
    closed: bool = False
    closed_reason: str | None = None
    cap_factor: float = 1.0        # accidents / potholes
    cap_reason: str | None = None
    cap_until: int = -1
    demand: float = 0.0
    flow: float = 0.0
    vc: float = 0.0
    speed: float = 0.0
    travel_min: float = 0.0
    density: float = 0.0
    occupancy: float = 0.0
    queue_m: float = 0.0
    noise: float = 0.0
    event_mult: float = 1.0


@dataclass
class Bin:
    id: str
    zone: str
    node: str
    x: float
    y: float
    waste_type: str
    capacity_l: float = 1100
    fill: float = 0.2              # 0..1 (>1 = overflowing)
    weight_kg: float = 20
    temperature: float = 28
    contamination: float = 0.1
    rate: float = 0.035            # base fraction per hour
    overflow_ticks: int = 0
    last_collected_tick: int = 0
    dumping_flag_until: int = -1
    contamination_event_until: int = -1


@dataclass
class Truck:
    id: str
    depot: str
    capacity_kg: float = 9000
    load_kg: float = 0
    node: str = ""
    status: str = "idle"           # idle | collecting | returning
    stops: list = field(default_factory=list)   # [(bin_id, eta_tick)]
    route_km: float = 0.0
    km_total: float = 0.0
    fuel_l: float = 0.0
    route_source: str = ""
    return_eta: int = -1
    missed: list = field(default_factory=list)


@dataclass
class Building:
    id: str
    name: str
    zone: str
    kind: str
    x: float
    y: float
    base_kw: float
    hvac_share: float
    light_kw: float
    hvac_offset: float = 0.0       # +°C set-point relaxation (AI action)
    hvac_offset_until: int = -1
    hvac_fault: float = 0.0        # extra HVAC fraction
    lighting_fault: bool = False
    plug_fault: float = 0.0
    power_kw: float = 0.0
    hvac_kw: float = 0.0
    lighting_kw: float = 0.0
    voltage: float = 230.0
    current: float = 0.0
    pf: float = 0.95
    energy_kwh: float = 0.0
    outage_factor: float = 1.0


@dataclass
class EnergyAsset:
    id: str
    name: str
    kind: str                      # solar | battery | ev_hub
    zone: str
    x: float
    y: float
    capacity: float                # kWp, kWh, or kW
    power_kw: float = 0.0
    soc: float = 0.5               # battery only
    max_kw: float = 0.0            # battery inverter / hub max
    curtail: float = 0.0           # solar curtailment fraction
    ev_shift: float = 0.0          # fraction of EV load deferred (AI)
    ev_shift_until: int = -1
    deferred_kwh: float = 0.0
    schedule: dict = field(default_factory=dict)   # battery: hour -> kW (+discharge / -charge)
    peak_guard: float = 0.0                         # battery controller: never let grid import exceed this when charging


@dataclass
class InfraAsset:
    id: str
    name: str
    kind: str                      # bridge | streetlight | water_main | pump | transformer | pavement
    zone: str
    x: float
    y: float
    criticality: int
    population_served: int
    repair_cost: float
    age_years: float
    condition: float
    usage: float = 1.0
    road: str | None = None
    status: str = "OPERATIONAL"    # OPERATIONAL | DEGRADED | FAILED | UNDER_REPAIR
    fault: float = 0.0             # hidden fault severity 0..1 (1 => failure)
    fault_rate: float = 0.0
    vibration: float = 1.0
    temperature: float = 30.0
    lamps_ok: float = 1.0
    flow: float = 0.0
    pressure: float = 4.5
    leak: float = 0.0
    work_order: str | None = None


@dataclass
class Unit:
    id: str
    kind: str                      # ambulance | fire | police | crew
    station: str
    node: str
    status: str = "available"      # available | en_route | on_scene | returning
    incident: str | None = None
    eta_tick: int = -1
    free_tick: int = -1
    path: list = field(default_factory=list)


@dataclass
class Incident:
    id: str
    type: str
    domain: str
    severity: int                  # 1..4
    zone: str
    x: float
    y: float
    node: str
    start_tick: int
    road: str | None = None
    status: str = "OPEN"           # OPEN | RESPONDING | ON_SCENE | CLEARED
    unit: str | None = None
    responded_tick: int = -1
    natural_end: int = -1
    end_tick: int = -1
    people_affected: int = 0
    description: str = ""


# --------------------------------------------------------------------------------------------
# Weather
# --------------------------------------------------------------------------------------------
class Weather:
    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.day_offset = 0.0
        self.temp = 26.0
        self.humidity = 60.0
        self.rain = 0.0
        self.wind = 10.0
        self.cloud = 0.2
        self.rain_left = 0
        self.heatwave = 0.0
        self.storm = 0.0
        self.alerts: list[str] = []

    # Patiala monthly climate normals (°C, mean) and rainy days per month — Jan … Dec
    MONTH_MEAN = [12.1, 15.9, 21.0, 27.0, 31.7, 31.9, 30.3, 29.2, 28.5, 25.5, 19.5, 14.5]
    RAIN_DAYS = [5, 5, 5, 5, 8, 9, 23, 23, 12, 2, 2, 2]

    def step(self, t: datetime, tick: int):
        h = t.hour + t.minute / 60
        m = t.month - 1
        monsoon = t.month in (7, 8, 9)
        if t.hour == 0 and t.minute == 0:
            self.day_offset = float(self.rng.normal(0, 1.5))
        amp = 4.5 if monsoon else 7.5
        self.temp = self.MONTH_MEAN[m] + self.day_offset + amp * math.sin((h - 9) / 24 * 2 * math.pi) + self.heatwave \
            + float(self.rng.normal(0, 0.25)) - (2.5 if self.rain > 0 else 0)
        if self.rain_left <= 0 and self.rng.random() < 0.006 * (self.RAIN_DAYS[m] / 8) * (1 + 3 * self.storm):
            self.rain_left = int(self.rng.integers(8, 48))
            self._rain_int = float(self.rng.uniform(1, 9) * (1.6 if monsoon else 1.0) * (1 + 2 * self.storm))
        if self.rain_left > 0:
            self.rain = max(0.0, self._rain_int * (0.6 + 0.4 * self.rng.random()))
            self.rain_left -= 1
            self.cloud = min(1, 0.75 + 0.2 * self.rng.random())
        else:
            self.rain = 0.0
            self.cloud = float(np.clip(self.cloud + self.rng.normal(0, 0.03), 0.3 if monsoon else 0.05, 0.75 if monsoon else 0.5))
        self.humidity = float(np.clip((80 if monsoon else 65) - 1.4 * (self.temp - 22) + (15 if self.rain else 0), 20, 100))
        self.wind = float(np.clip(10 + 6 * math.sin(h / 24 * 2 * math.pi) + self.rng.normal(0, 2) + 45 * self.storm, 0, 120))
        self.alerts = []
        if self.temp >= 42:
            self.alerts.append("EXTREME_HEAT")
        if self.rain >= 15:
            self.alerts.append("HEAVY_RAIN")
        if self.wind >= 60:
            self.alerts.append("HIGH_WIND")

    def snapshot(self):
        return dict(temperature=round(self.temp, 1), humidity=round(self.humidity, 0), rainfall=round(self.rain, 1),
                    wind=round(self.wind, 0), cloud=round(self.cloud, 2), alerts=list(self.alerts))


# --------------------------------------------------------------------------------------------
# The twin
# --------------------------------------------------------------------------------------------
class CityTwin:
    name = settings.city_name

    def __init__(self, seed: int | None = None, start: datetime | None = None, light: bool = False):
        self.rng = np.random.default_rng(settings.random_seed if seed is None else seed)
        self.sensor_rng = np.random.default_rng((settings.random_seed if seed is None else seed) + 101)  # sensor noise only
        self.dt_min = settings.sim_minutes_per_tick
        self.tick = 0
        if start is None:
            now = datetime.now()
            start = now.replace(hour=6, minute=0, second=0, microsecond=0)
        self.t0 = start
        self.t = start
        self.light = light                        # light mode = no sensor records (fast history)
        self.weather = Weather(self.rng)
        self.event_rate = settings.event_rate
        self.ground_truth: list[dict] = []        # injected anomalies, for precision/recall
        self.log: list[dict] = []                 # physical events (for event stream)
        self.incidents: dict[str, Incident] = {}
        self._ids = 0
        self.sensor_faults: dict[str, dict] = {}  # sensor_id -> fault
        self._last_values: dict[str, float] = {}
        self.reroutes: dict[str, dict] = {}       # edge_id -> {fraction, paths, until, source}
        self.diversions: dict[str, dict] = {}     # closed edge -> AI diversion plan
        self.corridors: list[dict] = []
        self.scenario: dict[str, float] = {}      # active what-if multipliers
        self.energy_totals = dict(kwh=0.0, cost=0.0, co2=0.0, peak_kw=0.0, day_peak_kw=0.0)
        self.waste_totals = dict(collected_kg=0.0, overflow_bin_ticks=0, truck_km=0.0, fuel_l=0.0, landfill_kg=0.0)
        self.traffic_totals = dict(delay_veh_h=0.0)
        self._build()

    # ------------------------------------------------------------------ build
    def _nid(self, prefix: str) -> str:
        self._ids += 1
        return f"{prefix}-{self._ids:04d}"

    def _zone_of(self, r: int, c: int) -> str:
        zc = c // 2
        zr = 0 if r < 2 else 1
        return ["Z1", "Z2", "Z3", "Z4", "Z5", "Z6"][zr * 3 + zc]

    def _build(self):
        rng = self.rng
        self.zones: dict[str, Zone] = {z.id: z for z in [
            Zone("Z1", "Thapar University & Bhadson Road", "campus", (0, 1), (0, 1), 21000, 52000, 0.95, 0.9),
            Zone("Z2", "Old City · Qila Mubarak", "cbd", (2, 3), (0, 1), 34000, 61000, 1.15, 1.1),
            Zone("Z3", "Leela Bhawan & Mall Road", "commercial", (4, 5), (0, 1), 26000, 48000, 1.05, 1.25),
            Zone("Z4", "Tripuri & Model Town", "residential", (0, 1), (2, 3), 48000, 70000, 0.85, 1.0),
            Zone("Z5", "Civil Lines · Mini Secretariat", "civic", (2, 3), (2, 3), 15000, 45000, 0.9, 0.8),
            Zone("Z6", "Focal Point Industrial Area", "industrial", (4, 5), (2, 3), 9000, 40000, 1.0, 0.95),
        ]}
        self.nodes: dict[str, Intersection] = {}
        for r in range(NROWS):
            for c in range(NCOLS):
                nid = f"I{r}{c}"
                self.nodes[nid] = Intersection(nid, r, c, X0 + c * DX, Y0 + r * DY, self._zone_of(r, c))
        ew_names = ["Bhupindra Rd", "Lower Mall Rd", "Badi Nadi Rd", "Sangrur Rd"]
        ns_names = ["Rajpura Rd", "Bhadson Rd", "Adalat Bazaar Rd", "Sheranwala Gate Rd", "Tripuri Rd", "Devigarh Rd"]
        self.roads: dict[str, Road] = {}
        k = 0
        for r in range(NROWS):
            for c in range(NCOLS - 1):
                k += 1
                u, v = f"I{r}{c}", f"I{r}{c + 1}"
                arterial = r in (1, 2)
                lanes = 4 if arterial else 2
                cap = lanes * 760
                zw = (self.zones[self.nodes[u].zone].traffic_w + self.zones[self.nodes[v].zone].traffic_w) / 2
                self.roads[f"R{k:02d}"] = Road(f"R{k:02d}", f"{ew_names[r]} · {ns_names[c]}–{ns_names[c + 1]}", u, v, "EW",
                                               DX * KM_PER_PX, lanes, cap, 50 if arterial else 40,
                                               cap * 0.74 * zw * float(rng.uniform(0.85, 1.08)), self.nodes[u].zone)
        for r in range(NROWS - 1):
            for c in range(NCOLS):
                crossing = r == 1
                if crossing and c not in BRIDGE_COLS:
                    continue
                k += 1
                u, v = f"I{r}{c}", f"I{r + 1}{c}"
                arterial = c in (2, 3) or crossing
                lanes = 4 if arterial else 2
                cap = lanes * 760
                zw = (self.zones[self.nodes[u].zone].traffic_w + self.zones[self.nodes[v].zone].traffic_w) / 2
                bf = cap * 0.74 * zw * float(rng.uniform(0.85, 1.08)) * (1.12 if crossing else 1.0)
                nm = f"{ns_names[c]} {'Bridge' if crossing else ''}".strip() + f" · {ew_names[r]}–{ew_names[r + 1]}"
                self.roads[f"R{k:02d}"] = Road(f"R{k:02d}", nm, u, v, "NS", DY * KM_PER_PX, lanes, cap,
                                               50 if arterial else 40, bf, self.nodes[u].zone, is_bridge=crossing)
        self.G = nx.Graph()
        for n in self.nodes.values():
            self.G.add_node(n.id, x=n.x, y=n.y)
        for e in self.roads.values():
            self.G.add_edge(e.u, e.v, id=e.id, length=e.length_km)
        self.edge_by_nodes = {frozenset((e.u, e.v)): e.id for e in self.roads.values()}
        self.incident_edges: dict[str, list[str]] = {n: [] for n in self.nodes}
        for e in self.roads.values():
            self.incident_edges[e.u].append(e.id)
            self.incident_edges[e.v].append(e.id)

        # ---- bins: 4 per zone
        self.bins: dict[str, Bin] = {}
        types = ["general", "recycling", "organic", "general"]
        for z in self.zones.values():
            zn = [n for n in self.nodes.values() if n.zone == z.id]
            for i in range(4):
                n = zn[i % len(zn)]
                bid = f"B-{z.id[1]}{i + 1}"
                wt = types[i]
                rate = (1 / 30) * z.waste_w * {"general": 1.0, "recycling": 0.75, "organic": 0.85}[wt] * float(rng.uniform(0.85, 1.2))
                self.bins[bid] = Bin(bid, z.id, n.id, n.x + float(rng.uniform(-38, 38)), n.y + float(rng.choice([-24, 24])), wt,
                                     fill=float(rng.uniform(0.1, 0.55)), rate=rate)
        depot = "I35"
        self.depot = depot
        self.trucks = {f"WT-{i}": Truck(f"WT-{i}", depot, node=depot) for i in (1, 2, 3)}
        # fixed (traditional) schedule: every bin once a day at 06:00, split in three static rounds
        ids = sorted(self.bins)
        self.fixed_rounds = {f"WT-{i + 1}": ids[i::3] for i in range(3)}

        # ---- buildings
        bdefs = [
            ("Z1", "Thapar University Library", "campus", 380, 0.45, 55), ("Z1", "Thapar University Hostels", "residential", 260, 0.35, 40),
            ("Z2", "District Courts Complex", "office", 720, 0.5, 90), ("Z2", "Adalat Bazaar Market", "retail", 540, 0.45, 70),
            ("Z3", "Leela Bhawan Market", "retail", 650, 0.48, 110), ("Z3", "Mall Road Complex", "retail", 300, 0.42, 45),
            ("Z4", "Tripuri Housing Blocks", "residential", 480, 0.35, 60), ("Z4", "Rajindra Hospital", "health", 850, 0.45, 80),
            ("Z5", "Mini Secretariat", "civic", 310, 0.45, 45), ("Z5", "City Water Works", "utility", 420, 0.15, 25),
            ("Z6", "Focal Point Manufacturing Unit", "industrial", 900, 0.25, 80), ("Z6", "Focal Point Cold Storage", "industrial", 520, 0.55, 20),
        ]
        self.buildings: dict[str, Building] = {}
        for i, (zid, nm, kind, base, hv, lk) in enumerate(bdefs):
            z = self.zones[zid]
            bx = z.bbox
            x = bx[0] + 40 + (i % 2) * (bx[2] - bx[0] - 80)
            y = (bx[1] + bx[3]) / 2 + (-30 if i % 2 else 30)
            bid = f"BLD-{i + 1:02d}"
            self.buildings[bid] = Building(bid, nm, zid, kind, x + DX * 0.5 * (1 if i % 2 == 0 else -1) * 0.5, y, base, hv, lk)

        self.energy_assets: dict[str, EnergyAsset] = {}
        for aid, nm, kind, zid, cap, mx in [
            ("PV-1", "Thapar University Solar", "solar", "Z1", 450, 0), ("PV-2", "Mall Road Rooftop PV", "solar", "Z3", 350, 0),
            ("PV-3", "Mini Secretariat Solar Canopy", "solar", "Z5", 300, 0), ("PV-4", "Focal Point PV Field", "solar", "Z6", 600, 0),
            ("BAT-1", "Civil Lines Grid Battery", "battery", "Z5", 2400, 600),
            ("EV-1", "Old City EV Hub", "ev_hub", "Z2", 0, 420), ("EV-2", "Tripuri EV Hub", "ev_hub", "Z4", 0, 360),
            ("EV-3", "Leela Bhawan EV Hub", "ev_hub", "Z3", 0, 300),
        ]:
            bx = self.zones[zid].bbox
            x = bx[0] + (bx[2] - bx[0]) * float(rng.uniform(0.3, 0.7))
            y = bx[1] + (bx[3] - bx[1]) * float(rng.uniform(0.25, 0.75))
            self.energy_assets[aid] = EnergyAsset(aid, nm, kind, zid, x, y, cap, max_kw=mx)

        # ---- infrastructure assets
        self.assets: dict[str, InfraAsset] = {}
        bridges = [e for e in self.roads.values() if e.is_bridge]
        for i, e in enumerate(bridges):
            u, v = self.nodes[e.u], self.nodes[e.v]
            self.assets[f"BR-{i + 1}"] = InfraAsset(f"BR-{i + 1}", f"{e.name.split(' ·')[0]}", "bridge", self.nodes[e.u].zone,
                                                    (u.x + v.x) / 2, RIVER_Y, 5, 60000, 850000, float(rng.uniform(18, 55)),
                                                    float(rng.uniform(62, 88)), road=e.id)
        for i, z in enumerate(self.zones.values()):
            bx = z.bbox
            self.assets[f"SL-{i + 1}"] = InfraAsset(f"SL-{i + 1}", f"Streetlights · {z.name}", "streetlight", z.id,
                                                    bx[0] + 26, bx[1] + 50, 3, z.population, 18000, float(rng.uniform(3, 15)),
                                                    float(rng.uniform(70, 95)))
        for i, zid in enumerate(["Z1", "Z2", "Z4", "Z5"]):
            z = self.zones[zid]
            bx = z.bbox
            self.assets[f"WM-{i + 1}"] = InfraAsset(f"WM-{i + 1}", f"Water Main · {z.name}", "water_main", zid,
                                                    bx[2] - 30, bx[3] - 30, 4, z.population, 120000, float(rng.uniform(15, 60)),
                                                    float(rng.uniform(55, 90)))
        for i, zid in enumerate(["Z5", "Z4", "Z6"]):
            bx = self.zones[zid].bbox
            self.assets[f"PU-{i + 1}"] = InfraAsset(f"PU-{i + 1}", f"Pump Station {i + 1}", "pump", zid,
                                                    bx[0] + 60, bx[3] - 25, 4, 40000, 60000, float(rng.uniform(5, 25)),
                                                    float(rng.uniform(60, 92)))
        for i, zid in enumerate(["Z2", "Z3", "Z6"]):
            bx = self.zones[zid].bbox
            self.assets[f"TX-{i + 1}"] = InfraAsset(f"TX-{i + 1}", f"Substation Transformer {i + 1}", "transformer", zid,
                                                    bx[2] - 60, bx[1] + 25, 5, 30000, 240000, float(rng.uniform(8, 35)),
                                                    float(rng.uniform(60, 92)))
        pav = list(self.roads.values())
        for i, idx in enumerate(rng.choice(len(pav), 6, replace=False)):
            e = pav[int(idx)]
            if e.is_bridge:
                continue
            u, v = self.nodes[e.u], self.nodes[e.v]
            self.assets[f"PV-R{i + 1}"] = InfraAsset(f"PV-R{i + 1}", f"Pavement · {e.name}", "pavement", e.zone,
                                                     (u.x + v.x) / 2, (u.y + v.y) / 2 + 8, 2, 8000, 45000,
                                                     float(rng.uniform(4, 20)), float(rng.uniform(45, 90)), road=e.id)

        # ---- units
        self.units = {
            "AMB-1": Unit("AMB-1", "ambulance", "I25", "I25"), "AMB-2": Unit("AMB-2", "ambulance", "I01", "I01"),
            "FIRE-1": Unit("FIRE-1", "fire", "I34", "I34"), "POL-1": Unit("POL-1", "police", "I12", "I12"),
            "POL-2": Unit("POL-2", "police", "I31", "I31"),
            "CREW-1": Unit("CREW-1", "crew", "I32", "I32"), "CREW-2": Unit("CREW-2", "crew", "I32", "I32"),
        }

    # ------------------------------------------------------------------ helpers
    @property
    def hour(self) -> float:
        return self.t.hour + self.t.minute / 60

    @property
    def weekend(self) -> bool:
        return self.t.weekday() >= 5

    def ticks_for(self, minutes: float) -> int:
        return max(1, int(round(minutes / self.dt_min)))

    def time_at(self, tick: int) -> datetime:
        return self.t0 + timedelta(minutes=tick * self.dt_min)

    def emit(self, kind: str, msg: str, **kw):
        self.log.append(dict(tick=self.tick, ts=self.t.isoformat(timespec="minutes"), kind=kind, message=msg, **kw))
        if len(self.log) > 600:
            self.log = self.log[-400:]

    def mark_truth(self, kind: str, entity: str, domain: str, duration_ticks: int):
        self.ground_truth.append(dict(kind=kind, entity=entity, domain=domain, start=self.tick, end=self.tick + duration_ticks))
        if len(self.ground_truth) > 2000:
            self.ground_truth = self.ground_truth[-1500:]

    def traffic_profile(self, h: float) -> float:
        if self.weekend:
            return 0.14 + 0.45 * g(h, 12.5, 3) + 0.42 * g(h, 19, 2)
        return 0.12 + 0.72 * g(h, 8.5, 1.1) + 0.33 * g(h, 13, 2.2) + 0.82 * g(h, 18, 1.4) + 0.1 * g(h, 21.5, 1.2)

    def path_edges(self, nodes: list[str]) -> list[str]:
        return [self.edge_by_nodes[frozenset((a, b))] for a, b in zip(nodes, nodes[1:])]

    def weight_fn(self, avoid: set | None = None, emergency: bool = False):
        avoid = avoid or set()

        def w(a, b, d):
            e = self.roads[d["id"]]
            if e.closed or e.id in avoid:
                return None
            tt = e.travel_min if e.travel_min > 0 else e.length_km / e.free_speed * 60
            if emergency:
                tt = e.length_km / max(e.speed, 8) * 60 / 1.35
            return tt
        return w

    def route(self, a: str, b: str, avoid: set | None = None, emergency: bool = False):
        try:
            p = nx.shortest_path(self.G, a, b, weight=self.weight_fn(avoid, emergency))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None, math.inf, 0.0
        edges = self.path_edges(p)
        w = self.weight_fn(avoid, emergency)
        tt = sum(w(None, None, {"id": eid}) or 0 for eid in edges)
        if not emergency:
            tt += sum(self._node_delay_for(eid, n) / 60 for eid, n in zip(edges, p[1:]))
        km = sum(self.roads[e].length_km for e in edges)
        return p, tt, km

    def _node_delay_for(self, eid: str, node: str) -> float:
        e, n = self.roads[eid], self.nodes[node]
        if n.preempt_until >= self.tick:
            return 0.0
        return n.d_ew if e.axis == "EW" else n.d_ns

    def nearest_node(self, x: float, y: float) -> str:
        return min(self.nodes.values(), key=lambda n: (n.x - x) ** 2 + (n.y - y) ** 2).id

    # ------------------------------------------------------------------ step
    def step(self) -> list[dict]:
        self.tick += 1
        self.t = self.time_at(self.tick)
        if self.t.hour == 0 and self.t.minute == 0:
            self.energy_totals["day_peak_kw"] = 0.0
        self.weather.step(self.t, self.tick)
        self._random_events()
        self._expire()
        self._traffic()
        self._safety()
        self._waste()
        self._energy()
        self._infrastructure()
        self._units()
        if self.light:
            return []
        return self._readings()

    # ------------------------------------------------------------------ random events
    def _p(self, per_day: float) -> float:
        return per_day * self.event_rate / (1440 / self.dt_min)

    def _random_events(self):
        rng = self.rng
        h = self.hour
        sc = self.scenario
        # accidents: proportional to traffic and rain
        for e in self.roads.values():
            if e.closed or e.cap_factor < 1:
                continue
            p = self._p(0.25) * (e.vc ** 1.5) * (1 + 0.25 * self.weather.rain) * sc.get("accident_mult", 1)
            if rng.random() < p:
                self.create_accident(e.id)
        # festival / crowd event in the evening
        if 16 <= h <= 19 and rng.random() < self._p(3.0):
            z = self.zones[str(rng.choice(["Z1", "Z2", "Z3", "Z5"]))]
            if z.event is None:
                self.start_event(z.id, "festival", float(rng.uniform(3.0, 4.6)), int(rng.integers(36, 60)))
        # abnormal activity (night)
        if (h >= 22 or h <= 4) and rng.random() < self._p(2.0):
            z = self.zones[str(rng.choice(list(self.zones)))]
            if z.anomaly_until < self.tick:
                z.anomaly_until = self.tick + int(rng.integers(6, 15))
                z.anomaly_mult = float(rng.uniform(3, 5))
                self.mark_truth("abnormal_activity", z.id, "safety", z.anomaly_until - self.tick)
        # illegal dumping
        if rng.random() < self._p(0.6):
            b = self.bins[str(rng.choice(list(self.bins)))]
            b.fill = min(1.3, b.fill + float(rng.uniform(0.3, 0.45)))
            b.weight_kg += float(rng.uniform(220, 380))
            b.dumping_flag_until = self.tick + 12
            self.mark_truth("illegal_dumping", b.id, "waste", 12)
            self.emit("physical", f"Unusual mass deposited at {b.id}", domain="waste", entity=b.id)
        # recycling contamination
        if rng.random() < self._p(0.4):
            rb = [b for b in self.bins.values() if b.waste_type == "recycling"]
            b = rb[int(rng.integers(len(rb)))]
            if b.contamination_event_until < self.tick:
                b.contamination_event_until = self.tick + int(rng.integers(60, 200))
                self.mark_truth("poor_segregation", b.id, "waste", b.contamination_event_until - self.tick)
        # energy faults
        if rng.random() < self._p(0.45):
            b = self.buildings[str(rng.choice(list(self.buildings)))]
            if b.hvac_fault == 0:
                b.hvac_fault = float(rng.uniform(0.45, 0.8))
                self.mark_truth("hvac_fault", b.id, "energy", 10_000)
        if (h >= 19 or h <= 5) and rng.random() < self._p(0.35):
            b = self.buildings[str(rng.choice(list(self.buildings)))]
            if not b.lighting_fault:
                b.lighting_fault = True
                self.mark_truth("excessive_lighting", b.id, "energy", 10_000)
        if (h >= 21 or h <= 5) and rng.random() < self._p(0.25):
            b = self.buildings[str(rng.choice(list(self.buildings)))]
            if b.plug_fault == 0:
                b.plug_fault = float(rng.uniform(0.25, 0.4))
                self.mark_truth("unexpected_consumption", b.id, "energy", 10_000)
        # infrastructure fault onset
        for a in self.assets.values():
            if a.fault == 0 and a.status == "OPERATIONAL":
                base = 0.07 * (1 + (100 - a.condition) / 60) * sc.get("infra_stress", 1)
                if rng.random() < self._p(base):
                    a.fault = 0.02
                    a.fault_rate = 1 / float(rng.uniform(10, 36)) / (60 / self.dt_min)   # reaches 1 in 10–36 h
                    self.mark_truth("infra_fault", a.id, "infrastructure", int(1 / a.fault_rate))
        # sensor faults (stuck / dropout / spike)
        srng = self.sensor_rng
        if not self.light and srng.random() < self._p(3):
            sid = self._random_sensor_id()
            if sid and sid not in self.sensor_faults:
                kind = str(srng.choice(["stuck", "dropout", "spike"]))
                dur = int(srng.integers(6, 30))
                self.sensor_faults[sid] = dict(kind=kind, until=self.tick + dur)
                self.mark_truth("sensor_failure", sid, "sensor", dur)

    def _random_sensor_id(self):
        pool = [f"S-{e}-speed" for e in self.roads] + [f"S-{b}-fill_level" for b in self.bins] + \
               [f"S-{b}-power" for b in self.buildings] + [f"S-{a}-vibration" for a in self.assets if self.assets[a].kind in ("bridge", "pump", "transformer")]
        return pool[int(self.sensor_rng.integers(len(pool)))] if pool else None

    def _expire(self):
        for e in self.roads.values():
            if e.cap_until >= 0 and e.cap_until <= self.tick:
                if e.cap_reason == "collision":
                    self.emit("physical", f"Collision on {e.name} cleared", domain="traffic", entity=e.id)
                e.cap_factor, e.cap_reason, e.cap_until = 1.0, None, -1
        for z in self.zones.values():
            if z.event and z.event_until <= self.tick:
                self.emit("physical", f"{z.event.title()} in {z.name} ended", domain="safety", entity=z.id)
                z.event, z.event_mult = None, 1.0
            if z.crowd_mgmt_until >= 0 and z.crowd_mgmt_until <= self.tick:
                z.crowd_factor, z.crowd_mgmt_until = 1.0, -1
        for k in [k for k, v in self.reroutes.items() if v["until"] <= self.tick]:
            del self.reroutes[k]
        self.corridors = [c for c in self.corridors if c["until"] > self.tick]
        for k in [k for k, v in self.sensor_faults.items() if v["until"] <= self.tick]:
            del self.sensor_faults[k]
        for b in self.buildings.values():
            if b.hvac_offset_until >= 0 and b.hvac_offset_until <= self.tick:
                b.hvac_offset, b.hvac_offset_until = 0.0, -1
        for a in self.energy_assets.values():
            if a.ev_shift_until >= 0 and a.ev_shift_until <= self.tick:
                a.ev_shift, a.ev_shift_until = 0.0, -1

    # ------------------------------------------------------------------ traffic
    def _traffic(self):
        h = self.hour
        prof = self.traffic_profile(h)
        w = self.weather
        wf = 1 + 0.04 * min(w.rain, 10) / 2
        cap_w = 1 - min(0.18, 0.02 * w.rain)
        sc = self.scenario.get("traffic_mult", 1.0) * self.scenario.get("population_mult", 1.0)
        rng = self.rng
        morning = h < 12
        for e in self.roads.values():
            e.noise = 0.85 * e.noise + float(rng.normal(0, 0.035))
            dirf = (1.10 if morning else 0.93) if e.axis == "EW" else (0.92 if morning else 1.08)
            zev = max(self.zones[self.nodes[e.u].zone].event_mult, self.zones[self.nodes[e.v].zone].event_mult)
            e.event_mult = 1 + 0.18 * (zev - 1)
            e.demand = max(0.0, e.base_flow * prof * dirf * wf * e.event_mult * sc * (1 + e.noise))
        flows = {eid: (0.0 if e.closed else e.demand) for eid, e in self.roads.items()}
        # closed roads → diverted demand
        for eid, e in self.roads.items():
            if not e.closed:
                continue
            plan = self.diversions.get(eid)
            if plan:   # AI diversion: spread along alternate paths (weights by spare capacity)
                for path, share in plan["paths"]:
                    for pe in path:
                        flows[pe] = flows.get(pe, 0) + e.demand * 0.92 * share
            else:      # organic diversion: drivers squeeze onto adjacent streets
                adj = [x for x in self.incident_edges[e.u] + self.incident_edges[e.v] if x != eid and not self.roads[x].closed]
                for a in adj:
                    flows[a] += e.demand * 0.92 / max(1, len(adj))
        # AI reroutes of congested (open) roads
        for eid, rr in self.reroutes.items():
            if self.roads[eid].closed:
                continue
            moved = self.roads[eid].demand * rr["fraction"]
            flows[eid] -= moved
            for path, share in rr["paths"]:
                for pe in path:
                    flows[pe] = flows.get(pe, 0) + moved * share
        delay_vh = 0.0
        for eid, e in self.roads.items():
            e.flow = 0.0 if e.closed else max(0.0, flows[eid])
            cap = e.capacity * e.cap_factor * cap_w
            e.vc = 0.0 if e.closed else e.flow / max(cap, 1)
            if e.closed:
                e.speed, e.travel_min, e.density, e.occupancy = 0.0, 0.0, 0.0, 0.0
                continue
            bpr = 1 + 0.15 * min(e.vc, 1.6) ** 4
            e.speed = e.free_speed / bpr
        # intersections – HCM uniform + incremental delay
        T = 0.25
        for n in self.nodes.values():
            qn = sum(self.roads[x].flow for x in self.incident_edges[n.id] if self.roads[x].axis == "NS") / 2
            qe = sum(self.roads[x].flow for x in self.incident_edges[n.id] if self.roads[x].axis == "EW") / 2
            ln = sum(self.roads[x].lanes / 2 for x in self.incident_edges[n.id] if self.roads[x].axis == "NS" and not self.roads[x].closed) or 1
            le = sum(self.roads[x].lanes / 2 for x in self.incident_edges[n.id] if self.roads[x].axis == "EW" and not self.roads[x].closed) or 1
            if n.plan_source == "AI-adaptive" and n.preempt_until < self.tick:
                # adaptive control: re-optimise splits/cycle every interval from smoothed approach volumes
                g_opt, c_opt = self.webster(0.6 * n.q_ns + 0.4 * qn, 0.6 * n.q_ew + 0.4 * qe, 1800 * ln * cap_w, 1800 * le * cap_w)
                n.g_ns, n.cycle = 0.6 * n.g_ns + 0.4 * g_opt, 0.6 * n.cycle + 0.4 * c_opt
            n.q_ns, n.q_ew = qn, qe
            n.d_ns = self._hcm_delay(qn, 1800 * ln * cap_w, n.g_ns, n.cycle, T)
            n.d_ew = self._hcm_delay(qe, 1800 * le * cap_w, 1 - n.g_ns, n.cycle, T)
            delay_vh += (qn * n.d_ns + qe * n.d_ew) / 3600 * self.dt_min / 60
        self.traffic_totals["delay_veh_h"] += delay_vh
        for e in self.roads.values():
            if e.closed:
                continue
            nd = (self._node_delay_for(e.id, e.u) + self._node_delay_for(e.id, e.v)) / 2
            e.travel_min = e.length_km / e.speed * 60 + nd / 60
            e.speed = e.length_km / (e.travel_min / 60)
            e.density = e.flow / max(e.speed, 1)
            e.occupancy = min(100.0, e.density / (110 * e.lanes) * 100)
            e.queue_m = (e.flow / 2) / 3600 * nd * 7.0

    @staticmethod
    def webster(q_ns: float, q_ew: float, s_ns: float, s_ew: float) -> tuple[float, float]:
        y_ns, y_ew = q_ns / s_ns, q_ew / s_ew
        Y = min(0.92, y_ns + y_ew)
        C = float(np.clip((1.5 * 8.0 + 5) / max(0.08, 1 - Y), 50, 140))
        g = float(np.clip(y_ns / max(1e-6, y_ns + y_ew), 0.22, 0.78)) if (y_ns + y_ew) > 0 else 0.5
        return g, C

    @staticmethod
    def _hcm_delay(q: float, s: float, g_frac: float, C: float, T: float) -> float:
        lost = 4.0
        ge = max(0.1, g_frac - lost / C)
        cap = s * ge
        if q <= 1:
            return 0.5 * C * (1 - ge) ** 2
        x = q / cap
        d1 = 0.5 * C * (1 - ge) ** 2 / max(0.05, 1 - min(1.0, x) * ge)
        d2 = 900 * T * ((x - 1) + math.sqrt((x - 1) ** 2 + 8 * 0.5 * x / (cap * T)))
        return min(d1 + d2, 600.0)

    def avg_intersection_delay(self, nodes: list[str] | None = None) -> float:
        nodes = nodes or list(self.nodes)
        num = sum(self.nodes[n].q_ns * self.nodes[n].d_ns + self.nodes[n].q_ew * self.nodes[n].d_ew for n in nodes)
        den = sum(self.nodes[n].q_ns + self.nodes[n].q_ew for n in nodes)
        return num / den if den else 0.0

    # ------------------------------------------------------------------ safety
    def _safety(self):
        h = self.hour
        rng = self.rng
        for z in self.zones.values():
            base = {
                "campus": 0.15 + 0.55 * g(h, 11, 3) + 0.3 * g(h, 16, 2.5),
                "cbd": 0.1 + 0.6 * g(h, 12.5, 2) + 0.45 * g(h, 18, 1.5),
                "commercial": 0.1 + 0.4 * g(h, 13, 2.5) + 0.7 * g(h, 19, 2),
                "residential": 0.12 + 0.3 * g(h, 8, 1.5) + 0.35 * g(h, 19.5, 2.5),
                "civic": 0.08 + 0.45 * g(h, 11.5, 2.5),
                "industrial": 0.06 + 0.3 * g(h, 9, 3) + 0.15 * g(h, 15, 2),
            }[z.kind]
            mult = z.event_mult if z.event else 1.0
            if z.anomaly_until >= self.tick:
                mult *= z.anomaly_mult
            dens = base * mult * z.crowd_factor * self.scenario.get("population_mult", 1) * (0.7 if self.weather.rain > 3 else 1)
            z.density = max(0.01, 0.7 * z.density + 0.3 * dens * (1 + float(rng.normal(0, 0.04))))
            # crowd incidents
            p = self._p(0.25) * (z.density ** 2) * (1.6 if z.event else 1) / max(0.6, z.crowd_factor)
            if rng.random() < p:
                kind = "medical" if rng.random() < 0.6 else "disturbance"
                self.create_incident(kind, "safety", z.id, sev=int(rng.integers(1, 4)))

    def create_incident(self, kind: str, domain: str, zone: str, sev: int = 2, node: str | None = None, road: str | None = None,
                        desc: str = "") -> Incident:
        z = self.zones[zone]
        if node is None:
            zn = [n for n in self.nodes.values() if n.zone == zone]
            node = zn[int(self.rng.integers(len(zn)))].id
        n = self.nodes[node]
        x, y = n.x + float(self.rng.uniform(-30, 30)), n.y + float(self.rng.uniform(-20, 20))
        if road:
            e = self.roads[road]
            x, y = (self.nodes[e.u].x + self.nodes[e.v].x) / 2, (self.nodes[e.u].y + self.nodes[e.v].y) / 2
        iid = self._nid("INC")
        inc = Incident(iid, kind, domain, sev, zone, x, y, node, self.tick, road=road,
                       natural_end=self.tick + self.ticks_for(float(self.rng.uniform(90, 150))),
                       people_affected=int(z.density * 400 + sev * 50), description=desc or kind.replace("_", " "))
        self.incidents[iid] = inc
        self.mark_truth("incident", iid, domain, inc.natural_end - self.tick)
        self.emit("physical", f"{inc.description.capitalize()} reported in {z.name}", domain=domain, entity=iid)
        return inc

    def create_accident(self, road_id: str, severity: int | None = None):
        e = self.roads[road_id]
        sev = severity or int(self.rng.choice([1, 2, 2, 3, 3, 4]))
        dur = self.ticks_for(float(self.rng.uniform(100, 160)))
        e.cap_factor, e.cap_reason, e.cap_until = 0.45, "collision", self.tick + dur
        inc = self.create_incident("traffic_collision", "traffic", e.zone, sev=sev, node=e.u, road=road_id,
                                   desc=f"collision on {e.name}")
        inc.natural_end = self.tick + dur
        return inc

    def start_event(self, zone: str, kind: str, mult: float, ticks: int):
        z = self.zones[zone]
        z.event, z.event_mult, z.event_until = kind, mult, self.tick + ticks
        self.mark_truth("crowd_event", zone, "safety", ticks)
        self.emit("physical", f"{kind.title()} started in {z.name}", domain="safety", entity=zone)

    # ------------------------------------------------------------------ waste
    def _waste(self):
        h = self.hour
        prof = 0.35 + 0.9 * g(h, 12, 3.5) + 0.6 * g(h, 19, 2)
        sc = self.scenario.get("waste_mult", 1.0) * self.scenario.get("population_mult", 1.0)
        dt_h = self.dt_min / 60
        rng = self.rng
        for b in self.bins.values():
            z = self.zones[b.zone]
            ev = (1 + 0.6 * (z.event_mult - 1)) if z.event else 1.0
            inc = b.rate * prof * ev * sc * dt_h * (1 + float(rng.normal(0, 0.15)))
            b.fill = min(1.35, b.fill + max(0.0, inc))
            dens = {"general": 0.11, "recycling": 0.06, "organic": 0.25}[b.waste_type]
            b.weight_kg += max(0.0, inc) * b.capacity_l * dens
            b.temperature = self.weather.temp + (3 + 6 * b.fill if b.waste_type == "organic" else 1 + b.fill) + float(rng.normal(0, 0.3))
            target = 0.42 if b.contamination_event_until >= self.tick else 0.11
            b.contamination = 0.9 * b.contamination + 0.1 * target if b.waste_type == "recycling" else 0.05
            if b.fill >= 1.0:
                b.overflow_ticks += 1
                self.waste_totals["overflow_bin_ticks"] += 1
        # traditional fixed schedule at 06:00 if a truck is idle
        if self.t.hour == 6 and self.t.minute < self.dt_min:
            for tid, ids in self.fixed_rounds.items():
                tr = self.trucks[tid]
                if tr.status == "idle":
                    self.dispatch_truck(tid, ids, source="fixed schedule")
        for tr in self.trucks.values():
            self._truck_step(tr)

    def dispatch_truck(self, tid: str, bin_ids: list[str], source: str = "AI route") -> dict:
        tr = self.trucks[tid]
        stops = []
        cur, t, km = tr.node or tr.depot, self.tick, 0.0
        for bid in bin_ids:
            b = self.bins[bid]
            p, tt, dkm = self.route(cur, b.node)
            if p is None:
                continue
            tt = max(tt, dkm / 25 * 60)  # trucks ≤ 25 km/h in urban collection
            t += self.ticks_for(tt + 4)
            km += dkm
            stops.append((bid, t))
            cur = b.node
        p, tt, dkm = self.route(cur, tr.depot)
        km += dkm if p else 0
        tr.stops = stops
        tr.status = "collecting" if stops else "idle"
        tr.route_km = km
        tr.return_eta = t + self.ticks_for(max(tt, dkm / 25 * 60)) if p else t + 6
        tr.route_source = source
        tr._ret_km = dkm  # type: ignore[attr-defined]
        return dict(truck=tid, stops=[s[0] for s in stops], km=round(km, 2), eta_end=self.time_at(tr.return_eta).isoformat(timespec="minutes"))

    def _truck_step(self, tr: Truck):
        if tr.status == "collecting" and tr.stops and tr.stops[0][1] <= self.tick:
            bid, _ = tr.stops.pop(0)
            b = self.bins[bid]
            prev = tr.node or tr.depot
            _, _, dkm = self.route(prev, b.node)
            tr.km_total += dkm
            fuel = dkm * settings.truck_l_per_km
            tr.fuel_l += fuel
            self.waste_totals["truck_km"] += dkm
            self.waste_totals["fuel_l"] += fuel
            tr.node = b.node
            if self.rng.random() < 0.02:     # missed collection (blocked access etc.)
                tr.missed.append(bid)
                self.mark_truth("missed_collection", bid, "waste", 24)
                self.emit("physical", f"{tr.id} could not service {bid}", domain="waste", entity=bid)
            else:
                tr.load_kg += b.weight_kg
                self.waste_totals["collected_kg"] += b.weight_kg
                if b.waste_type != "recycling" or b.contamination > 0.3:
                    self.waste_totals["landfill_kg"] += b.weight_kg
                b.fill, b.weight_kg, b.last_collected_tick = 0.02, 2.0, self.tick
            if not tr.stops:
                tr.status = "returning"
        elif tr.status == "returning" and tr.return_eta <= self.tick:
            dkm = getattr(tr, "_ret_km", 0.0)
            tr.km_total += dkm
            self.waste_totals["truck_km"] += dkm
            self.waste_totals["fuel_l"] += dkm * settings.truck_l_per_km
            tr.status, tr.node, tr.load_kg = "idle", tr.depot, 0.0

    # ------------------------------------------------------------------ energy
    def tariff(self, h: float) -> float:
        if 17 <= h < 21:
            return settings.tariff_peak
        if 7 <= h < 17 or 21 <= h < 23:
            return settings.tariff_shoulder
        return settings.tariff_offpeak

    def grid_intensity(self, h: float) -> float:
        return settings.grid_kgco2_per_kwh * (1.18 if 17 <= h < 22 else (0.82 if 10 <= h < 15 else 1.0))

    def _occupancy(self, kind: str, h: float) -> float:
        wk = self.weekend
        return {
            "campus": (0.15 + 0.85 * g(h, 13, 3.2)) * (0.5 if wk else 1),
            "residential": 0.45 + 0.35 * g(h, 7.5, 1.5) + 0.55 * g(h, 20.5, 2.5),
            "office": (0.08 + 0.92 * g(h, 13, 3)) * (0.25 if wk else 1),
            "retail": 0.1 + 0.9 * g(h, 15.5, 3.5),
            "health": 0.5 + 0.4 * g(h, 12, 4),
            "civic": (0.08 + 0.9 * g(h, 12.5, 2.8)) * (0.2 if wk else 1),
            "utility": 0.8,
            "industrial": (0.35 + 0.65 * g(h, 12, 4.5)) * (0.6 if wk else 1),
        }[kind]

    def _energy(self):
        h = self.hour
        w = self.weather
        rng = self.rng
        sc = self.scenario.get("energy_mult", 1.0) * self.scenario.get("population_mult", 1.0)
        daylight = max(0.0, math.sin((h - 6.2) / 12.6 * math.pi)) if 6.2 <= h <= 18.8 else 0.0
        load = 0.0
        for b in self.buildings.values():
            occ = self._occupancy(b.kind, h)
            temp_dev = max(0.0, w.temp - (23 + b.hvac_offset)) + 0.3 * max(0.0, 18 - w.temp)
            hvac = b.base_kw * b.hvac_share * (0.25 + 0.75 * occ) * (0.35 + temp_dev / 9) * (1 + b.hvac_fault)
            plug = b.base_kw * (1 - b.hvac_share) * (0.3 + 0.7 * occ) * (1 + b.plug_fault)
            lights_need = (1 - daylight * 0.8) * (0.15 + 0.85 * occ)
            lighting = b.light_kw * (1.0 if b.lighting_fault else lights_need)
            p = (hvac + plug + lighting) * sc * b.outage_factor * (1 + float(rng.normal(0, 0.02)))
            b.hvac_kw, b.lighting_kw, b.power_kw = hvac, lighting, max(0.0, p)
            b.pf = float(np.clip(0.96 - 0.12 * b.hvac_fault - 0.05 * b.plug_fault + rng.normal(0, 0.005), 0.7, 0.99))
            b.voltage = 230 - 6 * (b.power_kw / (b.base_kw * 1.6)) + float(rng.normal(0, 0.8))
            b.current = b.power_kw * 1000 / (math.sqrt(3) * 400 * b.pf)
            b.energy_kwh += b.power_kw * self.dt_min / 60
            load += b.power_kw
        # EV hubs
        ev_load = 0.0
        evm = self.scenario.get("ev_mult", 1.0) * self.scenario.get("population_mult", 1.0)
        for a in self.energy_assets.values():
            if a.kind != "ev_hub":
                continue
            demand = a.max_kw * (0.08 + 0.55 * g(h, 18.8, 1.6) + 0.3 * g(h, 12.5, 2.5)) * evm * (1 + float(rng.normal(0, 0.05)))
            demand = min(demand, a.max_kw * 1.6)
            shift = a.ev_shift if a.ev_shift_until >= self.tick else 0.0
            deferred = demand * shift
            a.deferred_kwh += deferred * self.dt_min / 60
            served_backlog = 0.0
            if a.deferred_kwh > 0 and (h >= 22 or h < 6 or (10 <= h < 15 and daylight > 0.5)) and shift == 0:
                served_backlog = min(a.deferred_kwh / (self.dt_min / 60), a.max_kw * 0.6)
                a.deferred_kwh -= served_backlog * self.dt_min / 60
            a.power_kw = demand - deferred + served_backlog
            ev_load += a.power_kw
        # solar
        solar = 0.0
        for a in self.energy_assets.values():
            if a.kind == "solar":
                a.power_kw = a.capacity * daylight * (1 - 0.7 * w.cloud) * 0.85 * (1 - a.curtail) * (1 + float(rng.normal(0, 0.02)))
                solar += a.power_kw
        net = load + ev_load - solar
        # battery
        bat = self.energy_assets["BAT-1"]
        sp = bat.schedule.get(self.t.hour)
        if sp is None:
            sp = 0.0                      # traditional: battery idle (inefficient usage)
        if sp < 0 and net < 0:
            sp = max(sp, net)
        if bat.peak_guard > 0:
            if sp < 0:                                  # charge only with headroom under the planned peak
                sp = -min(-sp, max(0.0, bat.peak_guard - net))
            shave_at = max(bat.peak_guard, self.energy_totals["day_peak_kw"] * 0.995)
            if net - sp > shave_at:                     # peak-ratchet: only shave new daily highs (keeps energy for the real peak)
                sp = max(sp, min(bat.max_kw, net - shave_at))
        sp = float(np.clip(sp, -bat.max_kw, bat.max_kw))
        e_dt = sp * self.dt_min / 60
        if e_dt > 0:
            e_dt = min(e_dt, (bat.soc - 0.1) * bat.capacity)
        else:
            e_dt = max(e_dt, -(0.95 - bat.soc) * bat.capacity / 0.92)
        bat.soc = float(np.clip(bat.soc - (e_dt if e_dt > 0 else e_dt * 0.92) / bat.capacity, 0.05, 0.98))
        bat.power_kw = e_dt / (self.dt_min / 60)
        # solar curtailment when exporting with full battery (poor utilisation)
        export = max(0.0, -(net - bat.power_kw))
        for a in self.energy_assets.values():
            if a.kind == "solar":
                a.curtail = 0.25 if export > 150 and bat.soc > 0.9 else max(0.0, a.curtail - 0.05)
        grid = max(0.0, net - bat.power_kw)
        self.grid_kw, self.load_kw, self.ev_kw, self.solar_kw = grid, load, ev_load, solar
        kwh = grid * self.dt_min / 60
        self.energy_totals["kwh"] += kwh
        self.energy_totals["cost"] += kwh * self.tariff(h)
        self.energy_totals["co2"] += kwh * self.grid_intensity(h)
        self.energy_totals["peak_kw"] = max(self.energy_totals["peak_kw"], grid)
        self.energy_totals["day_peak_kw"] = max(self.energy_totals["day_peak_kw"], grid)

    # ------------------------------------------------------------------ infrastructure
    def _infrastructure(self):
        rng = self.rng
        h = self.hour
        night = h >= 19 or h < 6
        stress = self.scenario.get("infra_stress", 1.0)
        for a in self.assets.values():
            use = 1.0
            if a.road:
                use = 0.5 + self.roads[a.road].vc
            a.usage = 0.97 * a.usage + 0.03 * use
            a.condition = max(0.0, a.condition - 0.0006 * a.usage * stress * self.dt_min / 5 - (0.02 * a.fault if a.fault else 0))
            if a.fault > 0 and a.status not in ("FAILED", "UNDER_REPAIR"):
                a.fault = min(1.0, a.fault + a.fault_rate * (1 + 0.5 * (self.weather.storm + (stress - 1))))
            if a.status == "UNDER_REPAIR":
                pass
            elif a.fault >= 1.0 and a.status != "FAILED":
                self._fail(a)
            elif a.status != "FAILED":
                a.status = "DEGRADED" if a.condition < 40 or a.fault > 0.5 else "OPERATIONAL"
            f = a.fault
            if a.kind in ("bridge", "pump", "transformer"):
                base_v = {"bridge": 1.2 + 1.8 * (a.usage - 0.5), "pump": 2.4, "transformer": 0.8}[a.kind]
                a.vibration = max(0.05, base_v * (1 + 0.006 * (100 - a.condition)) + 6.5 * f ** 1.6 + float(rng.normal(0, 0.12)))
                a.temperature = self.weather.temp + {"bridge": 0, "pump": 12, "transformer": 22}[a.kind] + 28 * f ** 2 + float(rng.normal(0, 0.4))
            elif a.kind == "streetlight":
                if a.status == "FAILED":
                    a.lamps_ok = 0.0
                else:
                    a.lamps_ok = float(np.clip(1 - 0.85 * f - (100 - a.condition) / 600, 0, 1))
                a.temperature = self.weather.temp
            elif a.kind == "water_main":
                pop = self.zones[a.zone].population
                demand = pop / 1000 * (0.4 + 0.6 * g(h, 7.5, 1.5) + 0.5 * g(h, 19.5, 2))
                a.leak = 25 * f if a.status != "FAILED" else 60
                a.flow = demand + a.leak + float(rng.normal(0, 0.5))
                a.pressure = float(4.8 - 0.00004 * a.flow ** 2 - 0.035 * a.leak + rng.normal(0, 0.03))
            elif a.kind == "pavement":
                a.temperature = self.weather.temp + 8 * (1 if 9 < h < 17 else 0)
        # streetlights affect safety risk at night
        self.dark_zones = {a.zone for a in self.assets.values() if a.kind == "streetlight" and a.lamps_ok < 0.6} if night else set()
        for a in self.assets.values():
            if a.kind == "transformer":
                for b in self.buildings.values():
                    if b.zone == a.zone:
                        b.outage_factor = 0.55 if a.status == "FAILED" else 1.0

    def _fail(self, a: InfraAsset):
        a.status = "FAILED"
        self.emit("physical", f"{a.name} FAILED", domain="infrastructure", entity=a.id)
        if a.kind == "bridge" and a.road:
            self.close_road(a.road, f"{a.name} structural failure")
        if a.kind == "pavement" and a.road:
            e = self.roads[a.road]
            e.cap_factor, e.cap_reason, e.cap_until = 0.6, "pothole", self.tick + 10_000

    def close_road(self, road_id: str, reason: str):
        e = self.roads[road_id]
        if not e.closed:
            e.closed, e.closed_reason = True, reason
            self.emit("physical", f"{e.name} closed — {reason}", domain="infrastructure", entity=road_id)

    def reopen_road(self, road_id: str):
        e = self.roads[road_id]
        if e.closed:
            e.closed, e.closed_reason = False, None
            self.diversions.pop(road_id, None)
            self.emit("physical", f"{e.name} reopened", domain="infrastructure", entity=road_id)

    # ------------------------------------------------------------------ units
    def _units(self):
        for u in self.units.values():
            if u.status == "en_route" and u.eta_tick <= self.tick:
                u.status = "on_scene"
                u.node = u.path[-1] if u.path else u.node
                if u.kind == "crew":
                    a = self.assets.get(u.incident or "")
                    if a:
                        a.status = "UNDER_REPAIR"
                        hours = {"bridge": 4, "water_main": 3, "pump": 2.5, "transformer": 3.5, "streetlight": 1.5, "pavement": 2}[a.kind]
                        u.free_tick = self.tick + self.ticks_for(hours * 60)
                    else:
                        u.free_tick = self.tick + self.ticks_for(75)   # building service visit
                elif (u.incident or "").startswith("standby:"):
                    u.free_tick = self.tick + self.ticks_for(180)
                else:
                    inc = self.incidents.get(u.incident or "")
                    if inc:
                        inc.status = "ON_SCENE"
                        inc.responded_tick = self.tick
                        clear = self.ticks_for({1: 15, 2: 25, 3: 35, 4: 50}[inc.severity])
                        u.free_tick = self.tick + clear
            elif u.status == "on_scene" and u.free_tick <= self.tick:
                if u.kind == "crew":
                    a = self.assets.get(u.incident or "")
                    if a:
                        self._repair(a, u.incident)
                    elif u.incident in self.buildings:
                        bd = self.buildings[u.incident]
                        bd.hvac_fault, bd.plug_fault = 0.0, 0.0
                        self.emit("physical", f"HVAC / plant service completed at {bd.name}", domain="energy", entity=bd.id)
                else:
                    inc = self.incidents.get(u.incident or "")
                    if inc:
                        self._clear_incident(inc)
                p, tt, _ = self.route(u.node, u.station)
                u.status, u.eta_tick, u.incident = "returning", self.tick + self.ticks_for(tt if p else 10), None
            elif u.status == "returning" and u.eta_tick <= self.tick:
                u.status, u.node = "available", u.station
        for inc in self.incidents.values():
            if inc.status in ("OPEN", "RESPONDING") and inc.natural_end <= self.tick:
                self._clear_incident(inc)
        # prune old cleared incidents
        old = [k for k, v in self.incidents.items() if v.status == "CLEARED" and v.end_tick < self.tick - 288]
        for k in old:
            del self.incidents[k]

    def _clear_incident(self, inc: Incident):
        inc.status, inc.end_tick = "CLEARED", self.tick
        if inc.road:
            e = self.roads[inc.road]
            if e.cap_reason == "collision":
                e.cap_factor, e.cap_reason, e.cap_until = 1.0, None, -1
        self.emit("physical", f"Incident {inc.id} ({inc.description}) cleared", domain=inc.domain, entity=inc.id)

    def _repair(self, a: InfraAsset, _):
        was_failed = a.status in ("FAILED", "UNDER_REPAIR")
        a.fault, a.fault_rate = 0.0, 0.0
        a.condition = min(96.0, a.condition + 28)
        a.status, a.work_order = "OPERATIONAL", None
        if a.kind == "pavement" and a.road:
            e = self.roads[a.road]
            if e.cap_reason == "pothole":
                e.cap_factor, e.cap_reason, e.cap_until = 1.0, None, -1
        if a.kind == "bridge" and a.road and self.roads[a.road].closed and was_failed:
            self.reopen_road(a.road)
        if a.kind == "pump":
            for b in self.buildings.values():
                if b.zone == a.zone and b.hvac_fault:
                    pass
        self.emit("physical", f"Maintenance completed on {a.name}", domain="infrastructure", entity=a.id)

    def dispatch_unit(self, unit_id: str, target_node: str, ref: str, corridor: bool = False) -> dict:
        u = self.units[unit_id]
        p, tt, km = self.route(u.node, target_node, emergency=u.kind != "crew")
        if p is None:
            return dict(ok=False, reason="no route")
        tt = max(tt, 1.5)                      # turnout + minimum travel
        if corridor:
            for n in p:
                self.nodes[n].preempt_until = self.tick + self.ticks_for(tt + 5)
            tt *= 0.78
            self.corridors.append(dict(nodes=p, until=self.tick + self.ticks_for(tt + 10), unit=unit_id))
        u.status, u.incident, u.path = "en_route", ref, p
        u.eta_tick = self.tick + self.ticks_for(tt)
        if ref in self.incidents:
            self.incidents[ref].status, self.incidents[ref].unit = "RESPONDING", unit_id
        return dict(ok=True, unit=unit_id, eta_min=round(tt, 1), km=round(km, 2), path=p)

    # ------------------------------------------------------------------ sensor readings
    def _readings(self) -> list[dict]:
        ts = self.t.isoformat(timespec="seconds")
        out: list[dict] = []
        rng = self.sensor_rng

        def rec(entity, etype, domain, zone, x, y, metric, value, unit, source="sim-iot"):
            sid = f"S-{entity}-{metric}"
            status = "OK"
            f = self.sensor_faults.get(sid)
            if f:
                if f["kind"] == "stuck":
                    value = self._last_values.get(sid, value)
                elif f["kind"] == "dropout":
                    value = None
                elif f["kind"] == "spike" and value is not None:
                    value = value * float(rng.choice([0.0, 4.0, 6.0]))
            elif value is not None and rng.random() < 0.002:
                value = None                               # random packet loss
            if value is not None and sid not in self._last_values or (f is None and value is not None):
                self._last_values[sid] = value
            out.append(dict(ts=ts, sensor_id=sid, entity_id=entity, entity_type=etype, domain=domain, zone=zone, x=round(x, 1),
                            y=round(y, 1), metric=metric, value=None if value is None else float(value), unit=unit,
                            status=status, source=source))
            if rng.random() < 0.003:
                out.append(dict(out[-1]))                    # duplicate message (exercise de-duplication)

        for e in self.roads.values():
            x = (self.nodes[e.u].x + self.nodes[e.v].x) / 2
            y = (self.nodes[e.u].y + self.nodes[e.v].y) / 2
            legacy = int(e.id[1:]) % 7 == 0
            rec(e.id, "road", "traffic", e.zone, x, y, "vehicle_count", e.flow * self.dt_min / 60, "veh/interval")
            rec(e.id, "road", "traffic", e.zone, x, y, "speed", e.speed / 1.609 if legacy else e.speed, "mph" if legacy else "km/h",
                "legacy-loop" if legacy else "sim-iot")
            rec(e.id, "road", "traffic", e.zone, x, y, "density", e.density, "veh/km")
            rec(e.id, "road", "traffic", e.zone, x, y, "occupancy", e.occupancy, "%")
            rec(e.id, "road", "traffic", e.zone, x, y, "queue_length", e.queue_m, "m")
        for z in self.zones.values():
            bx = z.bbox
            cx, cy = (bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2
            open_inc = sum(1 for i in self.incidents.values() if i.zone == z.id and i.status != "CLEARED")
            rec(z.id, "zone", "safety", z.id, cx, cy, "crowd_density", z.density, "p/m2", "cctv-metadata")
            rec(z.id, "zone", "safety", z.id, cx, cy, "incident_count", open_inc, "count", "dispatch-cad")
        for b in self.bins.values():
            rec(b.id, "bin", "waste", b.zone, b.x, b.y, "fill_level", b.fill * 100, "%", "smart-bin")
            rec(b.id, "bin", "waste", b.zone, b.x, b.y, "weight", b.weight_kg, "kg", "smart-bin")
            rec(b.id, "bin", "waste", b.zone, b.x, b.y, "temperature", b.temperature, "C", "smart-bin")
            rec(b.id, "bin", "waste", b.zone, b.x, b.y, "contamination", b.contamination * 100, "%", "smart-bin")
        for b in self.buildings.values():
            gen1 = b.id in ("BLD-03", "BLD-08")
            rec(b.id, "building", "energy", b.zone, b.x, b.y, "power", b.power_kw * (1000 if gen1 else 1), "W" if gen1 else "kW",
                "meter-gen1" if gen1 else "smart-meter")
            rec(b.id, "building", "energy", b.zone, b.x, b.y, "voltage", b.voltage, "V", "smart-meter")
            rec(b.id, "building", "energy", b.zone, b.x, b.y, "current", b.current, "A", "smart-meter")
            rec(b.id, "building", "energy", b.zone, b.x, b.y, "power_factor", b.pf, "", "smart-meter")
            rec(b.id, "building", "energy", b.zone, b.x, b.y, "energy_consumption", b.energy_kwh, "kWh", "smart-meter")
            rec(b.id, "building", "energy", b.zone, b.x, b.y, "hvac_power", b.hvac_kw, "kW", "bms")
            rec(b.id, "building", "energy", b.zone, b.x, b.y, "lighting_power", b.lighting_kw, "kW", "bms")
        for a in self.energy_assets.values():
            rec(a.id, a.kind, "energy", a.zone, a.x, a.y, "battery_power" if a.kind == "battery" else "power", a.power_kw, "kW", "der-gateway")
            if a.kind == "battery":
                rec(a.id, a.kind, "energy", a.zone, a.x, a.y, "soc", a.soc * 100, "%", "der-gateway")
        for a in self.assets.values():
            if a.kind in ("bridge", "pump", "transformer"):
                rec(a.id, a.kind, "infrastructure", a.zone, a.x, a.y, "vibration", a.vibration, "mm/s", "shm-node")
                rec(a.id, a.kind, "infrastructure", a.zone, a.x, a.y, "temperature", a.temperature, "C", "shm-node")
            if a.kind == "streetlight":
                rec(a.id, a.kind, "infrastructure", a.zone, a.x, a.y, "lamps_ok", a.lamps_ok * 100, "%", "lighting-cms")
            if a.kind == "water_main":
                rec(a.id, a.kind, "infrastructure", a.zone, a.x, a.y, "flow", a.flow, "L/s", "scada-sim")
                rec(a.id, a.kind, "infrastructure", a.zone, a.x, a.y, "pressure", a.pressure, "bar", "scada-sim")
            rec(a.id, a.kind, "infrastructure", a.zone, a.x, a.y, "structural_condition", a.condition, "index", "inspection-db")
            rec(a.id, a.kind, "infrastructure", a.zone, a.x, a.y, "operating_status",
                {"OPERATIONAL": 1, "DEGRADED": 0.5, "UNDER_REPAIR": 0.25, "FAILED": 0}[a.status], "state", "asset-mgmt")
        w = self.weather
        for m, v, u in [("temperature", w.temp, "C"), ("rainfall", w.rain, "mm/h"), ("humidity", w.humidity, "%"), ("wind", w.wind, "km/h")]:
            rec("WX-1", "weather_station", "weather", "Z5", 500, 300, m, v, u, "weather-sim")
        return out

    # ------------------------------------------------------------------ snapshots
    def clone(self) -> "CityTwin":
        """Fast deep copy for counterfactual simulation (drops logs / ground truth / sensor caches)."""
        heavy = dict(log=self.log, ground_truth=self.ground_truth, _last_values=self._last_values, sensor_faults=self.sensor_faults,
                     incidents=self.incidents)
        self.log, self.ground_truth, self._last_values, self.sensor_faults = [], [], {}, {}
        self.incidents = {k: v for k, v in heavy["incidents"].items() if v.status != "CLEARED"}
        try:
            c = copy.deepcopy(self)
        finally:
            for k, v in heavy.items():
                setattr(self, k, v)
        c.light = True
        return c

    def geometry(self) -> dict:
        return dict(
            city=self.name, viewbox=[0, 0, X0 * 2 + DX * (NCOLS - 1), Y0 * 2 + DY * (NROWS - 1)], river_y=RIVER_Y,
            zones=[dict(id=z.id, name=z.name, kind=z.kind, bbox=z.bbox, population=z.population) for z in self.zones.values()],
            nodes=[dict(id=n.id, x=n.x, y=n.y, zone=n.zone) for n in self.nodes.values()],
            roads=[dict(id=e.id, name=e.name, u=e.u, v=e.v, x1=self.nodes[e.u].x, y1=self.nodes[e.u].y, x2=self.nodes[e.v].x,
                        y2=self.nodes[e.v].y, lanes=e.lanes, bridge=e.is_bridge) for e in self.roads.values()],
            bins=[dict(id=b.id, x=b.x, y=b.y, zone=b.zone, type=b.waste_type) for b in self.bins.values()],
            buildings=[dict(id=b.id, name=b.name, x=b.x, y=b.y, zone=b.zone, kind=b.kind) for b in self.buildings.values()],
            energy_assets=[dict(id=a.id, name=a.name, kind=a.kind, x=a.x, y=a.y, zone=a.zone) for a in self.energy_assets.values()],
            assets=[dict(id=a.id, name=a.name, kind=a.kind, x=a.x, y=a.y, zone=a.zone, road=a.road) for a in self.assets.values()],
            depot=dict(node=self.depot, x=self.nodes[self.depot].x, y=self.nodes[self.depot].y),
            stations=[dict(id=u.id, kind=u.kind, node=u.station, x=self.nodes[u.station].x, y=self.nodes[u.station].y) for u in self.units.values()],
        )

    def live(self) -> dict:
        def unit_pos(u: Unit):
            n = self.nodes[u.node]
            if u.status == "en_route" and u.path:
                n = self.nodes[u.path[-1]]
            return n.x, n.y
        return dict(
            tick=self.tick, time=self.t.isoformat(timespec="minutes"), weather=self.weather.snapshot(),
            roads={e.id: dict(flow=round(e.flow), vc=round(e.vc, 3), speed=round(e.speed, 1), travel_min=round(e.travel_min, 2),
                              queue_m=round(e.queue_m), closed=e.closed, reason=e.closed_reason or e.cap_reason,
                              rerouted=e.id in self.reroutes or e.id in self.diversions) for e in self.roads.values()},
            nodes={n.id: dict(g_ns=round(n.g_ns, 2), cycle=round(n.cycle), d_ns=round(n.d_ns, 1), d_ew=round(n.d_ew, 1),
                              plan=n.plan_source, preempt=n.preempt_until >= self.tick) for n in self.nodes.values()},
            zones={z.id: dict(density=round(z.density, 3), event=z.event, dark=z.id in getattr(self, "dark_zones", set()),
                              managed=z.crowd_mgmt_until >= self.tick) for z in self.zones.values()},
            bins={b.id: dict(fill=round(b.fill, 3), weight=round(b.weight_kg), temp=round(b.temperature, 1),
                             contamination=round(b.contamination, 2), type=b.waste_type) for b in self.bins.values()},
            trucks={t.id: dict(status=t.status, node=t.node, x=self.nodes[t.node or t.depot].x, y=self.nodes[t.node or t.depot].y,
                               stops=[s[0] for s in t.stops], source=t.route_source, km=round(t.km_total, 1)) for t in self.trucks.values()},
            buildings={b.id: dict(power=round(b.power_kw, 1), hvac=round(b.hvac_kw, 1), lighting=round(b.lighting_kw, 1),
                                  pf=round(b.pf, 3), hvac_offset=b.hvac_offset) for b in self.buildings.values()},
            energy_assets={a.id: dict(power=round(a.power_kw, 1), soc=round(a.soc, 3), shift=a.ev_shift, curtail=round(a.curtail, 2))
                           for a in self.energy_assets.values()},
            grid=dict(grid_kw=round(getattr(self, "grid_kw", 0), 1), load_kw=round(getattr(self, "load_kw", 0), 1),
                      ev_kw=round(getattr(self, "ev_kw", 0), 1), solar_kw=round(getattr(self, "solar_kw", 0), 1),
                      day_peak_kw=round(self.energy_totals["day_peak_kw"], 1)),
            assets={a.id: dict(status=a.status, condition=round(a.condition, 1), vibration=round(a.vibration, 2),
                               temperature=round(a.temperature, 1), lamps_ok=round(a.lamps_ok, 2), pressure=round(a.pressure, 2),
                               flow=round(a.flow, 1), work_order=a.work_order) for a in self.assets.values()},
            units={u.id: dict(kind=u.kind, status=u.status, node=u.node, x=unit_pos(u)[0], y=unit_pos(u)[1], incident=u.incident,
                              path=u.path if u.status == "en_route" else []) for u in self.units.values()},
            incidents=[dict(id=i.id, type=i.type, severity=i.severity, zone=i.zone, x=i.x, y=i.y, status=i.status, unit=i.unit,
                            road=i.road, description=i.description, age_min=(self.tick - i.start_tick) * self.dt_min)
                       for i in self.incidents.values() if i.status != "CLEARED"],
            corridors=[c["nodes"] for c in self.corridors],
            scenario=dict(self.scenario),
        )
