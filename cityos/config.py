"""Central configuration. Every value can be overridden from the environment / .env file."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # python-dotenv is optional at import time so the engine can run in bare environments
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except Exception:  # pragma: no cover
    pass

ROOT = Path(__file__).resolve().parent.parent


def _f(name: str, default: float) -> float:
    try:
        return float(os.getenv(name) or default)
    except ValueError:
        return default


def _i(name: str, default: int) -> int:
    try:
        return int(os.getenv(name) or default)
    except ValueError:
        return default


def _b(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    # ---- paths
    data_dir: Path = ROOT / "data"
    model_dir: Path = ROOT / "models"
    frontend_dir: Path = ROOT / "frontend"
    db_path: Path = Path(os.getenv("DB_PATH") or str(ROOT / "data" / "cityos.db"))

    # ---- simulation clock
    tick_seconds: float = _f("TICK_SECONDS", 2.0)            # real seconds per tick
    sim_minutes_per_tick: int = _i("SIM_MINUTES_PER_TICK", 5)  # simulated minutes per tick
    random_seed: int = _i("RANDOM_SEED", 7)
    event_rate: float = _f("EVENT_RATE", 1.0)                # multiplier on random incident frequency
    history_days: int = _i("HISTORY_DAYS", 14)               # synthetic history used to train models

    # ---- LLM (provider-agnostic)
    llm_provider: str = os.getenv("LLM_PROVIDER", "none").lower()  # none | anthropic | openai | openai_compatible
    llm_model: str = os.getenv("LLM_MODEL", "")
    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    llm_base_url: str = os.getenv("LLM_BASE_URL", "")
    llm_min_interval_s: float = _f("LLM_MIN_INTERVAL_SECONDS", 30)
    llm_timeout_s: float = _f("LLM_TIMEOUT_SECONDS", 40)
    llm_max_tool_rounds: int = _i("LLM_MAX_TOOL_ROUNDS", 4)

    # ---- priority thresholds (Priority = Severity x Probability x Impact x Urgency, each 0..1 → score 0..100)
    priority_medium: float = _f("PRIORITY_MEDIUM", 8)
    priority_high: float = _f("PRIORITY_HIGH", 20)
    priority_critical: float = _f("PRIORITY_CRITICAL", 40)

    # ---- detection thresholds (adapted at runtime by the ADAPT step)
    z_threshold: float = _f("Z_THRESHOLD", 3.0)
    congestion_vc: float = _f("CONGESTION_VC_THRESHOLD", 0.85)
    overflow_horizon_h: float = _f("OVERFLOW_HORIZON_HOURS", 6)
    overflow_fill: float = _f("OVERFLOW_FILL", 0.80)
    peak_threshold_kw: float = _f("PEAK_THRESHOLD_KW", 0)   # 0 → derived automatically from history
    failure_prob_threshold: float = _f("FAILURE_PROB_THRESHOLD", 0.35)

    # ---- safety / authorization
    auto_approve_high_in_sim: bool = _b("AUTO_APPROVE_HIGH_IMPACT_IN_SIMULATION", False)
    approval_timeout_ticks: int = _i("APPROVAL_TIMEOUT_TICKS", 36)
    verification_ticks: int = _i("VERIFICATION_TICKS", 6)

    # ---- persistence
    persist_sensor_every: int = _i("PERSIST_SENSOR_EVERY_TICKS", 3)
    max_sensor_rows: int = _i("MAX_SENSOR_ROWS", 250_000)

    # ---- city
    city_name: str = os.getenv("CITY_NAME") or "Patiala"
    currency: str = os.getenv("CURRENCY") or "₹"

    # ---- economics / carbon (editable ASSUMPTIONS in rupees, used by calculate_cost / calculate_carbon / the battery LP)
    tariff_offpeak: float = _f("TARIFF_OFFPEAK_PER_KWH", 6.0)
    tariff_shoulder: float = _f("TARIFF_SHOULDER_PER_KWH", 7.5)
    tariff_peak: float = _f("TARIFF_PEAK_PER_KWH", 9.0)
    demand_charge_per_kw: float = _f("DEMAND_CHARGE_PER_KW", 250.0)     # per kW per month
    fuel_cost_per_l: float = _f("FUEL_COST_PER_L", 90.0)
    value_of_time_per_h: float = _f("VALUE_OF_TIME_PER_HOUR", 150.0)
    crew_cost_per_h: float = _f("CREW_COST_PER_HOUR", 500.0)
    truck_cost_per_km: float = _f("TRUCK_COST_PER_KM", 40.0)
    carbon_shadow_price: float = _f("CARBON_SHADOW_PRICE_PER_KG", 2.5)
    truck_l_per_km: float = _f("TRUCK_L_PER_KM", 0.55)
    grid_kgco2_per_kwh: float = _f("GRID_KGCO2_PER_KWH", 0.71)   # approx. Indian grid average
    diesel_kgco2_per_l: float = _f("DIESEL_KGCO2_PER_L", 2.68)
    car_idle_l_per_h: float = _f("CAR_IDLE_L_PER_HOUR", 0.8)
    petrol_kgco2_per_l: float = _f("PETROL_KGCO2_PER_L", 2.31)
    landfill_kgco2e_per_kg: float = _f("LANDFILL_KGCO2E_PER_KG", 0.58)

    # ---- optional external connectors (configurable; disabled by default)
    weather_source: str = os.getenv("WEATHER_SOURCE", "synthetic")  # synthetic | open-meteo
    weather_lat: float = _f("WEATHER_LAT", 30.34)    # Patiala
    weather_lon: float = _f("WEATHER_LON", 76.39)
    dataset_connectors: str = os.getenv("DATASET_CONNECTORS", "")  # e.g. "traffic=csv:data/my_traffic.csv;energy=rest:https://..."

    extra: dict = field(default_factory=dict)


settings = Settings()
settings.data_dir.mkdir(parents=True, exist_ok=True)
settings.model_dir.mkdir(parents=True, exist_ok=True)
