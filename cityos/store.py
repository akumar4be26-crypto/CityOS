"""SQLite persistence (prototype). Schema mirrors the spec; swap the DSN for PostgreSQL/TimescaleDB/PostGIS in production."""
from __future__ import annotations

import json
import sqlite3
import threading

from .config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS sensor_data (ts TEXT, sensor TEXT, entity TEXT, domain TEXT, zone TEXT, x REAL, y REAL,
    measurement TEXT, value REAL, unit TEXT, status TEXT, source TEXT);
CREATE INDEX IF NOT EXISTS ix_sensor_entity ON sensor_data(entity, measurement, ts);
CREATE TABLE IF NOT EXISTS events (event_id TEXT PRIMARY KEY, type TEXT, domain TEXT, severity TEXT, location TEXT, ts TEXT,
    status TEXT, title TEXT, detail TEXT, priority REAL);
CREATE TABLE IF NOT EXISTS predictions (id INTEGER PRIMARY KEY AUTOINCREMENT, model TEXT, entity TEXT, prediction REAL,
    horizon_min REAL, confidence REAL, made_ts TEXT, target_tick INTEGER, actual_outcome REAL);
CREATE TABLE IF NOT EXISTS agent_log (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, tick INTEGER, agent TEXT, phase TEXT,
    message TEXT, ref TEXT);
CREATE TABLE IF NOT EXISTS actions (id TEXT PRIMARY KEY, ts TEXT, domain TEXT, type TEXT, target TEXT, title TEXT, risk TEXT,
    status TEXT, decision TEXT, verification TEXT, execution_mode TEXT, payload TEXT);
CREATE TABLE IF NOT EXISTS infrastructure (asset TEXT, ts TEXT, condition REAL, criticality INTEGER, maintenance_status TEXT,
    failure_probability REAL, rul_hours REAL);
CREATE TABLE IF NOT EXISTS model_metrics (ts TEXT, model TEXT, metrics TEXT);
"""


class Store:
    def __init__(self, path=None):
        self.path = str(path or settings.db_path)
        self.conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.lock = threading.Lock()
        self.conn.executescript(SCHEMA)
        self._n_sensor = self.conn.execute("SELECT COUNT(*) FROM sensor_data").fetchone()[0]

    def _exec(self, sql, rows=None, many=False):
        with self.lock:
            try:
                if many:
                    self.conn.execute("BEGIN")
                    self.conn.executemany(sql, rows)
                    self.conn.execute("COMMIT")
                else:
                    self.conn.execute(sql, rows or ())
            except sqlite3.Error:
                try:
                    self.conn.execute("ROLLBACK")
                except sqlite3.Error:
                    pass

    def sensors(self, records: list[dict]):
        rows = [(r["ts"], r["sensor_id"], r["entity_id"], r["domain"], r["zone"], r["x"], r["y"], r["metric"], r["clean_value"], r["unit"],
                 r["status"], r["source"]) for r in records]
        self._exec("INSERT INTO sensor_data VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows, many=True)
        self._n_sensor += len(rows)
        if self._n_sensor > settings.max_sensor_rows:
            self._exec("DELETE FROM sensor_data WHERE rowid IN (SELECT rowid FROM sensor_data ORDER BY rowid LIMIT ?)",
                       (settings.max_sensor_rows // 4,))
            self._n_sensor -= settings.max_sensor_rows // 4

    def event(self, f: dict, ts: str):
        self._exec("INSERT OR REPLACE INTO events VALUES (?,?,?,?,?,?,?,?,?,?)",
                   (f["key"], f["kind"], f["domain"], f["category"], f.get("zone") or "", f.get("first_seen", ts), f.get("status", "OPEN"),
                    f["title"], f["detail"], f["priority"]))

    def close_event(self, key: str):
        self._exec("UPDATE events SET status='RESOLVED' WHERE event_id=?", (key,))

    def log(self, e: dict):
        self._exec("INSERT INTO agent_log (ts,tick,agent,phase,message,ref) VALUES (?,?,?,?,?,?)",
                   (e["ts"], e["tick"], e["agent"], e["phase"], e["message"], e.get("ref", "")))

    def action(self, a: dict):
        self._exec("INSERT OR REPLACE INTO actions VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                   (a["id"], a["created"], a["domain"], a["type"], a["target"], a["title"], a["risk"], a["status"],
                    json.dumps(a.get("decision"), default=str), json.dumps(a.get("verification"), default=str), a["execution_mode"],
                    json.dumps(a.get("params"), default=str)))

    def prediction(self, p: dict, ts: str):
        self._exec("INSERT INTO predictions (model,entity,prediction,horizon_min,confidence,made_ts,target_tick,actual_outcome) VALUES (?,?,?,?,?,?,?,?)",
                   (p["model"], p["entity"], p["predicted"], p["horizon_min"], p.get("confidence"), ts, p["target_tick"], p["actual"]))

    def infra(self, rows: list[dict], ts: str):
        self._exec("INSERT INTO infrastructure VALUES (?,?,?,?,?,?,?)",
                   [(r["asset"], ts, r["condition"], r["criticality"], r["status"], r["failure_probability"], r["rul_hours"]) for r in rows], many=True)

    def metrics(self, ts: str, model: str, m: dict):
        self._exec("INSERT INTO model_metrics VALUES (?,?,?)", (ts, model, json.dumps(m, default=str)))

    def query(self, sql: str, args=()) -> list[dict]:
        with self.lock:
            cur = self.conn.execute(sql, args)
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]

    def sensor_series(self, entity: str, metric: str, limit: int = 288) -> list[dict]:
        rows = self.query("SELECT ts, value, status FROM sensor_data WHERE entity=? AND measurement=? ORDER BY rowid DESC LIMIT ?", (entity, metric, limit))
        return rows[::-1]
