/* CityOS frontend — operational workspace for the autonomous city agent.
   No external libraries: SVG digital twin + lightweight canvas charts. */
(() => {
  "use strict";

  // ------------------------------------------------------------------ constants
  const ICONS = {
    traffic: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><rect x="7" y="2.5" width="10" height="19" rx="3.5"/><circle cx="12" cy="7.3" r="1.6"/><circle cx="12" cy="12" r="1.6"/><circle cx="12" cy="16.7" r="1.6"/></svg>',
    safety: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2.8 4.5 5.6v6.1c0 4.6 3.2 8.2 7.5 9.5 4.3-1.3 7.5-4.9 7.5-9.5V5.6Z"/><path d="m8.8 12.2 2.2 2.2 4.3-4.4"/></svg>',
    waste: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M4 6.5h16"/><path d="M9.5 6.5V4.2h5v2.3"/><path d="M6 6.5 7.2 20a1.6 1.6 0 0 0 1.6 1.4h6.4a1.6 1.6 0 0 0 1.6-1.4L18 6.5"/><path d="M10 11v6M14 11v6"/></svg>',
    energy: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M13.2 2.5 5 13.4h6.3l-1 8.1 8.2-11h-6.3Z"/></svg>',
    infrastructure: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M2.5 17.5h19"/><path d="M3.5 17.5c2.5-6 14.5-6 17 0"/><path d="M7 14.2v3.3M12 12.6v4.9M17 14.2v3.3"/><path d="M4.5 20.5h15"/></svg>',
    orchestrator: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><circle cx="12" cy="3.6" r="1.6"/><circle cx="19.6" cy="8.2" r="1.6"/><circle cx="17.6" cy="18.6" r="1.6"/><circle cx="6.4" cy="18.6" r="1.6"/><circle cx="4.4" cy="8.2" r="1.6"/><path d="M12 5.2V9M18.2 9l-3.4 1.6M16.4 17.4l-2.3-3M7.6 17.4l2.3-3M5.8 9l3.4 1.6"/></svg>',
  };
  const AGENTS = [
    { id: "traffic", name: "Traffic Agent", sub: "Mobility Intelligence", tabs: ["Live Traffic", "Predictions", "Incidents", "Actions"] },
    { id: "safety", name: "Safety Agent", sub: "Public Safety Intelligence", tabs: ["Zones", "Incidents", "Coverage", "Actions"] },
    { id: "waste", name: "Waste Agent", sub: "Waste & Collection Intelligence", tabs: ["Bins", "Forecasts", "Collection", "Actions"] },
    { id: "energy", name: "Energy Agent", sub: "Energy & Efficiency Intelligence", tabs: ["Demand", "Buildings", "Forecasts", "Optimization"] },
    { id: "infrastructure", name: "Infrastructure Agent", sub: "Asset & Maintenance Intelligence", tabs: ["Assets", "Maintenance", "Risk", "Actions"] },
    { id: "orchestrator", name: "City Orchestrator", sub: "Cross-Domain City Intelligence", tabs: ["Overview", "Coordination", "What-if", "Learning"] },
  ];
  const AG = Object.fromEntries(AGENTS.map((a) => [a.id, a]));
  const KPI_DEF = {
    traffic: [["avg_intersection_delay_s", "Avg intersection delay", "s"], ["mean_vc", "Mean v/c", ""], ["congested_roads", "Congested roads", ""], ["avg_speed_kmh", "Avg speed", "km/h"], ["ai_signal_plans", "Adaptive signals", ""]],
    safety: [["open_incidents", "Open incidents", ""], ["units_available", "Units available", ""], ["max_density", "Peak crowd density", "p/m²"], ["avg_response_coverage_min", "Avg response time", "min"]],
    waste: [["avg_fill_pct", "Avg bin fill", "%"], ["overflowing_bins", "Overflowing", ""], ["at_risk", "At overflow risk", ""], ["trucks_active", "Trucks active", ""], ["truck_km", "Fleet km", "km"]],
    energy: [["grid_kw", "Grid import", "kW"], ["solar_kw", "Solar", "kW"], ["ev_kw", "EV charging", "kW"], ["battery_soc_pct", "Battery", "%"], ["day_peak_kw", "Today's peak", "kW"]],
    infrastructure: [["avg_condition", "Avg condition", "/100"], ["failed", "Failed assets", ""], ["at_risk", "At risk", ""], ["work_orders", "Work orders", ""], ["crews_available", "Crews free", ""]],
  };
  const DOMAIN_LABEL = { traffic: "Traffic", safety: "Safety", waste: "Waste", energy: "Energy", infrastructure: "Infrastructure", environment: "Environment", reliability: "Service reliability", sensor: "Sensors", city: "City" };

  // ------------------------------------------------------------------ state
  const S = { snap: null, geo: null, agent: null, tab: 0, filter: "all", whatif: null, whatifBusy: false, fc: null, fcAt: 0, sheet: null, mapBuilt: false };

  // ------------------------------------------------------------------ helpers
  const $ = (s, r = document) => r.querySelector(s);
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const fmt = (v, d = 0) => (v == null || Number.isNaN(v) ? "—" : Number(v).toLocaleString(undefined, { maximumFractionDigits: d, minimumFractionDigits: d }));
  const pct = (v, d = 0) => (v == null ? "—" : `${fmt(v * 100, d)}%`);
  const hm = (ts) => (ts ? String(ts).slice(11, 16) : "");
  const col = (d) => `var(--c-${d === "city" ? "orchestrator" : d})`;
  const vcColor = (vc, closed) => (closed ? "#e5383b" : vc > 1 ? "#c81e24" : vc > 0.85 ? "#ff5a4f" : vc > 0.7 ? "#f59e0b" : vc > 0.5 ? "#a8c94a" : "#3fbf6a");
  const lvlColor = { NORMAL: "var(--ok)", WATCH: "var(--watch)", WARNING: "var(--warn)", CRITICAL: "var(--crit)" };
  const catColor = { LOW: "var(--ok)", MEDIUM: "var(--watch)", HIGH: "var(--warn)", CRITICAL: "var(--crit)" };
  const api = async (path, opts = {}) => {
    const r = await fetch(path, { headers: { "content-type": "application/json" }, ...opts });
    const j = await r.json().catch(() => ({}));
    if (!r.ok) throw Object.assign(new Error(j.detail || j.reason || r.statusText), { body: j });
    return j;
  };
  const svgEl = (tag, attrs = {}, parent) => {
    const e = document.createElementNS("http://www.w3.org/2000/svg", tag);
    for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
    if (parent) parent.appendChild(e);
    return e;
  };

  // ------------------------------------------------------------------ entry
  function renderEntry() {
    const s = S.snap;
    const box = $("#profiles");
    if (!box.children.length) {
      box.innerHTML = AGENTS.map((a) => `
        <button class="profile" role="listitem" data-agent="${a.id}" style="--c:${col(a.id)}">
          <div class="avatar glass">${ICONS[a.id]}${a.id === "orchestrator" ? '<span class="p-tag">DEFAULT</span>' : ""}</div>
          <div class="p-name">${a.name}</div><div class="p-sub">${a.sub}</div>
          <div class="p-status" data-status="${a.id}">—</div>
        </button>`).join("");
      box.querySelectorAll(".profile").forEach((b) => b.addEventListener("click", () => enter(b.dataset.agent)));
    }
    if (!s) return;
    $("#entry-sub").textContent = s.orchestrator.headline;
    $("#entry-clock").textContent = `${s.city} · ${s.time.replace("T", " ")}`;
    document.title = `CityOS — ${s.city}`;
    $("#entry-engine").textContent = s.running ? "Closed loop running" : "Paused";
    $("#entry-llm").textContent = s.orchestrator.llm.enabled ? `LLM: ${s.orchestrator.llm.model}` : "Deterministic reasoner";
    for (const a of AGENTS) {
      const el = box.querySelector(`[data-status="${a.id}"]`);
      if (a.id === "orchestrator") {
        const h = s.health ? Math.round(s.health.score) : "—";
        el.innerHTML = `<span class="dot ${h >= 80 ? "NORMAL" : h >= 65 ? "WATCH" : "WARNING"}"></span>Health ${h}`;
      } else {
        const g = s.agents[a.id];
        if (g) el.innerHTML = `<span class="dot ${esc(g.level)}"></span>${esc(g.label)} · ${Math.round(g.health)}`;
      }
    }
  }

  function enter(agentId) {
    const entry = $("#entry");
    entry.classList.add("leaving");
    setTimeout(() => {
      entry.hidden = true;
      entry.classList.remove("leaving");
      $("#workspace").hidden = false;
      switchAgent(agentId);
    }, 210);
  }

  function toEntry() {
    $("#workspace").hidden = true;
    $("#entry").hidden = false;
    S.agent = null;
    renderEntry();
  }

  function switchAgent(id) {
    S.agent = id;
    S.tab = 0;
    try { localStorage.setItem("cityos.agent", id); } catch (e) { /* storage unavailable */ }
    const h = $("#headline");
    h.style.opacity = 0;
    setTimeout(() => (h.style.opacity = 1), 160);
    renderSwitcher();
    renderTabs();
    renderAll();
  }

  // ------------------------------------------------------------------ top bar
  function renderSwitcher() {
    const s = S.snap;
    $("#switcher").innerHTML = [AG.orchestrator, ...AGENTS.filter((a) => a.id !== "orchestrator")].map((a) => {
      const lvl = a.id === "orchestrator" ? "" : `<span class="dot ${esc(s?.agents?.[a.id]?.level || "NORMAL")}"></span>`;
      const short = { orchestrator: "Orchestrator", infrastructure: "Infrastructure" }[a.id] || a.name.replace(" Agent", "");
      return `<button class="sw ${S.agent === a.id ? "on" : ""}" data-agent="${a.id}" style="--c:${col(a.id)}" title="${a.name}"><span class="ic">${ICONS[a.id]}</span><span class="lbl">${short}</span>${lvl}</button>`;
    }).join("");
    $("#switcher").querySelectorAll(".sw").forEach((b) => b.addEventListener("click", () => switchAgent(b.dataset.agent)));
  }

  function renderTop() {
    const s = S.snap;
    $("#chip-time").textContent = `${s.time.slice(0, 10)} · ${hm(s.time)}`;
    const w = s.weather;
    $("#chip-weather").innerHTML = `${fmt(w.temperature, 1)}°C · ${w.rainfall > 0 ? `${fmt(w.rainfall, 1)} mm/h` : `${Math.round(w.humidity)}% RH`}${w.alerts.length ? ` · <b style="color:var(--warn)">${esc(w.alerts.join(", "))}</b>` : ""}`;
    const h = s.health;
    $("#chip-health").innerHTML = h ? `<span class="dot ${h.score >= 80 ? "NORMAL" : h.score >= 65 ? "WATCH" : "WARNING"}"></span>Health ${Math.round(h.score)}` : "Health —";
    const n = s.approvals.length;
    $("#auth-count").textContent = n;
    $("#chip-auth").classList.toggle("has", n > 0);
    document.querySelectorAll("#speed button").forEach((b) => {
      const v = b.dataset.speed;
      b.classList.toggle("on", v === "pause" ? !s.running : s.running && Number(v) === s.speed);
      if (v === "pause") b.textContent = s.running ? "❚❚" : "▶";
    });
    document.querySelectorAll("#switcher .sw").forEach((b) => {
      const d = b.querySelector(".dot");
      if (d && s.agents[b.dataset.agent]) d.className = `dot ${s.agents[b.dataset.agent].level}`;
    });
  }

  // ------------------------------------------------------------------ hero
  function renderHero() {
    const s = S.snap, a = AG[S.agent];
    $("#hero-agent").innerHTML = `<span class="ic" style="--c:${col(a.id)}">${ICONS[a.id]}</span>${a.name} · ${a.sub}`;
    let head, meta = [], kpis = [];
    if (a.id === "orchestrator") {
      head = s.orchestrator.headline;
      const llm = s.orchestrator.llm;
      meta.push(`<span class="pill">${llm.enabled ? `Reasoning: ${esc(llm.model)}` : "Deterministic reasoner"}</span>`);
      meta.push(`<span>${Object.keys(s.state).length} domain agents · ${s.issues.length} open issues · cycle ${fmt(s.cycle_ms)} ms</span>`);
      if (s.orchestrator.llm_summary) meta.push(`<span>${esc(s.orchestrator.llm_summary)}</span>`);
      const ver = s.actions.filter((x) => x.verification && ["SUCCESS", "PARTIAL", "FAILED"].includes(x.verification.result));
      const ok = ver.filter((x) => x.verification.result === "SUCCESS").length;
      kpis = [["City Health", s.health ? fmt(s.health.score) : "—", "/100"], ["Critical issues", s.issues.filter((i) => i.category === "CRITICAL").length, ""],
        ["Actions active", s.actions.filter((x) => x.status === "ACTIVE").length, ""], ["Awaiting approval", s.approvals.length, ""],
        ["Verified success", ver.length ? Math.round((ok / ver.length) * 100) : "—", ver.length ? "%" : ""]];
    } else {
      const g = s.agents[a.id];
      head = g.headline;
      meta.push(`<span class="pill ${esc(g.level)}"><span class="dot ${esc(g.level)}"></span>${esc(g.label)}</span>`, `<span>Domain health ${Math.round(g.health)}/100</span>`);
      kpis = KPI_DEF[a.id].map(([k, l, u]) => [l, fmt(g.kpis[k], typeof g.kpis[k] === "number" && g.kpis[k] % 1 && Math.abs(g.kpis[k]) < 10 ? 2 : 0), u]);
    }
    $("#headline").textContent = head;
    $("#hero-meta").innerHTML = meta.join("");
    $("#kpis").innerHTML = kpis.map(([l, v, u]) => `<div class="kpi glass"><div class="v">${v}<small>${u}</small></div><div class="l">${l}</div></div>`).join("");
  }

  function renderTabs() {
    const a = AG[S.agent];
    $("#tabs").innerHTML = a.tabs.map((t, i) => `<button class="tab ${i === S.tab ? "on" : ""}" role="tab" data-i="${i}">${t}</button>`).join("");
    $("#tabs").querySelectorAll(".tab").forEach((b) => b.addEventListener("click", () => {
      S.tab = Number(b.dataset.i);
      renderTabs();
      renderPanel();
    }));
  }

  // ------------------------------------------------------------------ map (SVG digital twin)
  function buildMap() {
    const g = S.geo;
    if (!g || S.mapBuilt) return;
    const svg = $("#map");
    svg.setAttribute("viewBox", g.viewbox.join(" "));
    svg.innerHTML = "";
    const L = {};
    ["zones", "river", "roads", "corridors", "nodes", "buildings", "eassets", "assets", "bins", "trucks", "units", "incidents", "issues"].forEach((k) => (L[k] = svgEl("g", { id: `L-${k}` }, svg)));
    for (const z of g.zones) {
      const [x1, y1, x2, y2] = z.bbox;
      svgEl("rect", { x: x1 + 6, y: y1 + 6, width: x2 - x1 - 12, height: y2 - y1 - 12, rx: 18, fill: "var(--zone)", stroke: "var(--line)", id: `z-${z.id}`, class: "hit", "data-tip": `zone:${z.id}` }, L.zones);
      const t = svgEl("text", { x: x1 + 18, y: y1 + 24, class: "zone-label" }, L.zones);
      t.textContent = z.name;
    }
    svgEl("rect", { x: 0, y: g.river_y - 16, width: g.viewbox[2], height: 32, fill: "var(--river)", opacity: 0.9 }, L.river);
    for (const r of g.roads) {
      svgEl("line", { x1: r.x1, y1: r.y1, x2: r.x2, y2: r.y2, stroke: "var(--road)", "stroke-width": r.lanes >= 4 ? 9 : 6, class: "road hit", id: `r-${r.id}`, "data-tip": `road:${r.id}` }, L.roads);
    }
    for (const n of g.nodes) svgEl("circle", { cx: n.x, cy: n.y, r: 5, fill: "var(--glass-strong)", stroke: "var(--faint)", "stroke-width": 1.5, id: `n-${n.id}`, class: "hit", "data-tip": `node:${n.id}` }, L.nodes);
    for (const b of g.buildings) svgEl("rect", { x: b.x - 9, y: b.y - 9, width: 18, height: 18, rx: 5, fill: "var(--glass-strong)", stroke: "var(--faint)", id: `b-${b.id}`, class: "hit", "data-tip": `bld:${b.id}` }, L.buildings);
    for (const a of g.energy_assets) {
      const shape = a.kind === "solar" ? svgEl("rect", { x: a.x - 8, y: a.y - 6, width: 16, height: 12, rx: 2 }, L.eassets) : svgEl("circle", { cx: a.x, cy: a.y, r: 7 }, L.eassets);
      Object.entries({ fill: "var(--glass-strong)", stroke: "var(--c-energy)", "stroke-width": 1.5, id: `e-${a.id}`, class: "hit", "data-tip": `eas:${a.id}` }).forEach(([k, v]) => shape.setAttribute(k, v));
    }
    for (const a of g.assets) svgEl("path", { d: `M${a.x} ${a.y - 8} L${a.x + 8} ${a.y} L${a.x} ${a.y + 8} L${a.x - 8} ${a.y} Z`, fill: "var(--glass-strong)", stroke: "var(--c-infrastructure)", "stroke-width": 1.5, id: `a-${a.id}`, class: "hit", "data-tip": `asset:${a.id}` }, L.assets);
    for (const b of g.bins) svgEl("circle", { cx: b.x, cy: b.y, r: 6, fill: "var(--ok)", stroke: "var(--glass-strong)", "stroke-width": 1.5, id: `bin-${b.id}`, class: "hit", "data-tip": `bin:${b.id}` }, L.bins);
    S.L = L;
    S.mapBuilt = true;
    const tip = $("#map-tip");
    svg.addEventListener("mousemove", (ev) => {
      const t = ev.target.closest("[data-tip]");
      if (!t) { tip.hidden = true; return; }
      tip.innerHTML = tipText(t.dataset.tip);
      tip.hidden = false;
      const r = $(".map-wrap").getBoundingClientRect();
      tip.style.left = `${Math.min(ev.clientX - r.left + 14, r.width - 290)}px`;
      tip.style.top = `${ev.clientY - r.top + 14}px`;
    });
    svg.addEventListener("mouseleave", () => (tip.hidden = true));
    svg.addEventListener("click", (ev) => {
      const t = ev.target.closest("[data-issue]");
      if (t) openIssue(t.dataset.issue);
    });
  }

  function tipText(key) {
    const [kind, id] = key.split(/:(.+)/);
    const lv = S.snap.live, g = S.geo;
    if (kind === "road") {
      const r = lv.roads[id], meta = g.roads.find((x) => x.id === id);
      return `<b>${esc(meta.name)}</b>${r.closed ? `Closed — ${esc(r.reason)}` : `v/c ${fmt(r.vc, 2)} · ${fmt(r.flow)} veh/h · ${fmt(r.speed)} km/h<br>queue ${fmt(r.queue_m)} m${r.reason ? ` · ${esc(r.reason)}` : ""}${r.rerouted ? " · AI reroute active" : ""}`}`;
    }
    if (kind === "node") { const n = lv.nodes[id]; return `<b>Intersection ${id}</b>${esc(n.plan)} · NS green ${pct(n.g_ns)} · cycle ${n.cycle}s<br>delay NS ${n.d_ns}s · EW ${n.d_ew}s${n.preempt ? "<br><b>Emergency pre-emption</b>" : ""}`; }
    if (kind === "bin") { const b = lv.bins[id]; return `<b>Bin ${id}</b>${b.type} · fill ${pct(b.fill)} · ${b.weight} kg · ${b.temp}°C`; }
    if (kind === "bld") { const b = lv.buildings[id], m = g.buildings.find((x) => x.id === id); return `<b>${esc(m.name)}</b>${fmt(b.power)} kW · HVAC ${fmt(b.hvac)} · lighting ${fmt(b.lighting)} · PF ${b.pf}${b.hvac_offset ? ` · set-point +${b.hvac_offset}°C` : ""}`; }
    if (kind === "eas") { const a = lv.energy_assets[id], m = g.energy_assets.find((x) => x.id === id); return `<b>${esc(m.name)}</b>${fmt(a.power)} kW${m.kind === "battery" ? ` · SoC ${pct(a.soc)}` : ""}${a.shift ? ` · ${pct(a.shift)} shifted` : ""}${a.curtail ? ` · curtailed ${pct(a.curtail)}` : ""}`; }
    if (kind === "asset") { const a = lv.assets[id], m = g.assets.find((x) => x.id === id); return `<b>${esc(m.name)}</b>${a.status} · condition ${a.condition}${m.kind === "water_main" ? ` · ${a.pressure} bar` : m.kind === "streetlight" ? ` · lamps ${pct(a.lamps_ok)}` : ` · vib ${a.vibration} mm/s`}${a.work_order ? ` · ${a.work_order}` : ""}`; }
    if (kind === "zone") { const z = lv.zones[id], m = g.zones.find((x) => x.id === id); return `<b>${esc(m.name)}</b>crowd ${z.density} p/m²${z.event ? ` · ${z.event}` : ""}${z.dark ? " · lighting outage" : ""}${z.managed ? " · crowd management active" : ""}`; }
    if (kind === "unit") { const u = lv.units[id]; return `<b>${id}</b>${u.kind} · ${u.status}${u.incident ? ` · ${esc(u.incident)}` : ""}`; }
    if (kind === "truck") { const t = lv.trucks[id]; return `<b>${id}</b>${t.status}${t.stops.length ? ` · ${t.stops.length} stops left` : ""}${t.source ? ` · ${esc(t.source)}` : ""}`; }
    if (kind === "inc") { const i = lv.incidents.find((x) => x.id === id); return i ? `<b>${esc(i.description)}</b>severity ${i.severity}/4 · ${i.status}${i.unit ? ` · ${i.unit}` : ""} · ${i.age_min} min` : id; }
    if (kind === "issue") { const i = S.snap.issues.find((x) => x.key === id); return i ? `<b>${esc(i.title)}</b>${i.category} · priority ${fmt(i.priority)}<br>Click for details` : id; }
    return id;
  }

  function renderMap() {
    buildMap();
    if (!S.mapBuilt) return;
    const s = S.snap, lv = s.live, ag = S.agent, L = S.L, g = S.geo;
    const focus = (d) => ag === "orchestrator" || ag === d;
    // roads
    for (const r of g.roads) {
      const e = $(`#r-${r.id}`), st = lv.roads[r.id];
      const colored = ag === "traffic" || ag === "orchestrator";
      e.setAttribute("stroke", colored ? vcColor(st.vc, st.closed) : "var(--road)");
      e.setAttribute("stroke-dasharray", st.closed ? "6 7" : st.rerouted && colored ? "14 5" : "");
      e.setAttribute("opacity", colored ? 0.95 : 0.55);
    }
    for (const n of g.nodes) {
      const e = $(`#n-${n.id}`), st = lv.nodes[n.id];
      e.setAttribute("stroke", st.preempt ? "var(--crit)" : st.plan === "AI-adaptive" && focus("traffic") ? "var(--c-traffic)" : "var(--faint)");
      e.setAttribute("stroke-width", st.plan === "AI-adaptive" && focus("traffic") ? 3 : 1.5);
      e.setAttribute("opacity", focus("traffic") ? 1 : 0.5);
    }
    // zones (crowd levels)
    const zl = s.agents.safety?.extra?.zones || {};
    for (const z of g.zones) {
      const e = $(`#z-${z.id}`), lvl = zl[z.id]?.level || "NORMAL";
      const tint = focus("safety") && lvl !== "NORMAL" ? `color-mix(in srgb, ${lvlColor[lvl]} 16%, transparent)` : "var(--zone)";
      e.setAttribute("fill", tint);
      e.setAttribute("stroke", lv.zones[z.id].dark && focus("safety") ? "var(--crit)" : "var(--line)");
      e.setAttribute("stroke-dasharray", lv.zones[z.id].dark && focus("safety") ? "4 4" : "");
    }
    // buildings
    for (const b of g.buildings) {
      const e = $(`#b-${b.id}`), st = lv.buildings[b.id];
      const sz = 8 + Math.min(14, st.power / 70);
      e.setAttribute("x", b.x - sz / 2); e.setAttribute("y", b.y - sz / 2); e.setAttribute("width", sz); e.setAttribute("height", sz);
      const flagged = s.issues.some((i) => i.entity === b.id && i.domain === "energy");
      e.setAttribute("fill", focus("energy") ? (flagged ? "color-mix(in srgb, var(--warn) 35%, var(--glass-strong))" : "color-mix(in srgb, var(--c-energy) 18%, var(--glass-strong))") : "var(--glass-strong)");
      e.setAttribute("stroke", focus("energy") ? (flagged ? "var(--warn)" : "var(--c-energy)") : "var(--faint)");
      e.setAttribute("opacity", focus("energy") ? 1 : 0.45);
    }
    L.eassets.setAttribute("opacity", focus("energy") ? 1 : 0.3);
    // assets
    for (const a of g.assets) {
      const e = $(`#a-${a.id}`), st = lv.assets[a.id];
      const c = st.status === "FAILED" ? "var(--crit)" : st.status === "UNDER_REPAIR" ? "var(--c-traffic)" : st.status === "DEGRADED" ? "var(--warn)" : "var(--c-infrastructure)";
      e.setAttribute("stroke", c);
      e.setAttribute("fill", st.status === "OPERATIONAL" ? "var(--glass-strong)" : `color-mix(in srgb, ${c} 30%, var(--glass-strong))`);
      e.setAttribute("opacity", focus("infrastructure") ? 1 : 0.35);
    }
    // bins
    for (const b of g.bins) {
      const e = $(`#bin-${b.id}`), f = lv.bins[b.id].fill;
      e.setAttribute("fill", f >= 1 ? "var(--crit)" : f >= 0.8 ? "var(--warn)" : f >= 0.6 ? "var(--watch)" : "var(--ok)");
      e.setAttribute("r", focus("waste") ? 6 + Math.min(1.2, f) * 2 : 4);
      e.setAttribute("opacity", focus("waste") ? 1 : 0.35);
    }
    // dynamic layers
    L.trucks.innerHTML = "";
    if (focus("waste")) {
      for (const [id, t] of Object.entries(lv.trucks)) {
        if (t.stops.length && ag === "waste") {
          const pts = [[t.x, t.y], ...t.stops.map((b) => { const m = g.bins.find((x) => x.id === b); return [m.x, m.y]; })];
          svgEl("polyline", { points: pts.map((p) => p.join(",")).join(" "), fill: "none", stroke: "var(--c-waste)", "stroke-width": 2, "stroke-dasharray": "3 5", opacity: 0.8 }, L.trucks);
        }
        const gr = svgEl("g", { class: "hit", "data-tip": `truck:${id}` }, L.trucks);
        svgEl("rect", { x: t.x - 11, y: t.y - 22, width: 22, height: 14, rx: 4, fill: t.status === "idle" ? "var(--glass-strong)" : "var(--c-waste)", stroke: "var(--c-waste)" }, gr);
        const tx = svgEl("text", { x: t.x, y: t.y - 12, "text-anchor": "middle", "font-size": 8.5, "font-weight": 700, fill: t.status === "idle" ? "var(--c-waste)" : "#fff" }, gr);
        tx.textContent = id.replace("WT-", "T");
      }
    }
    L.units.innerHTML = "";
    L.corridors.innerHTML = "";
    if (focus("safety") || focus("infrastructure")) {
      for (const [id, u] of Object.entries(lv.units)) {
        if (u.kind === "crew" && !focus("infrastructure")) continue;
        if (u.kind !== "crew" && !focus("safety")) continue;
        if (u.path?.length > 1) {
          const pts = u.path.map((n) => { const m = g.nodes.find((x) => x.id === n); return `${m.x},${m.y}`; }).join(" ");
          svgEl("polyline", { points: pts, fill: "none", stroke: u.kind === "crew" ? "var(--c-infrastructure)" : "var(--c-safety)", "stroke-width": 3, opacity: 0.55 }, L.corridors);
        }
        const gr = svgEl("g", { class: "hit", "data-tip": `unit:${id}` }, L.units);
        const c = u.kind === "crew" ? "var(--c-infrastructure)" : "var(--c-safety)";
        svgEl("circle", { cx: u.x + 12, cy: u.y + 12, r: 8.5, fill: u.status === "available" ? "var(--glass-strong)" : c, stroke: c, "stroke-width": 1.5 }, gr);
        const tx = svgEl("text", { x: u.x + 12, y: u.y + 15.2, "text-anchor": "middle", "font-size": 8.5, "font-weight": 700, fill: u.status === "available" ? c : "#fff" }, gr);
        tx.textContent = { ambulance: "A", fire: "F", police: "P", crew: "M" }[u.kind];
      }
    }
    for (const c of lv.corridors) {
      const pts = c.map((n) => { const m = g.nodes.find((x) => x.id === n); return `${m.x},${m.y}`; }).join(" ");
      svgEl("polyline", { points: pts, fill: "none", stroke: "var(--crit)", "stroke-width": 5, opacity: 0.35, "stroke-linecap": "round" }, L.corridors);
    }
    L.incidents.innerHTML = "";
    if (focus("safety") || focus("traffic")) {
      for (const i of lv.incidents) {
        const gr = svgEl("g", { class: "hit", "data-tip": `inc:${i.id}` }, L.incidents);
        svgEl("circle", { cx: i.x, cy: i.y, r: 9, fill: "var(--crit)", opacity: 0.9 }, gr);
        const tx = svgEl("text", { x: i.x, y: i.y + 3.5, "text-anchor": "middle", "font-size": 10, "font-weight": 800, fill: "#fff" }, gr);
        tx.textContent = "!";
      }
    }
    // issue markers (priority rings) — click to open
    L.issues.innerHTML = "";
    const iss = s.issues.filter((i) => i.x != null && (ag === "orchestrator" ? ["HIGH", "CRITICAL"].includes(i.category) : i.domain === ag) && i.kind !== "collection_plan").slice(0, 25);
    for (const i of iss) {
      const r = { LOW: 13, MEDIUM: 15, HIGH: 17, CRITICAL: 19 }[i.category];
      const gr = svgEl("g", { class: "hit", "data-tip": `issue:${i.key}`, "data-issue": i.key }, L.issues);
      svgEl("circle", { cx: i.x, cy: i.y, r: r + 6, fill: "transparent" }, gr);
      svgEl("circle", { cx: i.x, cy: i.y, r, class: "issue-ring", stroke: catColor[i.category], opacity: 0.9 }, gr);
    }
    // legend
    const leg = {
      traffic: '<span><i style="background:#3fbf6a"></i>free</span><span><i style="background:#f59e0b"></i>busy</span><span><i style="background:#ff5a4f"></i>congested</span><span><i class="c" style="border:3px solid var(--c-traffic)"></i>adaptive signal</span>',
      safety: '<span><i class="c" style="background:var(--watch)"></i>watch</span><span><i class="c" style="background:var(--warn)"></i>warning</span><span><i class="c" style="background:var(--crit)"></i>incident</span><span>A/F/P units</span>',
      waste: '<span><i class="c" style="background:var(--ok)"></i>&lt;60%</span><span><i class="c" style="background:var(--watch)"></i>60–80%</span><span><i class="c" style="background:var(--warn)"></i>&gt;80%</span><span><i class="c" style="background:var(--crit)"></i>overflow</span>',
      energy: '<span><i style="background:var(--c-energy)"></i>building load (size = kW)</span><span><i style="background:var(--warn)"></i>anomaly</span><span>▭ solar · ○ battery/EV</span>',
      infrastructure: '<span><i class="c" style="background:var(--c-infrastructure)"></i>operational</span><span><i class="c" style="background:var(--warn)"></i>degraded</span><span><i class="c" style="background:var(--crit)"></i>failed</span><span>M crews</span>',
      orchestrator: '<span><i style="background:#ff5a4f"></i>congestion</span><span><i class="c" style="border:2px solid var(--warn)"></i>high</span><span><i class="c" style="border:2px solid var(--crit)"></i>critical issue</span>',
    };
    $("#legend").innerHTML = leg[ag];
    $("#map-title").textContent = ag === "orchestrator" ? "Digital twin — all domains" : `Digital twin — ${DOMAIN_LABEL[ag]}`;
  }

  // ------------------------------------------------------------------ side column
  function issueItem(i) {
    const d = i.domain === "sensor" ? "sensor" : i.domain;
    const acted = i.actions?.length ? `<span class="tag" style="--c:var(--c-waste)">action ${esc(i.actions[i.actions.length - 1])}</span>` : "";
    return `<button class="item" data-issue="${esc(i.key)}" style="--c:${col(d)}">
      <div class="t"><span class="dot ${i.category}"></span><span class="grow">${esc(i.title)}</span><span class="pill ${i.category}">${i.category}</span></div>
      <div class="d">${esc(i.detail)}</div>
      <div class="meta"><span class="tag">${DOMAIN_LABEL[d] || d}</span><span>priority ${fmt(i.priority)}</span><span>since ${hm(i.first_seen)}</span>${i.plan ? `<span class="tag" style="--c:var(--c-orchestrator)">${esc(i.plan)}</span>` : ""}${acted}</div>
    </button>`;
  }

  function actionItem(a) {
    const v = a.verification;
    const res = v ? v.result : a.status;
    const imp = v?.improvement_vs_counterfactual_pct ?? v?.improvement_pct;
    return `<button class="item" data-action="${esc(a.id)}" style="--c:${col(a.domain)}">
      <div class="t"><span class="grow">${esc(a.title)}</span><span class="pill ${esc(res)}">${esc(res.replace("_", " "))}</span></div>
      <div class="d">${esc(a.decision?.reasoner || "")}${a.result?.summary ? ` · ${esc(a.result.summary)}` : ""}</div>
      <div class="meta"><span class="tag">${DOMAIN_LABEL[a.domain] || a.domain}</span><span>${hm(a.created)}</span><span>risk ${a.risk}</span>${imp != null ? `<span>${imp >= 0 ? "▲" : "▼"} ${fmt(Math.abs(imp), 1)}% ${v.improvement_vs_counterfactual_pct != null ? "vs no-action twin" : "vs before"}</span>` : ""}<span>SIMULATED</span></div>
    </button>`;
  }

  function renderSide() {
    const s = S.snap, ag = S.agent;
    const iss = s.issues.filter((i) => (ag === "orchestrator" ? i.category !== "LOW" || i.actions?.length : i.domain === ag)).slice(0, 30);
    $("#attn-count").textContent = `${iss.length} open`;
    $("#attention").innerHTML = iss.length ? iss.map(issueItem).join("") : '<div class="empty">Nothing needs attention right now.</div>';
    const acts = s.actions.filter((a) => ag === "orchestrator" || a.domain === ag).slice(0, 25);
    $("#actions").innerHTML = acts.length ? acts.map(actionItem).join("") : '<div class="empty">No actions yet — the agent acts when an issue warrants it.</div>';
    const f = S.filter;
    const rows = s.activity.filter((e) => (ag === "orchestrator" || e.domain === ag || e.domain === "city") && (f === "all" || e.phase.toLowerCase() === f)).slice(-70).reverse();
    $("#activity").innerHTML = rows.length ? rows.map((e) => `<div class="log-row"><span class="ts">${hm(e.ts)}</span><span class="phase ${e.phase}">${e.phase}<br><span class="who">${esc(e.agent.replace(" Agent", ""))}</span></span><span>${esc(e.message)}</span></div>`).join("") : '<div class="empty">No activity yet.</div>';
    bindItems($("#attention")); bindItems($("#actions"));
  }

  function bindItems(root) {
    root.querySelectorAll("[data-issue]").forEach((b) => (b.onclick = () => openIssue(b.dataset.issue)));
    root.querySelectorAll("[data-action]").forEach((b) => (b.onclick = () => openAction(b.dataset.action)));
  }

  // ------------------------------------------------------------------ charts (canvas)
  function chart(canvas, { labels, series, threshold, yLabel = "", min = null }) {
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth || 600, h = canvas.clientHeight || 220;
    canvas.width = w * dpr; canvas.height = h * dpr;
    const c = canvas.getContext("2d");
    c.scale(dpr, dpr);
    const cs = getComputedStyle(document.documentElement);
    const muted = cs.getPropertyValue("--muted").trim(), line = cs.getPropertyValue("--line").trim();
    const P = { l: 46, r: 12, t: 12, b: 26 };
    const all = series.flatMap((s) => s.data).concat(threshold ? [threshold.value] : []).filter((v) => v != null);
    let lo = min != null ? Math.min(min, ...all) : Math.min(0, ...all), hi = Math.max(...all) * 1.08 || 1;
    if (min != null) hi = Math.min(hi, Math.max(...all) + 5);
    const X = (i) => P.l + (i / Math.max(1, labels.length - 1)) * (w - P.l - P.r);
    const Y = (v) => P.t + (1 - (v - lo) / (hi - lo)) * (h - P.t - P.b);
    c.font = "11px " + cs.getPropertyValue("--font"); c.fillStyle = muted; c.strokeStyle = line; c.lineWidth = 1;
    for (let k = 0; k <= 4; k++) {
      const v = lo + ((hi - lo) * k) / 4, y = Y(v);
      c.beginPath(); c.moveTo(P.l, y); c.lineTo(w - P.r, y); c.stroke();
      c.fillText(v >= 1000 ? `${(v / 1000).toFixed(1)}k` : v.toFixed(v < 10 ? 1 : 0), 6, y + 4);
    }
    const step = Math.ceil(labels.length / 8);
    labels.forEach((l, i) => { if (i % step === 0) c.fillText(l, X(i) - 14, h - 8); });
    if (yLabel) c.fillText(yLabel, P.l, P.t - 2 + 10);
    if (threshold) {
      c.save(); c.setLineDash([5, 5]); c.strokeStyle = threshold.color; c.beginPath(); c.moveTo(P.l, Y(threshold.value)); c.lineTo(w - P.r, Y(threshold.value)); c.stroke(); c.restore();
      c.fillStyle = threshold.color; c.fillText(threshold.label, w - P.r - c.measureText(threshold.label).width, Y(threshold.value) - 5);
    }
    for (const s of series) {
      c.strokeStyle = s.color; c.lineWidth = 2; c.beginPath();
      s.data.forEach((v, i) => (i ? c.lineTo(X(i), Y(v)) : c.moveTo(X(i), Y(v))));
      c.stroke();
      if (s.fill) {
        c.lineTo(X(s.data.length - 1), Y(lo)); c.lineTo(X(0), Y(lo)); c.closePath();
        c.globalAlpha = 0.12; c.fillStyle = s.color; c.fill(); c.globalAlpha = 1;
      }
    }
    let lx = P.l + 6;
    for (const s of series) { c.fillStyle = s.color; c.fillRect(lx, P.t + 2, 10, 3); c.fillStyle = muted; c.fillText(s.name, lx + 14, P.t + 7); lx += c.measureText(s.name).width + 32; }
  }
  const colorOf = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();

  // ------------------------------------------------------------------ panels
  const bar = (v, max, c = "var(--c-traffic)") => `<div class="bar" style="--c:${c}"><i style="width:${Math.max(0, Math.min(100, (v / max) * 100))}%"></i></div>`;
  const actionsTable = (domain) => {
    const acts = S.snap.actions.filter((a) => a.domain === domain);
    return acts.length ? `<div class="list" style="max-height:none">${acts.slice(0, 20).map(actionItem).join("")}</div>` : '<div class="empty">No actions in this domain yet.</div>';
  };

  function renderPanel() {
    const s = S.snap, ag = S.agent, tab = AG[ag].tabs[S.tab];
    const P = $("#panel");
    let html = "";
    const r = s.agents[ag];
    const lv = s.live;
    if (ag === "traffic") {
      if (tab === "Live Traffic") {
        const roads = S.geo.roads.map((m) => ({ ...m, ...lv.roads[m.id] })).sort((a, b) => b.vc - a.vc);
        html = `<div class="card-head"><h3>Road network — live</h3><span class="muted small">${roads.filter((x) => x.vc > 0.85).length} congested · ${r.kpis.ai_signal_plans} intersections on AI adaptive timing</span></div>
        <table class="t"><tr><th>Road</th><th>v/c</th><th></th><th class="num">Flow</th><th class="num">Speed</th><th class="num">Queue</th><th>Status</th></tr>
        ${roads.slice(0, 14).map((x) => `<tr><td>${esc(x.name)}</td><td>${fmt(x.vc, 2)}</td><td style="width:22%">${bar(x.vc, 1.3, vcColor(x.vc, x.closed))}</td><td class="num">${fmt(x.flow)}</td><td class="num">${fmt(x.speed)} km/h</td><td class="num">${fmt(x.queue_m)} m</td><td>${x.closed ? '<span class="pill CRITICAL">closed</span>' : x.reason ? `<span class="pill HIGH">${esc(x.reason)}</span>` : x.rerouted ? '<span class="pill MEDIUM">rerouted</span>' : ""}</td></tr>`).join("")}</table>`;
      } else if (tab === "Predictions") {
        const fc = S.fc;
        html = `<div class="card-head"><h3>Congestion forecast (ML)</h3><span class="muted small">${fc?.model ? `validation MAE ${fmt(fc.model.mae)} veh/h · MAPE ${fmt(fc.model.mape, 1)}%` : ""}${fc?.live_error ? ` · live MAPE ${fmt(fc.live_error.mape, 1)}%` : ""}</span></div>`;
        if (!fc) html += '<div class="empty">Loading forecast…</div>';
        else {
          const rows = Object.entries(fc.roads).map(([id, x]) => ({ id, ...x })).sort((a, b) => b.horizons["+60min"].vc - a.horizons["+60min"].vc).slice(0, 12);
          html += `<table class="t"><tr><th>Road</th><th class="num">now</th><th class="num">+15</th><th class="num">+30</th><th class="num">+60</th><th class="num">+120 min</th><th class="num">P(congestion, 60 min)</th></tr>
          ${rows.map((x) => `<tr><td>${esc(x.name)}</td><td class="num">${fmt(x.current_vc, 2)}</td>${["+15min", "+30min", "+60min", "+120min"].map((k) => `<td class="num" style="color:${vcColor(x.horizons[k].vc)}">${fmt(x.horizons[k].vc, 2)}</td>`).join("")}<td class="num">${pct(x.horizons["+60min"].p_congestion)}</td></tr>`).join("")}</table>
          <p class="muted small">Gradient-boosted flow model with weather, event and capacity features; probability from validation residuals. Values are v/c (volume / capacity).</p>`;
        }
      } else if (tab === "Incidents") {
        const inc = lv.incidents.filter((i) => i.type === "traffic_collision");
        const closed = S.geo.roads.filter((m) => lv.roads[m.id].closed);
        html = `<div class="card-head"><h3>Incidents &amp; closures</h3></div>
        ${inc.length || closed.length ? `<table class="t"><tr><th>Event</th><th>Severity</th><th>Status</th><th>Unit</th><th class="num">Age</th></tr>
        ${inc.map((i) => `<tr><td>${esc(i.description)}</td><td>${i.severity}/4</td><td><span class="pill ${i.status === "OPEN" ? "HIGH" : "MEDIUM"}">${i.status}</span></td><td>${i.unit || "—"}</td><td class="num">${i.age_min} min</td></tr>`).join("")}
        ${closed.map((m) => `<tr><td>${esc(m.name)}</td><td>closure</td><td><span class="pill CRITICAL">closed</span></td><td>—</td><td class="num">${esc(lv.roads[m.id].reason || "")}</td></tr>`).join("")}</table>` : '<div class="empty">No traffic incidents or closures.</div>'}
        <h4>Coordinated plans</h4>${plansHtml(s.plans.slice(0, 4))}`;
      } else html = `<div class="card-head"><h3>Traffic actions</h3></div>${actionsTable("traffic")}`;
    }
    if (ag === "safety") {
      const zones = r.extra.zones;
      if (tab === "Zones") {
        html = `<div class="card-head"><h3>Zones — crowd &amp; risk</h3><span class="muted small">Classification NORMAL → WATCH → WARNING → CRITICAL</span></div><div class="panel-grid">
        ${S.geo.zones.map((z) => { const x = zones[z.id]; return `<div class="tile"><div class="tt"><span>${esc(z.name)}</span><span class="pill ${x.level}">${x.level}</span></div><div class="big">${fmt(x.density, 2)} <small class="muted" style="font-size:12px">p/m²</small></div>
        <div class="muted small">normal ${fmt(x.expected, 2)} · crowding risk (1 h) ${pct(x.crowding_risk_1h)}${x.event ? ` · ${esc(x.event)}` : ""}${x.dark ? " · lighting outage" : ""}</div></div>`; }).join("")}</div>`;
      } else if (tab === "Incidents") {
        const inc = lv.incidents;
        html = `<div class="card-head"><h3>Open incidents</h3><span class="muted small">Priority = Severity × Probability × Population impact × Response time</span></div>
        ${inc.length ? `<table class="t"><tr><th>Incident</th><th>Zone</th><th>Severity</th><th>Status</th><th>Unit</th><th class="num">Age</th></tr>${inc.map((i) => `<tr class="click" data-issue="safety:incident:${i.id}"><td>${esc(i.description)}</td><td>${esc(S.geo.zones.find((z) => z.id === i.zone)?.name)}</td><td>${i.severity}/4</td><td><span class="pill ${i.status === "OPEN" ? "HIGH" : "MEDIUM"}">${i.status}</span></td><td>${i.unit || "—"}</td><td class="num">${i.age_min} min</td></tr>`).join("")}</table>` : '<div class="empty">No open incidents.</div>'}
        <h4>Emergency units</h4><table class="t"><tr><th>Unit</th><th>Type</th><th>Status</th><th>Assignment</th></tr>${Object.entries(lv.units).filter(([, u]) => u.kind !== "crew").map(([id, u]) => `<tr><td>${id}</td><td>${u.kind}</td><td><span class="pill ${u.status === "available" ? "LOW" : "MEDIUM"}">${u.status}</span></td><td>${esc(u.incident || "—")}</td></tr>`).join("")}</table>`;
      } else if (tab === "Coverage") {
        const cov = r.extra.coverage;
        const max = Math.max(...Object.values(cov).filter((v) => v != null), 1);
        html = `<div class="card-head"><h3>Emergency coverage</h3><span class="muted small">Fastest available ambulance/fire unit to each zone (live road network)</span></div>
        ${S.geo.zones.map((z) => `<div class="contrib"><span>${esc(z.name)}</span>${bar(cov[z.id] ?? 0, max * 1.2, "var(--c-safety)")}<span class="num">${cov[z.id] == null ? "—" : fmt(cov[z.id], 1) + " min"}</span><span></span></div>`).join("")}`;
      } else html = `<div class="card-head"><h3>Safety actions</h3><span class="muted small">Dispatch & pre-emption require human authorisation</span></div>${actionsTable("safety")}`;
    }
    if (ag === "waste") {
      const fc = r.extra.forecasts;
      if (tab === "Bins") {
        html = `<div class="card-head"><h3>Smart bins</h3><span class="muted small">${r.kpis.overflowing_bins} overflowing · ${r.kpis.at_risk} at risk</span></div><div class="panel-grid">
        ${S.geo.bins.map((b) => { const x = lv.bins[b.id], f = x.fill, c = f >= 1 ? "var(--crit)" : f >= 0.8 ? "var(--warn)" : f >= 0.6 ? "var(--watch)" : "var(--ok)"; return `<div class="tile row"><div class="gauge" style="--c:${c}"><i style="height:${Math.min(100, f * 100)}%"></i></div><div><div class="tt">${b.id}</div><div class="muted small">${b.type} · ${esc(S.geo.zones.find((z) => z.id === b.zone).name)}</div><div class="small">${pct(f)} · ${x.weight} kg${b.type === "recycling" ? ` · contam. ${pct(x.contamination)}` : ""}</div><div class="muted small">overflow ${fc[b.id]?.overflow_hours === 0 ? "now" : fc[b.id]?.overflow_hours ? `~${fc[b.id].overflow_hours} h` : "> 12 h"}</div></div></div>`; }).join("")}</div>`;
      } else if (tab === "Forecasts") {
        const rows = Object.entries(fc).sort((a, b) => (a[1].overflow_hours ?? 99) - (b[1].overflow_hours ?? 99)).slice(0, 14);
        html = `<div class="card-head"><h3>Fill trajectory &amp; overflow forecast (ML)</h3></div><table class="t"><tr><th>Bin</th><th class="num">Now</th><th class="num">+2 h</th><th class="num">+4 h</th><th class="num">+6 h</th><th class="num">Overflow</th><th class="num">P(overflow ≤ 6 h)</th></tr>
        ${rows.map(([id, x]) => `<tr><td>${id}</td><td class="num">${pct(lv.bins[id].fill)}</td><td class="num">${pct(Math.min(1.2, x.traj["+2h"]))}</td><td class="num">${pct(Math.min(1.2, x.traj["+4h"]))}</td><td class="num">${pct(Math.min(1.2, x.traj["+6h"]))}</td><td class="num">${x.overflow_hours === 0 ? "now" : x.overflow_hours ? `~${x.overflow_hours} h` : "> 12 h"}</td><td class="num">${pct(x.p)}</td></tr>`).join("")}</table>`;
      } else if (tab === "Collection") {
        const plan = s.issues.find((i) => i.key === "waste:collection_plan:city");
        html = `<div class="card-head"><h3>Collection — predict → prioritise → optimise → dispatch</h3></div>
        <table class="t"><tr><th>Truck</th><th>Status</th><th>Route source</th><th class="num">Stops left</th><th class="num">km today</th></tr>${Object.entries(lv.trucks).map(([id, t]) => `<tr><td>${id}</td><td><span class="pill ${t.status === "idle" ? "LOW" : "MEDIUM"}">${t.status}</span></td><td>${esc(t.source || "—")}</td><td class="num">${t.stops.length}</td><td class="num">${fmt(t.km, 1)}</td></tr>`).join("")}</table>
        ${plan ? `<div class="callout" style="--c:var(--c-waste)"><b>${esc(plan.title)}</b><br>${esc(plan.detail)}</div>` : '<p class="muted small">No collection plan pending — trucks are dispatched when ML forecasts overflow within the horizon.</p>'}
        <p class="muted small">Traditional baseline: every bin collected once a day at 06:00 on fixed rounds.</p>`;
      } else html = `<div class="card-head"><h3>Waste actions</h3></div>${actionsTable("waste")}`;
    }
    if (ag === "energy") {
      const f = r.extra.forecast;
      if (tab === "Demand" || tab === "Forecasts") {
        html = `<div class="card-head"><h3>${tab === "Demand" ? "Demand vs solar — next 24 h" : "Forecasts & expected impact"}</h3><span class="muted small">peak threshold ${fmt(r.kpis.peak_threshold_kw)} kW</span></div><canvas class="chart" id="ch-energy"></canvas>
        <div class="row wrap" style="margin-top:10px;gap:22px"><div><div class="muted small">Expected cost (24 h)</div><b>${esc(s.currency)}${fmt(r.kpis.expected_cost_24h)}</b></div><div><div class="muted small">Expected CO₂ (24 h)</div><b>${fmt(r.kpis.expected_co2_24h_kg)} kg</b></div><div><div class="muted small">vs expected now</div><b>${r.kpis.vs_expected_pct > 0 ? "+" : ""}${fmt(r.kpis.vs_expected_pct, 1)}%</b></div>
        ${tab === "Forecasts" ? `<div><div class="muted small">Demand model</div><b>MAPE ${fmt(s.metrics?.offline_validation?.energy_demand?.mape, 1)}%</b></div><div><div class="muted small">Live error</div><b>${s.metrics?.live_forecast_errors?.energy_demand ? `MAPE ${fmt(s.metrics.live_forecast_errors.energy_demand.mape, 1)}%` : "collecting"}</b></div>` : ""}</div>`;
        P.innerHTML = html;
        chart($("#ch-energy"), { labels: f.map((x) => x.t), series: [{ name: "Demand (kW)", data: f.map((x) => x.demand), color: colorOf("--c-energy"), fill: true }, { name: "Solar (kW)", data: f.map((x) => x.solar), color: colorOf("--c-waste") }, { name: "Net (kW)", data: f.map((x) => x.demand - x.solar), color: colorOf("--c-traffic") }], threshold: { value: r.kpis.peak_threshold_kw, label: "peak threshold", color: colorOf("--crit") } });
        return;
      }
      if (tab === "Buildings") {
        const bl = S.geo.buildings.map((m) => ({ ...m, ...lv.buildings[m.id] })).sort((a, b) => b.power - a.power);
        const max = Math.max(...bl.map((b) => b.power));
        html = `<div class="card-head"><h3>Buildings — live load</h3></div><table class="t"><tr><th>Building</th><th class="num">kW</th><th></th><th class="num">HVAC</th><th class="num">Lighting</th><th class="num">PF</th><th></th></tr>
        ${bl.map((b) => { const fl = s.issues.find((i) => i.entity === b.id && i.domain === "energy"); return `<tr ${fl ? `class="click" data-issue="${esc(fl.key)}"` : ""}><td>${esc(b.name)}</td><td class="num">${fmt(b.power)}</td><td style="width:20%">${bar(b.power, max, fl ? "var(--warn)" : "var(--c-energy)")}</td><td class="num">${fmt(b.hvac)}</td><td class="num">${fmt(b.lighting)}</td><td class="num">${b.pf}</td><td>${fl ? `<span class="pill ${fl.category}">${esc(fl.kind.replace("_", " "))}</span>` : b.hvac_offset ? `<span class="pill MEDIUM">+${b.hvac_offset}°C</span>` : ""}</td></tr>`; }).join("")}</table>`;
      } else {
        const ea = lv.energy_assets;
        html = `<div class="card-head"><h3>Optimisation</h3><span class="muted small">Minimise cost + peak + carbon within operating constraints</span></div>
        <div class="panel-grid">${S.geo.energy_assets.map((m) => `<div class="tile"><div class="tt"><span>${esc(m.name)}</span></div><div class="big">${fmt(ea[m.id].power)} <small class="muted" style="font-size:12px">kW</small></div><div class="muted small">${m.kind === "battery" ? `SoC ${pct(ea[m.id].soc)}` : m.kind === "ev_hub" ? (ea[m.id].shift ? `${pct(ea[m.id].shift)} deferred` : "unmanaged") : ea[m.id].curtail ? `curtailed ${pct(ea[m.id].curtail)}` : "full output"}</div></div>`).join("")}</div>
        <h4>Energy actions</h4>${actionsTable("energy")}`;
      }
    }
    if (ag === "infrastructure") {
      const q = r.extra.priority_queue;
      if (tab === "Assets") {
        html = `<div class="card-head"><h3>Assets</h3></div><table class="t"><tr><th>Asset</th><th>Status</th><th class="num">Condition</th><th></th><th>Signal</th></tr>
        ${S.geo.assets.map((m) => { const a = lv.assets[m.id]; return `<tr><td>${esc(m.name)}</td><td><span class="pill ${a.status === "OPERATIONAL" ? "LOW" : a.status === "FAILED" ? "CRITICAL" : "HIGH"}">${a.status.replace("_", " ")}</span></td><td class="num">${fmt(a.condition)}</td><td style="width:18%">${bar(a.condition, 100, a.condition < 40 ? "var(--warn)" : "var(--c-infrastructure)")}</td><td class="muted small">${m.kind === "water_main" ? `${a.pressure} bar · ${a.flow} L/s` : m.kind === "streetlight" ? `${pct(a.lamps_ok)} lamps` : m.kind === "pavement" ? "PCI survey" : `${a.vibration} mm/s · ${a.temperature}°C`}</td></tr>`; }).join("")}</table>`;
      } else if (tab === "Maintenance") {
        html = `<div class="card-head"><h3>Maintenance priority queue</h3><span class="muted small">Risk × Failure probability × Impact × Criticality</span></div><table class="t"><tr><th>#</th><th>Asset</th><th class="num">Score</th><th class="num">P(fail 24 h)</th><th class="num">RUL</th><th class="num">Crit.</th><th>Work order</th></tr>
        ${q.map((x, i) => `<tr><td>${i + 1}</td><td>${esc(x.name)}</td><td class="num">${fmt(x.score, 1)}</td><td class="num">${pct(x.failure_probability)}</td><td class="num">${fmt(x.rul_hours)} h</td><td class="num">${x.criticality}</td><td>${x.work_order ? `<span class="pill MEDIUM">${x.work_order}</span>` : "—"}</td></tr>`).join("")}</table>`;
      } else if (tab === "Risk") {
        const v = s.metrics?.offline_validation?.infra_failure;
        html = `<div class="card-head"><h3>Failure risk (ML, 24 h)</h3><span class="muted small">${v ? `held-out precision ${pct(v.precision)} · recall ${pct(v.recall)}` : ""}</span></div>
        ${[...q].sort((a, b) => b.failure_probability - a.failure_probability).slice(0, 12).map((x) => `<div class="contrib"><span>${esc(x.name.length > 22 ? x.name.slice(0, 21) + "…" : x.name)}</span>${bar(x.failure_probability, 1, x.failure_probability > 0.5 ? "var(--crit)" : "var(--c-infrastructure)")}<span class="num">${pct(x.failure_probability)}</span><span class="num muted">${fmt(x.rul_hours)} h</span></div>`).join("")}`;
      } else html = `<div class="card-head"><h3>Infrastructure actions</h3></div>${actionsTable("infrastructure")}`;
    }
    if (ag === "orchestrator") {
      if (tab === "Overview") html = overviewHtml();
      else if (tab === "Coordination") {
        html = `<div class="card-head"><h3>Cross-domain coordination</h3></div><h4>Coordinated plans</h4>${plansHtml(s.plans)}
        <h4>Agent messages</h4>${s.messages.length ? s.messages.slice(-14).reverse().map((m) => `<div class="log-row"><span class="ts">#${m.tick}</span><span class="phase PLAN">${esc(m.sender.replace(" Agent", ""))}<br><span class="who">→ ${esc(m.to.replace(" Agent", ""))}</span></span><span>${esc(m.text)}</span></div>`).join("") : '<div class="empty">No cross-agent messages yet. Try a road-closure scenario in What-if → Inject live.</div>'}
        <h4>City priority queue</h4><table class="t"><tr><th>Issue</th><th>Domain</th><th class="num">S×P×I×U</th><th>Category</th></tr>${s.issues.slice(0, 10).map((i) => `<tr class="click" data-issue="${esc(i.key)}"><td>${esc(i.title)}</td><td>${DOMAIN_LABEL[i.domain] || i.domain}</td><td class="num">${fmt(i.priority, 1)}</td><td><span class="pill ${i.category}">${i.category}</span></td></tr>`).join("")}</table>`;
      } else if (tab === "What-if") html = whatifHtml();
      else html = learningHtml();
    }
    P.innerHTML = html;
    bindItems(P);
    if (ag === "orchestrator" && tab === "Overview") {
      const t = s.health_trend;
      if (t.length > 1) chart($("#ch-health"), { labels: t.map((x) => x.t), series: [{ name: "City Health", data: t.map((x) => x.s), color: colorOf("--c-orchestrator"), fill: true }], min: Math.max(0, Math.min(...t.map((x) => x.s)) - 10) });
    }
    if (ag === "orchestrator" && tab === "What-if") bindWhatif();
  }

  function plansHtml(plans) {
    if (!plans.length) return '<div class="empty">No coordinated plans active.</div>';
    return plans.map((p) => `<div class="tile" style="margin-bottom:8px"><div class="tt"><span>${esc(p.id)} · ${esc(p.trigger)}</span><span class="pill ${p.status === "ACTIVE" ? "MEDIUM" : "LOW"}">${p.status}</span></div>
      ${p.steps.map((st) => `<div class="small" style="margin-top:6px"><b>${esc(st.agent)}</b> → ${esc(st.text)}</div>`).join("")}
      <div class="small" style="margin-top:6px"><b>City Orchestrator</b> → ${esc(p.actions.join(" + "))}</div></div>`).join("");
  }

  function overviewHtml() {
    const s = S.snap, h = s.health;
    if (!h) return '<div class="empty">Computing city health…</div>';
    const doms = Object.keys(h.domains);
    const sc = h.score, R = 52, C = 2 * Math.PI * R;
    const ring = `<svg class="ring" viewBox="0 0 132 132"><circle cx="66" cy="66" r="${R}" fill="none" stroke="var(--line)" stroke-width="11"/><circle cx="66" cy="66" r="${R}" fill="none" stroke="var(--c-orchestrator)" stroke-width="11" stroke-linecap="round" stroke-dasharray="${(C * sc) / 100} ${C}" transform="rotate(-90 66 66)"/><text x="66" y="70" text-anchor="middle" font-size="30" font-weight="600" fill="var(--text)">${Math.round(sc)}</text><text x="66" y="90" text-anchor="middle" font-size="10.5" fill="var(--muted)">/ 100</text></svg>`;
    return `<div class="card-head"><h3>City Health — explainable</h3><span class="muted small">weighted domain scores · Δ vs 1 h ago</span></div>
      <div class="row" style="align-items:flex-start;gap:22px">${ring}<div style="flex:1;min-width:0">
      ${doms.map((d) => { const v = h.domains[d], c = h.contributions[d], dl = h.delta_1h[d]; return `<div class="contrib"><span>${DOMAIN_LABEL[d]}</span>${bar(v, 100, v >= 80 ? "var(--ok)" : v >= 60 ? "var(--watch)" : "var(--crit)")}<span class="num">${fmt(v)}</span><span class="num ${dl > 0.5 ? "delta-up" : dl < -0.5 ? "delta-down" : "muted"}">${dl > 0 ? "+" : ""}${fmt(dl, 1)}</span></div>`; }).join("")}
      <p class="muted small">Points lost: ${doms.filter((d) => h.contributions[d] < -0.05).sort((a, b) => h.contributions[a] - h.contributions[b]).map((d) => `${DOMAIN_LABEL[d]} ${fmt(h.contributions[d], 1)}`).join(" · ") || "none"}</p></div></div>
      <h4>City state engine</h4><div class="panel-grid">${Object.entries(s.state).map(([d, x]) => `<div class="tile"><div class="tt"><span>${DOMAIN_LABEL[d]}</span><span class="dot ${x.level}"></span></div><div class="big" style="font-size:17px">${esc(x.label)}</div><div class="muted small">health ${fmt(x.health)}</div></div>`).join("")}</div>
      <h4>Health trend</h4><canvas class="chart" id="ch-health" style="height:160px"></canvas>`;
  }

  function whatifHtml() {
    const w = S.whatif;
    const opts = [["population_increase", "Population +20%"], ["festival", "City festival (CBD)"], ["road_closure", "Bridge closure"], ["extreme_weather", "Heatwave + storm"], ["ev_adoption", "EV adoption ×1.8"], ["waste_increase", "Waste +40%"], ["energy_demand_increase", "Energy demand +25%"], ["infrastructure_failure", "Transformer failure"], ["fleet_reduction", "Fleet reduction"]];
    let out = `<div class="card-head"><h3>What-if simulation</h3><span class="muted small">paired digital-twin runs · identical random seed</span></div>
      <div class="row wrap"><select class="sel" id="wi-s">${opts.map(([k, l]) => `<option value="${k}" ${w?.scenario === k ? "selected" : ""}>${l}</option>`).join("")}</select>
      <select class="sel" id="wi-h"><option value="12">12 h</option><option value="24" selected>24 h</option><option value="48">48 h</option></select>
      <button class="btn" id="wi-run" ${S.whatifBusy ? "disabled" : ""}>${S.whatifBusy ? "Simulating…" : "Run simulation"}</button>
      <button class="btn ghost" id="wi-live" title="Apply this scenario to the running city">Inject live</button><button class="btn ghost" id="wi-clear">Clear live scenario</button></div>`;
    if (w && !w.error) {
      const L = { vehicle_km: "Vehicle-km", max_vc: "Max v/c", congested_road_hours: "Congested road-hours", delay_vehicle_hours: "Delay (veh·h)", waste_generated_kg: "Waste generated (kg)", overflow_bin_hours: "Overflow bin-hours", energy_kwh: "Energy (kWh)", peak_kw: "Peak demand (kW)", energy_cost: `Energy cost (${S.snap.currency})`, max_crowd_density: "Max crowd density", new_incidents: "New incidents", asset_failures: "Asset failures", carbon_kgco2e: "Carbon (kg CO₂e)" };
      out += `<h4>${esc(w.label)} — ${w.horizon_hours} h</h4><table class="t"><tr><th>Metric</th><th class="num">Baseline</th><th class="num">Scenario</th><th class="num">Change</th></tr>
        ${Object.entries(w.comparison).map(([k, v]) => `<tr><td>${L[k] || k}</td><td class="num">${fmt(v.baseline, k.includes("vc") || k.includes("density") ? 2 : 0)}</td><td class="num">${fmt(v.scenario, k.includes("vc") || k.includes("density") ? 2 : 0)}</td><td class="num ${v.change > 0 ? "delta-down" : v.change < 0 ? "delta-up" : "muted"}">${v.change_pct != null ? `${v.change_pct > 0 ? "+" : ""}${fmt(v.change_pct, 1)}%` : fmt(v.change, 2)}</td></tr>`).join("")}</table>
        <h4>Recommended preparation</h4><ul class="small">${w.recommendations.map((r) => `<li>${esc(r)}</li>`).join("")}</ul><p class="muted small">${esc(w.method)}</p>`;
    } else if (w?.error) out += `<p class="muted">${esc(w.error)}</p>`;
    return out;
  }

  function bindWhatif() {
    $("#wi-run").onclick = async () => {
      S.whatifBusy = true; renderPanel();
      try { S.whatif = await api("/api/city/simulation", { method: "POST", body: JSON.stringify({ scenario: $("#wi-s").value, hours: Number($("#wi-h").value) }) }); }
      catch (e) { S.whatif = { error: e.message }; }
      S.whatifBusy = false; renderPanel();
    };
    $("#wi-live").onclick = async () => { await api("/api/city/simulation", { method: "POST", body: JSON.stringify({ scenario: $("#wi-s").value, mode: "live" }) }); };
    $("#wi-clear").onclick = async () => { await api("/api/city/simulation/clear", { method: "POST" }); };
  }

  function learningHtml() {
    const m = S.snap.metrics;
    if (!m) return '<div class="empty">Loading model metrics…</div>';
    const ov = m.offline_validation || {}, live = m.live_forecast_errors || {};
    const reg = ["traffic_flow", "energy_demand", "solar", "waste_fill_rate"].filter((k) => ov[k]);
    return `<div class="card-head"><h3>Learning &amp; adaptation</h3><span class="muted small">model v${m.model_version} · ${esc(ov.engine || "")} · trained ${esc((m.trained_at || "").replace("T", " "))}${m.retraining ? " · retraining…" : ""}</span></div>
      <h4>Forecast accuracy — held-out validation vs live outcomes</h4><table class="t"><tr><th>Model</th><th class="num">MAE</th><th class="num">RMSE</th><th class="num">MAPE</th><th class="num">Live MAE</th><th class="num">Live MAPE</th><th class="num">Bias corr.</th></tr>
      ${reg.map((k) => { const lk = k === "waste_fill_rate" ? "waste_fill" : k; const l = live[lk]; return `<tr><td>${k.replace(/_/g, " ")}</td><td class="num">${fmt(ov[k].mae, 3)}</td><td class="num">${fmt(ov[k].rmse, 3)}</td><td class="num">${fmt(ov[k].mape, 1)}%</td><td class="num">${l ? fmt(l.mae, 3) : "—"}</td><td class="num">${l ? fmt(l.mape, 1) + "%" : "—"}</td><td class="num">${l ? fmt(l.bias, 3) : "—"}</td></tr>`; }).join("")}</table>
      <h4>Classifiers (held-out)</h4><table class="t"><tr><th>Model</th><th class="num">Precision</th><th class="num">Recall</th><th class="num">F1</th></tr>${["infra_failure", "crowding_risk"].filter((k) => ov[k]).map((k) => `<tr><td>${k.replace("_", " ")}</td><td class="num">${pct(ov[k].precision)}</td><td class="num">${pct(ov[k].recall)}</td><td class="num">${fmt(ov[k].f1, 2)}</td></tr>`).join("")}</table>
      <h4>Live anomaly detection vs injected ground truth</h4><table class="t"><tr><th>Domain</th><th class="num">TP</th><th class="num">FP</th><th class="num">FN</th><th class="num">Precision</th><th class="num">Recall</th><th class="num">F1</th></tr>${Object.entries(m.detection).map(([k, d]) => `<tr><td>${DOMAIN_LABEL[k] || k}</td><td class="num">${d.tp}</td><td class="num">${d.fp}</td><td class="num">${d.fn}</td><td class="num">${pct(d.precision)}</td><td class="num">${pct(d.recall)}</td><td class="num">${fmt(d.f1, 2)}</td></tr>`).join("") || '<tr><td colspan="7" class="muted">collecting…</td></tr>'}</table>
      <h4>Action effectiveness (verified)</h4><table class="t"><tr><th>Action type</th><th class="num">Success</th><th class="num">Partial</th><th class="num">Failed</th></tr>${Object.entries(m.action_effectiveness).filter(([, v]) => v.total).map(([k, v]) => `<tr><td>${k.replace(/_/g, " ")}</td><td class="num">${v.success}</td><td class="num">${v.partial}</td><td class="num">${v.failed}</td></tr>`).join("") || '<tr><td colspan="4" class="muted">no verified actions yet</td></tr>'}</table>
      <h4>Adaptive thresholds</h4><div class="kv">${Object.entries(m.thresholds).map(([k, v]) => `<div>${k.replace(/_/g, " ")}</div><div>${fmt(v, 3)}</div>`).join("")}</div>
      <h4>Adaptation log</h4>${(m.adaptation || []).slice().reverse().map((a) => `<div class="small" style="padding:4px 0;border-bottom:1px solid var(--line)">${a.predicted != null ? `<b>${esc(a.model)}</b>: predicted ${fmt(a.predicted)} · actual ${fmt(a.actual)} · error ${fmt(a.error_pct, 1)}% → bias ${a.old_bias} → ${a.new_bias}` : `<b>${esc(a.model)}</b>: ${esc(a.note)}`}</div>`).join("") || '<div class="empty">No adaptations yet.</div>'}
      <h4>Data pipeline</h4><div class="kv">${Object.entries(m.processing).map(([k, v]) => `<div>${k.replace(/_/g, " ")}</div><div>${fmt(v)}</div>`).join("")}</div>`;
  }

  // ------------------------------------------------------------------ sheets
  function openSheet(html) {
    $("#sheet-body").innerHTML = html;
    $("#sheet").hidden = false; $("#scrim").hidden = false;
    bindItems($("#sheet-body"));
  }
  function closeSheet() { $("#sheet").hidden = true; $("#scrim").hidden = true; S.sheet = null; }

  function openIssue(key) {
    const i = S.snap.issues.find((x) => x.key === key);
    if (!i) return;
    S.sheet = { kind: "issue", key };
    const ex = (e) => e && e.before != null && e.after != null ? `${esc(e.metric || "")}: ${fmt(e.before, 2)} → ${fmt(e.after, 2)}${e.improvement_pct != null ? ` (${fmt(e.improvement_pct, 1)}%)` : ""}` : esc(e?.metric || e?.note || "");
    const acts = (i.actions || []).map((id) => S.snap.actions.find((a) => a.id === id)).filter(Boolean);
    openSheet(`<span class="tag" style="--c:${col(i.domain)}">${DOMAIN_LABEL[i.domain] || i.domain}</span> <span class="pill ${i.category}">${i.category}</span>
      <h2>${esc(i.title)}</h2><p class="muted">${esc(i.detail)}</p>
      <h4>Priority = Severity × Probability × Impact × Urgency</h4>
      <div class="kv"><div>Severity</div><div>${fmt(i.severity, 2)}</div><div>Probability</div><div>${fmt(i.probability, 2)}</div><div>Impact</div><div>${fmt(i.impact, 2)}</div><div>Urgency</div><div>${fmt(i.urgency, 2)}</div><div>Priority score</div><div><b>${fmt(i.priority, 1)}</b> (${i.category})</div><div>First detected</div><div>${esc((i.first_seen || "").replace("T", " "))}</div></div>
      ${i.root_causes?.length ? `<h4>Investigation — likely contributing factors</h4>${i.root_causes.map((c) => `<div class="cause"><span>${esc(c.factor)}<br><span class="muted small">${esc(c.evidence)}</span></span>${bar(c.contribution, 100, col(i.domain))}<span class="num">${c.contribution}%</span></div>`).join("")}` : ""}
      ${i.forecast && Object.keys(i.forecast).length ? `<h4>Prediction</h4><pre class="json">${esc(JSON.stringify(i.forecast, null, 1))}</pre>` : ""}
      ${i.evidence && Object.keys(i.evidence).length ? `<h4>Evidence (from sensors / ML / tools)</h4><pre class="json">${esc(JSON.stringify(i.evidence, null, 1))}</pre>` : ""}
      <h4>Candidate actions</h4>${i.proposals.length ? i.proposals.map((p) => `<div class="tile" style="margin-bottom:8px"><div class="tt"><span>${esc(p.title)}</span><span class="pill ${p.risk}">${p.risk}</span></div><div class="small muted" style="margin-top:4px">${ex(p.expected)}</div></div>`).join("") : '<div class="empty">Monitoring only.</div>'}
      ${i.llm && i.llm !== "pending" ? `<div class="callout">${esc(i.llm)}</div>` : ""}
      ${acts.length ? `<h4>Actions taken</h4><div class="list" style="max-height:none">${acts.map(actionItem).join("")}</div>` : ""}`);
  }

  function openAction(id) {
    const a = S.snap.actions.find((x) => x.id === id) || S.snap.approvals.find((x) => x.id === id);
    if (!a) return;
    S.sheet = { kind: "action", id };
    const v = a.verification;
    const sc = a.safety || {};
    openSheet(`<span class="tag" style="--c:${col(a.domain)}">${DOMAIN_LABEL[a.domain]}</span> <span class="pill ${a.status}">${a.status.replace("_", " ")}</span>
      <h2>${esc(a.title)}</h2><p class="muted">For: ${esc(a.issue_title)}</p>
      <h4>AI decision</h4><div class="callout" style="--c:${col(a.domain)}"><b>${esc(a.decision?.reasoner)}</b><br>${esc(a.decision?.rationale)}</div>
      ${a.decision?.alternatives?.length ? `<div class="small muted">Alternatives considered: ${a.decision.alternatives.map(esc).join(" · ")}</div>` : ""}
      <h4>Safety validation</h4><div class="kv"><div>Risk class</div><div>${a.risk}</div><div>Policy</div><div>${esc(sc.policy)}</div><div>Authorisation</div><div>${sc.requires_authorization ? `required${a.authorized_by ? ` — granted by ${esc(a.authorized_by)}` : ""}` : "not required"}</div><div>Execution</div><div>${esc(a.execution_mode)}</div></div>
      ${(sc.checks || []).map((c) => `<div class="check">✓ ${esc(c)}</div>`).join("")}${(sc.blocked || []).map((c) => `<div class="check" style="color:var(--crit)">✗ ${esc(c)}</div>`).join("")}
      ${a.result?.summary ? `<h4>Action</h4><p>${esc(a.result.summary)}</p>` : ""}
      <h4>Verification — ACTION → MEASURE → COMPARE → VERIFY</h4>
      ${v ? `<div class="kv"><div>Result</div><div><span class="pill ${v.result}">${v.result}</span></div><div>Before</div><div>${fmt(v.before, 2)}</div><div>After</div><div>${fmt(v.after, 2)}</div><div>Without action (twin)</div><div>${v.counterfactual != null ? fmt(v.counterfactual, 2) : "—"}</div><div>Improvement</div><div>${v.improvement_vs_counterfactual_pct != null ? `${fmt(v.improvement_vs_counterfactual_pct, 1)}% vs counterfactual` : v.improvement_pct != null ? `${fmt(v.improvement_pct, 1)}% vs before` : "—"}</div></div><p class="small muted">${esc(v.note || "")}</p>` : `<p class="muted small">${a.status === "PENDING_APPROVAL" ? "Waiting for human authorisation." : "Measuring — verification runs after the action's effect window."}</p>`}
      ${a.status === "PENDING_APPROVAL" ? `<div class="row" style="margin-top:14px"><button class="btn" data-approve="${a.id}">Authorise</button><button class="btn ghost" data-reject="${a.id}">Reject</button></div>` : ""}`);
    bindApprovals($("#sheet-body"));
  }

  function openApprovals() {
    const p = S.snap.approvals;
    S.sheet = { kind: "approvals" };
    openSheet(`<h2>Human authorisation</h2><p class="muted">High-impact actions (emergency systems, critical infrastructure, public-safety resources) pass AI decision → safety validation → <b>your authorisation</b> → action. All execution is simulated in the digital twin.</p>
      ${p.length ? p.map((a) => `<div class="tile" style="margin-bottom:10px"><div class="tt"><span>${esc(a.title)}</span><span class="pill HIGH">HIGH</span></div>
      <div class="small muted" style="margin:6px 0">${esc(a.issue_title)} · requested ${hm(a.created)}</div><div class="small">${esc(a.decision?.rationale)}</div>
      <div class="small muted" style="margin-top:4px">${esc(a.safety?.policy)}</div>
      <div class="row" style="margin-top:10px"><button class="btn" data-approve="${a.id}">Authorise</button><button class="btn ghost" data-reject="${a.id}">Reject</button><button class="btn ghost" data-action="${a.id}">Details</button></div></div>`).join("") : '<div class="empty">Nothing awaiting authorisation.</div>'}`);
    bindApprovals($("#sheet-body"));
  }

  function bindApprovals(root) {
    root.querySelectorAll("[data-approve]").forEach((b) => (b.onclick = async () => {
      b.disabled = true;
      try { await api(`/api/actions/${b.dataset.approve}/approve`, { method: "POST", body: JSON.stringify({ approver: "operator" }) }); }
      catch (e) { alertInline(root, e.message); }
      await refresh(); closeSheet();
    }));
    root.querySelectorAll("[data-reject]").forEach((b) => (b.onclick = async () => {
      b.disabled = true;
      try { await api(`/api/actions/${b.dataset.reject}/reject`, { method: "POST", body: JSON.stringify({ approver: "operator", reason: "operator decision" }) }); }
      catch (e) { alertInline(root, e.message); }
      await refresh(); closeSheet();
    }));
    root.querySelectorAll("[data-action]").forEach((b) => (b.onclick = () => openAction(b.dataset.action)));
  }
  function alertInline(root, msg) { root.insertAdjacentHTML("afterbegin", `<div class="callout" style="--c:var(--crit)">${esc(msg)}</div>`); }

  // ------------------------------------------------------------------ data flow
  function renderAll() {
    if (!S.snap) return;
    if (!S.agent) { renderEntry(); return; }
    renderTop();
    renderHero();
    renderMap();
    renderSide();
    const scrollY = window.scrollY;
    const panelFocus = document.activeElement && $("#panel").contains(document.activeElement);
    if (!(S.agent === "orchestrator" && AG.orchestrator.tabs[S.tab] === "What-if" && panelFocus)) renderPanel();
    window.scrollTo(0, scrollY);
    if (S.sheet?.kind === "approvals") openApprovals();
  }

  function onSnapshot(snap) {
    if (!snap.metrics && S.snap?.metrics) snap.metrics = S.snap.metrics;
    S.snap = snap;
    if (S.agent === "traffic" && AG.traffic.tabs[S.tab] === "Predictions" && Date.now() - S.fcAt > 8000) {
      S.fcAt = Date.now();
      api("/api/traffic/forecast").then((f) => { S.fc = f; if (S.agent === "traffic") renderPanel(); }).catch(() => {});
    }
    renderAll();
  }

  async function refresh() {
    try { onSnapshot(await api("/api/city/snapshot")); } catch (e) { /* engine still starting */ }
  }

  let ws, pollTimer;
  function connect() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    try { ws = new WebSocket(`${proto}://${location.host}/ws`); } catch (e) { return startPolling(); }
    ws.onmessage = (ev) => { const m = JSON.parse(ev.data); if (m.type === "snapshot") onSnapshot(m.data); };
    ws.onopen = () => { clearInterval(pollTimer); pollTimer = null; };
    ws.onclose = () => { startPolling(); setTimeout(connect, 4000); };
  }
  function startPolling() { if (!pollTimer) pollTimer = setInterval(refresh, 2500); }

  async function boot() {
    renderEntry();
    // wait for the engine (first start trains ML models)
    for (;;) {
      try {
        const h = await api("/api/health");
        $("#boot").innerHTML = (h.boot || []).map((b) => `<li>${esc(b)}</li>`).join("");
        if (h.ready) break;
        $("#entry-sub").textContent = "Preparing the city — generating history and training models (first start only)…";
      } catch (e) { $("#entry-sub").textContent = "Waiting for the CityOS engine…"; }
      await new Promise((r) => setTimeout(r, 1500));
    }
    const [geo] = await Promise.all([api("/api/city/twin")]);
    S.geo = geo;
    await refresh();
    connect();
  }

  // ------------------------------------------------------------------ events
  $("#to-entry").onclick = toEntry;
  $("#chip-auth").onclick = openApprovals;
  $("#chip-health").onclick = () => { if (S.agent !== "orchestrator") switchAgent("orchestrator"); };
  $("#sheet-close").onclick = closeSheet;
  $("#scrim").onclick = closeSheet;
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeSheet(); });
  $("#speed").addEventListener("click", async (e) => {
    const b = e.target.closest("button");
    if (!b || !S.snap) return;
    const v = b.dataset.speed;
    const body = v === "pause" ? { running: !S.snap.running } : { running: true, speed: Number(v) };
    const r = await api("/api/city/control", { method: "POST", body: JSON.stringify(body) });
    S.snap.running = r.running; S.snap.speed = r.speed; renderTop();
  });
  $("#phase-filter").addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (!b) return;
    S.filter = b.dataset.f;
    document.querySelectorAll("#phase-filter button").forEach((x) => x.classList.toggle("on", x === b));
    renderSide();
  });
  window.addEventListener("resize", () => { if (S.agent) renderPanel(); });

  boot();
})();
