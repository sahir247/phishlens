/**
 * PhishLens Chrome Extension Popup Engine — v0.3.0
 * Shows risk gauge, category bars, reasons, and handles backend-offline state.
 */

async function getActiveTab() {
  const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
  return tabs[0];
}

async function getTabData(tabId) {
  return new Promise((resolve) => {
    chrome.runtime.sendMessage({ type: "PHISHLENS_GET_DATA", tabId }, (resp) => {
      resolve(resp || { data: null, backendOnline: false, error: null });
    });
  });
}

function updateGauge(score, tier) {
  const pct = Math.round((score || 0) * 100);
  const scoreEl = document.getElementById("score-val");
  const arcEl   = document.getElementById("gauge-arc");
  const badgeEl = document.getElementById("badge-level");

  scoreEl.textContent = pct + "%";

  // Arc calculation
  const circumference = 314.159; // 2 * PI * 50
  const offset = circumference * (1 - (pct / 100));
  arcEl.style.strokeDasharray  = `${circumference}`;
  arcEl.style.strokeDashoffset = `${offset}`;

  let color = "#10b981";
  let badgeClass = "badge-safe";
  let tierText   = "SAFE";

  if (pct >= 80 || tier === "DANGEROUS") {
    color      = "#ff3366";
    badgeClass = "badge-dangerous";
    tierText   = "DANGEROUS";
  } else if (pct >= 50 || tier === "SUSPICIOUS") {
    color      = "#f59e0b";
    badgeClass = "badge-suspicious";
    tierText   = "SUSPICIOUS";
  }

  arcEl.style.stroke       = color;
  badgeEl.className        = `badge-pill ${badgeClass}`;
  badgeEl.textContent      = tierText;
}

function showOfflineState(tab) {
  const domainEl  = document.getElementById("domain-text");
  const badgeEl   = document.getElementById("badge-level");
  const reasonsEl = document.getElementById("reasons-list");

  domainEl.textContent = tab?.url ? new URL(tab.url).hostname : "—";
  updateGauge(0, "SAFE");

  badgeEl.className   = "badge-pill";
  badgeEl.textContent = "OFFLINE";
  badgeEl.style.background    = "#475569";
  badgeEl.style.color         = "#fff";

  reasonsEl.innerHTML = `
    <div class="reason-item text-muted" style="color:#f59e0b;">
      ⚠ PhishLens backend is offline.<br>
      Start it with <code>run_phishlens.bat</code> and reload this tab.
    </div>
  `;
}

function setButtonLoading(loading) {
  const btn = document.getElementById("btn-recheck");
  if (!btn) return;
  if (loading) {
    btn.textContent = "Scanning…";
    btn.disabled    = true;
  } else {
    btn.textContent = "Recheck";
    btn.disabled    = false;
  }
}

function render(data, tab, backendOnline) {
  const domainEl  = document.getElementById("domain-text");
  const brandWrap = document.getElementById("brand-alert-wrap");
  const brandChip = document.getElementById("brand-target-chip");
  const reasonsEl = document.getElementById("reasons-list");

  if (!backendOnline) {
    showOfflineState(tab);
    return;
  }

  if (!data) {
    domainEl.textContent = tab?.url ? new URL(tab.url).hostname : "No active page";
    updateGauge(0, "SAFE");
    reasonsEl.innerHTML = '<div class="reason-item text-muted">No scan data yet. Refresh tab.</div>';
    return;
  }

  const domain = data.meta?.domain || (tab?.url ? new URL(tab.url).hostname : "Unknown Domain");
  domainEl.textContent = domain;

  updateGauge(data.risk_score, data.risk_level);

  // Allowlisted domain special display
  if (data.allowlisted) {
    reasonsEl.innerHTML = `<div class="reason-item" style="color:#10b981;">
      ✓ Verified legitimate domain. Allowlisted as official ${(data.allowlisted_brand || "").toUpperCase()} site.
    </div>`;
    brandWrap.style.display = "none";
    return;
  }

  // Brand Alert
  const brand = data.brand_target || data.meta?.brand_target;
  if (brand) {
    brandWrap.style.display = "block";
    brandChip.textContent   = `Impersonating ${brand}`;
  } else {
    brandWrap.style.display = "none";
  }

  // Category Sub-Scores (including ml_risk)
  const cat  = data.category_scores || {};
  const dPct = Math.round((cat.domain_risk || 0) * 100);
  const uPct = Math.round((cat.url_risk    || 0) * 100);
  const bPct = Math.round((cat.brand_risk  || 0) * 100);
  const mPct = Math.round((cat.dom_risk    || 0) * 100);
  const mlPct = Math.round((cat.ml_risk    || 0) * 100);

  document.getElementById("cat-domain-val").textContent  = `${dPct}%`;
  document.getElementById("cat-domain-fill").style.width = `${dPct}%`;

  document.getElementById("cat-url-val").textContent  = `${uPct}%`;
  document.getElementById("cat-url-fill").style.width = `${uPct}%`;

  document.getElementById("cat-brand-val").textContent  = `${bPct}%`;
  document.getElementById("cat-brand-fill").style.width = `${bPct}%`;

  document.getElementById("cat-dom-val").textContent  = `${mPct}%`;
  document.getElementById("cat-dom-fill").style.width = `${mPct}%`;

  // ML bar (may or may not exist in older popup.html — guard)
  const mlVal  = document.getElementById("cat-ml-val");
  const mlFill = document.getElementById("cat-ml-fill");
  if (mlVal)  mlVal.textContent  = `${mlPct}%`;
  if (mlFill) mlFill.style.width = `${mlPct}%`;

  // Reasons
  reasonsEl.innerHTML = "";
  if (data.structured_reasons && data.structured_reasons.length) {
    data.structured_reasons.forEach(r => {
      const item = document.createElement("div");
      item.className   = `reason-item ${r.severity}`;
      item.textContent = `• [${r.severity}] ${r.title}`;
      reasonsEl.appendChild(item);
    });
  } else if (data.reasons && data.reasons.length) {
    data.reasons.forEach(r => {
      const item = document.createElement("div");
      item.className   = "reason-item";
      item.textContent = `• ${r}`;
      reasonsEl.appendChild(item);
    });
  } else {
    reasonsEl.innerHTML = '<div class="reason-item text-muted">No phishing triggers detected. Clean site.</div>';
  }
}

async function init() {
  const tab = await getActiveTab();
  if (!tab || !tab.id) return;

  const { data, backendOnline, error } = await getTabData(tab.id);
  render(data, tab, backendOnline);

  // Button: Highlight Elements in Page
  document.getElementById("btn-explain").onclick = async () => {
    const { data: latest } = await getTabData(tab.id);
    if (!latest) return;
    chrome.tabs.sendMessage(tab.id, { type: "PHISHLENS_RESULT", data: latest });
    chrome.tabs.sendMessage(tab.id, { type: "PHISHLENS_APPLY", selectors: latest.highlights || [] });
  };

  // Button: Recheck — reload tab which triggers fresh analysis
  document.getElementById("btn-recheck").onclick = async () => {
    setButtonLoading(true);
    try {
      chrome.tabs.reload(tab.id);
      window.close();
    } catch {
      setButtonLoading(false);
    }
  };

  // Button: Open Dashboard
  document.getElementById("btn-open-dashboard").onclick = () => {
    chrome.tabs.create({ url: "http://127.0.0.1:8000/dashboard/index.html" });
  };
}

document.addEventListener("DOMContentLoaded", init);
