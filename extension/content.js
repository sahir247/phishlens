/**
 * PhishLens In-Page Threat Protection & DOM Explainability Engine
 */

const HIGHLIGHT_CLASS = "phishlens-flagged-element";
const BADGE_CLASS = "phishlens-element-badge";

// Inject CSS Styles for In-Page Banners, Overlays, and Highlighting
(function injectPhishLensCSS() {
  if (document.getElementById("phishlens-injected-styles")) return;
  const style = document.createElement("style");
  style.id = "phishlens-injected-styles";
  style.textContent = `
    /* Flagged Element Highlight */
    .${HIGHLIGHT_CLASS} {
      outline: 3px solid #ff3366 !important;
      outline-offset: 2px !important;
      box-shadow: 0 0 16px rgba(255, 51, 102, 0.45) !important;
      position: relative !important;
      background: rgba(255, 51, 102, 0.08) !important;
      transition: all 0.3s ease !important;
    }
    .${BADGE_CLASS} {
      position: absolute;
      top: -12px;
      left: 0;
      background: #ff3366;
      color: #fff;
      font: 700 10px/1 'JetBrains Mono', system-ui, sans-serif;
      padding: 3px 6px;
      border-radius: 3px;
      z-index: 2147483647;
      pointer-events: none;
      box-shadow: 0 2px 6px rgba(0,0,0,0.4);
      text-transform: uppercase;
    }

    /* Floating Warning Banner (50-79%) */
    #phishlens-banner {
      position: fixed;
      top: 0;
      left: 0;
      right: 0;
      z-index: 2147483646;
      padding: 12px 20px;
      background: rgba(15, 23, 42, 0.95);
      backdrop-filter: blur(12px);
      border-bottom: 2px solid #f59e0b;
      box-shadow: 0 6px 20px rgba(0,0,0,0.4);
      color: #f8fafc;
      font: 13px/1.4 'Inter', system-ui, -apple-system, sans-serif;
      display: none;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
    }
    #phishlens-banner.visible {
      display: flex;
    }
    .phishlens-banner-left {
      display: flex;
      align-items: center;
      gap: 10px;
      flex: 1;
    }
    .phishlens-banner-tag {
      background: rgba(245, 158, 11, 0.2);
      color: #f59e0b;
      border: 1px solid rgba(245, 158, 11, 0.4);
      padding: 2px 8px;
      border-radius: 4px;
      font-weight: 700;
      font-size: 11px;
    }
    .phishlens-banner-btn {
      padding: 6px 12px;
      border-radius: 6px;
      font-size: 12px;
      font-weight: 600;
      cursor: pointer;
      border: none;
      transition: all 0.2s ease;
    }
    .phishlens-btn-explain {
      background: #00f2fe;
      color: #051329;
    }
    .phishlens-btn-dismiss {
      background: rgba(255, 255, 255, 0.1);
      color: #fff;
      border: 1px solid rgba(255, 255, 255, 0.2);
    }

    /* Severe Threat Interstitial Blocking Overlay (>=80%) */
    #phishlens-blocking-overlay {
      position: fixed;
      inset: 0;
      background: rgba(10, 14, 24, 0.98);
      backdrop-filter: blur(20px);
      z-index: 2147483647;
      color: #fff;
      font-family: 'Inter', system-ui, -apple-system, sans-serif;
      display: none;
      align-items: center;
      justify-content: center;
      padding: 24px;
    }
    #phishlens-blocking-overlay.visible {
      display: flex;
    }
    .phishlens-overlay-card {
      max-width: 580px;
      width: 100%;
      background: #11192e;
      border: 2px solid #ff3366;
      border-radius: 16px;
      padding: 32px;
      box-shadow: 0 0 50px rgba(255, 51, 102, 0.35);
      text-align: center;
    }
    .phishlens-overlay-icon {
      width: 64px;
      height: 64px;
      margin: 0 auto 16px;
      background: rgba(255, 51, 102, 0.15);
      border: 1px solid rgba(255, 51, 102, 0.4);
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      color: #ff3366;
    }
    .phishlens-overlay-title {
      font-size: 22px;
      font-weight: 700;
      color: #ff3366;
      margin-bottom: 8px;
    }
    .phishlens-overlay-desc {
      font-size: 14px;
      color: #94a3b8;
      line-height: 1.5;
      margin-bottom: 20px;
    }
    .phishlens-reasons-summary {
      background: rgba(0, 0, 0, 0.4);
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: 8px;
      padding: 12px 16px;
      text-align: left;
      font-size: 13px;
      color: #cbd5e1;
      margin-bottom: 24px;
      max-height: 140px;
      overflow-y: auto;
    }
    .phishlens-overlay-actions {
      display: flex;
      flex-direction: column;
      gap: 10px;
    }
    .phishlens-btn-safety {
      padding: 12px 24px;
      background: #ff3366;
      color: #fff;
      border: none;
      border-radius: 8px;
      font-size: 15px;
      font-weight: 700;
      cursor: pointer;
      box-shadow: 0 4px 16px rgba(255, 51, 102, 0.4);
    }
    .phishlens-btn-bypass {
      background: transparent;
      border: none;
      color: #64748b;
      font-size: 12px;
      cursor: pointer;
      text-decoration: underline;
    }
    .phishlens-btn-bypass:hover {
      color: #94a3b8;
    }
  `;
  document.documentElement.appendChild(style);
})();

// Create / Retrieve Floating Banner
function ensureBanner() {
  let banner = document.getElementById("phishlens-banner");
  if (!banner) {
    banner = document.createElement("div");
    banner.id = "phishlens-banner";
    banner.innerHTML = `
      <div class="phishlens-banner-left">
        <span class="phishlens-banner-tag" id="phishlens-banner-tag">RISK 65%</span>
        <span id="phishlens-banner-text">Suspicious activity detected on this page.</span>
      </div>
      <div style="display:flex;gap:8px;">
        <button class="phishlens-banner-btn phishlens-btn-explain" id="phishlens-btn-explain-action">Explain</button>
        <button class="phishlens-banner-btn phishlens-btn-dismiss" id="phishlens-btn-dismiss-action">Ignore</button>
      </div>
    `;
    document.documentElement.appendChild(banner);

    document.getElementById("phishlens-btn-explain-action").onclick = () => {
      // Ask the background worker for the cached result for THIS tab.
      // We must pass the tabId explicitly because sendMessage from a
      // content script does NOT populate sender.tab.id on the other end.
      chrome.runtime.sendMessage(
        { type: "PHISHLENS_GET_CURRENT_TAB_DATA" },
        (resp) => {
          const highlights = resp?.data?.highlights || [];
          applyHighlights(highlights);
        }
      );
    };

    document.getElementById("phishlens-btn-dismiss-action").onclick = () => {
      banner.classList.remove("visible");
      clearHighlights();
    };
  }
  return banner;
}

// Create / Retrieve Severe Blocking Overlay
function ensureBlockingOverlay() {
  let overlay = document.getElementById("phishlens-blocking-overlay");
  if (!overlay) {
    overlay = document.createElement("div");
    overlay.id = "phishlens-blocking-overlay";
    overlay.innerHTML = `
      <div class="phishlens-overlay-card">
        <div class="phishlens-overlay-icon">
          <svg viewBox="0 0 24 24" width="36" height="36" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
            <line x1="12" y1="9" x2="12" y2="13"/>
            <line x1="12" y1="17" x2="12.01" y2="17"/>
          </svg>
        </div>
        <div class="phishlens-overlay-title">High-Risk Phishing Blocked</div>
        <div class="phishlens-overlay-desc">
          PhishLens intercepted a dangerous phishing attempt. Attackers may attempt to steal your passwords, credentials, or personal information on this page.
        </div>
        <div class="phishlens-reasons-summary" id="phishlens-overlay-reasons"></div>
        <div class="phishlens-overlay-actions">
          <button class="phishlens-btn-safety" id="phishlens-btn-safety">← Take Me Back to Safety</button>
          <button class="phishlens-btn-bypass" id="phishlens-btn-bypass">I understand the risks, bypass warning and proceed</button>
        </div>
      </div>
    `;
    document.documentElement.appendChild(overlay);

    document.getElementById("phishlens-btn-safety").onclick = () => {
      if (window.history.length > 1) {
        window.history.back();
      } else {
        window.location.href = "about:blank";
      }
    };

    document.getElementById("phishlens-btn-bypass").onclick = () => {
      overlay.classList.remove("visible");
    };
  }
  return overlay;
}

// Display Alerts Based on Risk Score
function updateInPageAlerts(score, reasons, brandTarget) {
  const pct = Math.round((score || 0) * 100);

  if (pct >= 80) {
    // Critical Threat Overlay
    const overlay = ensureBlockingOverlay();
    const reasonsBox = document.getElementById("phishlens-overlay-reasons");
    const reasonsList = (reasons && reasons.length) ? reasons.map(r => `<div>• ${r}</div>`).join("") : "<div>• Deceptive domain and credential harvesting detected.</div>";
    const brandNotice = brandTarget ? `<div style="color:#ff3366;font-weight:700;margin-bottom:6px;">Target Impersonation: ${brandTarget.toUpperCase()}</div>` : "";
    reasonsBox.innerHTML = brandNotice + reasonsList;
    overlay.classList.add("visible");
  } else if (pct >= 50) {
    // Moderate Risk Banner
    const banner = ensureBanner();
    document.getElementById("phishlens-banner-tag").textContent = `RISK ${pct}%`;
    const brandText = brandTarget ? ` (Impersonating ${brandTarget})` : "";
    const reasonText = (reasons && reasons.length) ? reasons[0] : "Suspicious indicators found";
    document.getElementById("phishlens-banner-text").textContent = `PhishLens Warning: ${reasonText}${brandText}`;
    banner.classList.add("visible");
  }
}

// Clear Highlights
function clearHighlights() {
  document.querySelectorAll(`.${HIGHLIGHT_CLASS}`).forEach(el => {
    el.classList.remove(HIGHLIGHT_CLASS);
  });
  document.querySelectorAll(`.${BADGE_CLASS}`).forEach(el => {
    el.remove();
  });
}

// Apply Highlights to flagged DOM Selectors
function applyHighlights(selectors = []) {
  clearHighlights();
  if (!selectors || !selectors.length) return;

  selectors.forEach(sel => {
    try {
      document.querySelectorAll(sel).forEach(el => {
        el.classList.add(HIGHLIGHT_CLASS);
        // Create badge if not already tagged
        const badge = document.createElement("span");
        badge.className = BADGE_CLASS;
        badge.textContent = "PhishLens Flagged";
        if (getComputedStyle(el).position === "static") {
          el.style.position = "relative";
        }
        el.appendChild(badge);
      });
    } catch (e) {
      // Ignore invalid CSS selectors
    }
  });
}

// Message Listener from Background and Popup
chrome.runtime.onMessage.addListener((msg) => {
  if (msg?.type === "PHISHLENS_RESULT") {
    const d = msg.data || {};
    updateInPageAlerts(d.risk_score, d.reasons, d.brand_target);
  } else if (msg?.type === "PHISHLENS_APPLY") {
    applyHighlights(msg.selectors || []);
  }
});
