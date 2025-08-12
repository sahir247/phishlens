// Inject banner and highlight suspicious elements when requested by background/popup

const HIGHLIGHT_CLASS = "phishlens-highlight";

function ensureBanner() {
  let banner = document.getElementById("phishlens-banner");
  if (!banner) {
    banner = document.createElement("div");
    banner.id = "phishlens-banner";
    banner.style.cssText = `position:fixed;top:0;left:0;right:0;z-index:2147483647;` +
      `padding:10px 16px;font: 13px/1.4 system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Noto Sans, sans-serif;` +
      `display:none;align-items:center;gap:12px;` +
      `box-shadow:0 2px 8px rgba(0,0,0,0.15);`;
    const text = document.createElement("div");
    text.id = "phishlens-banner-text";
    const btnExplain = document.createElement("button");
    btnExplain.textContent = "Explain";
    btnExplain.style.cssText = `padding:6px 10px;border-radius:6px;border:none;background:#1e88e5;color:#fff;cursor:pointer;`;
    btnExplain.onclick = () => {
      // Ask background for latest data for this tab and then highlight
      chrome.runtime.sendMessage({ type: "PHISHLENS_GET_DATA" }, (resp) => {
        const data = resp && resp.data ? resp.data : {};
        applyHighlights(data.highlights || []);
      });
    };
    const btnDismiss = document.createElement("button");
    btnDismiss.textContent = "Ignore";
    btnDismiss.style.cssText = `padding:6px 10px;border-radius:6px;border:1px solid #777;background:#fff;color:#333;cursor:pointer;`;
    btnDismiss.onclick = () => {
      banner.style.display = "none";
      clearHighlights();
    };
    banner.appendChild(text);
    banner.appendChild(btnExplain);
    banner.appendChild(btnDismiss);
    document.documentElement.appendChild(banner);
    // Push content down when banner visible
    const spacer = document.createElement("div");
    spacer.id = "phishlens-banner-spacer";
    spacer.style.height = "0px";
    document.body && document.body.prepend(spacer);
  }
  return banner;
}

function setBanner(score, reasons) {
  const banner = ensureBanner();
  const pct = Math.round((score || 0) * 100);
  const color = pct >= 80 ? "#ffebee" : pct >= 50 ? "#fff8e1" : "#e8f5e9";
  const border = pct >= 80 ? "#e53935" : pct >= 50 ? "#fb8c00" : "#43a047";
  banner.style.background = color;
  banner.style.borderBottom = `3px solid ${border}`;
  banner.style.display = pct >= 50 ? "flex" : "none";
  const text = banner.querySelector("#phishlens-banner-text");
  text.textContent = `PhishLens: Risk ${pct}%` + (reasons?.length ? ` — ${reasons.slice(0,3).join('; ')}` : "");
  const spacer = document.getElementById("phishlens-banner-spacer");
  if (spacer) spacer.style.height = banner.style.display === "flex" ? "48px" : "0px";
}

function clearHighlights() {
  document.querySelectorAll(`.${HIGHLIGHT_CLASS}`).forEach((el) => {
    el.classList.remove(HIGHLIGHT_CLASS);
  });
}

function applyHighlights(selectors = []) {
  clearHighlights();
  try {
    selectors.forEach((sel) => {
      document.querySelectorAll(sel).forEach((el) => {
        el.classList.add(HIGHLIGHT_CLASS);
      });
    });
  } catch (e) {
    // Invalid selectors may throw; ignore
  }
}

// Listen for results from background
chrome.runtime.onMessage.addListener((msg) => {
  if (msg?.type === "PHISHLENS_RESULT") {
    const d = msg.data || {};
    setBanner(d.risk_score, d.reasons);
  } else if (msg?.type === "PHISHLENS_APPLY") {
    const sels = msg.selectors || [];
    applyHighlights(sels);
  }
});

// Inject highlight CSS
(function injectCSS(){
  const style = document.createElement('style');
  style.textContent = `.phishlens-highlight{outline:3px solid #e53935 !important; background: rgba(229,57,53,.06) !important;}`;
  document.documentElement.appendChild(style);
})();
