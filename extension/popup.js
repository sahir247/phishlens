async function getActiveTabId() {
  const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
  return tabs[0]?.id;
}

async function getData(tabId) {
  return new Promise((resolve) => {
    chrome.runtime.sendMessage({ type: "PHISHLENS_GET_DATA", tabId }, (resp) => {
      resolve(resp && resp.data ? resp.data : null);
    });
  });
}

function render(data) {
  const scoreEl = document.getElementById("score");
  const reasonsEl = document.getElementById("reasons");
  const metaEl = document.getElementById("meta");

  if (!data) {
    scoreEl.textContent = "--";
    reasonsEl.textContent = "No data yet. Refresh the page.";
    metaEl.textContent = "";
    return;
  }
  const pct = Math.round((data.risk_score || 0) * 100);
  scoreEl.textContent = pct + "%";
  scoreEl.style.background = pct >= 80 ? "#e53935" : pct >= 50 ? "#fb8c00" : "#43a047";

  reasonsEl.innerHTML = "";
  (data.reasons || []).forEach((r) => {
    const li = document.createElement("div");
    li.className = "reason";
    li.textContent = "• " + r;
    reasonsEl.appendChild(li);
  });

  const domain = data.meta?.domain || "";
  metaEl.textContent = domain ? `Domain: ${domain}` : "";
}

async function main() {
  const tabId = await getActiveTabId();
  const data = await getData(tabId);
  render(data);

  document.getElementById("explain").onclick = async () => {
    const latest = await getData(tabId);
    if (!latest) return;
    chrome.tabs.sendMessage(tabId, { type: "PHISHLENS_RESULT", data: latest });
    chrome.tabs.sendMessage(tabId, { type: "PHISHLENS_APPLY", selectors: latest.highlights || [] });
  };

  document.getElementById("refresh").onclick = async () => {
    // Reload tab to trigger re-analysis via background
    chrome.tabs.reload(tabId);
    setTimeout(async () => render(await getData(tabId)), 1200);
  };
}

main();
