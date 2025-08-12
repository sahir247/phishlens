const API_BASE = "http://127.0.0.1:8000";

async function analyzeTab(tabId) {
  try {
    // Ask content script for HTML
    const [{ result: html }] = await chrome.scripting.executeScript({
      target: { tabId },
      func: () => document.documentElement.outerHTML,
    });

    const tab = await chrome.tabs.get(tabId);
    const payload = { url: tab.url, html };

    const res = await fetch(`${API_BASE}/check`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error("Backend /check failed");
    const data = await res.json();

    // Store result for popup and notify content to highlight
    await chrome.storage.session.set({ ["phishlens:" + tabId]: data });

    chrome.tabs.sendMessage(tabId, { type: "PHISHLENS_RESULT", data });

    // Badge
    const pct = Math.round((data.risk_score || 0) * 100);
    await chrome.action.setBadgeBackgroundColor({ color: pct >= 80 ? "#e53935" : pct >= 50 ? "#fb8c00" : "#43a047" });
    await chrome.action.setBadgeText({ tabId, text: String(pct) });
  } catch (e) {
    // Clear badge on error
    await chrome.action.setBadgeText({ tabId, text: "" });
    // console.error(e);
  }
}

chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  if (changeInfo.status === "complete" && /^https?:/i.test(tab.url || "")) {
    analyzeTab(tabId);
  }
});

chrome.tabs.onActivated.addListener(async (activeInfo) => {
  const tab = await chrome.tabs.get(activeInfo.tabId);
  if (tab && /^https?:/i.test(tab.url || "")) {
    analyzeTab(activeInfo.tabId);
  }
});

// Provide latest data to popup/content
chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg && msg.type === "PHISHLENS_GET_DATA") {
    const senderTabId = sender?.tab?.id;
    const tabId = msg.tabId || senderTabId;
    if (!tabId) {
      sendResponse({ data: null });
      return; 
    }
    chrome.storage.session.get("phishlens:" + tabId).then((obj) => {
      sendResponse({ data: obj["phishlens:" + tabId] || null });
    });
    return true; // async response
  }
});
