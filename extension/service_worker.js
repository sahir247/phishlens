/**
 * PhishLens Background Service Worker (Manifest V3) — v0.3.0
 * Analyses each tab on load/activation, caches result in session storage,
 * and updates the action badge with the risk score.
 * Shows "OFF" badge in grey when backend is unreachable.
 */

const API_BASE = "http://127.0.0.1:8000";

// Track when session data was last updated per tab
const _scanTimestamps = {};

async function analyzeTab(tabId) {
  try {
    const tab = await chrome.tabs.get(tabId);
    if (!tab || !tab.url || !/^https?:/i.test(tab.url)) {
      return;
    }

    // Capture tab DOM HTML
    const injectionResults = await chrome.scripting.executeScript({
      target: { tabId },
      func: () => document.documentElement.outerHTML,
    });

    const html = injectionResults?.[0]?.result || "";
    const payload = { url: tab.url, html };

    // Request PhishLens Backend Assessment
    const res = await fetch(`${API_BASE}/check`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!res.ok) throw new Error(`Backend /check responded with status ${res.status}`);
    const data = await res.json();

    // Cache in session storage for popup and in-page inspector
    const cacheEntry = { ...data, _cachedAt: Date.now() };
    await chrome.storage.session.set({ [`phishlens:${tabId}`]: cacheEntry });
    _scanTimestamps[tabId] = Date.now();

    // Notify in-page content script
    chrome.tabs.sendMessage(tabId, { type: "PHISHLENS_RESULT", data }).catch(() => {});

    // Update Action Badge
    const pct = Math.round((data.risk_score || 0) * 100);
    const badgeColor = pct >= 80 ? "#ff3366" : (pct >= 50 ? "#f59e0b" : "#10b981");

    await chrome.action.setBadgeBackgroundColor({ tabId, color: badgeColor });
    await chrome.action.setBadgeText({ tabId, text: `${pct}%` });

    // Store backend-online state
    await chrome.storage.session.set({ "phishlens:backend_online": true });

  } catch (err) {
    // Mark backend as offline
    await chrome.storage.session.set({ "phishlens:backend_online": false }).catch(() => {});

    // Show grey "OFF" badge
    await chrome.action.setBadgeBackgroundColor({ tabId, color: "#475569" }).catch(() => {});
    await chrome.action.setBadgeText({ tabId, text: "OFF" }).catch(() => {});

    // Store error for popup to display
    await chrome.storage.session.set({
      [`phishlens:${tabId}:error`]: { message: "Backend offline", ts: Date.now() }
    }).catch(() => {});
  }
}

// Tab lifecycle event listeners
chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  if (changeInfo.status === "complete" && tab.url && /^https?:/i.test(tab.url)) {
    analyzeTab(tabId);
  }
});

chrome.tabs.onActivated.addListener(async (activeInfo) => {
  const tab = await chrome.tabs.get(activeInfo.tabId).catch(() => null);
  if (tab && tab.url && /^https?:/i.test(tab.url)) {
    analyzeTab(activeInfo.tabId);
  }
});

// Message hub for popup and content scripts
chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  // Called from popup.js (passes explicit tabId)
  if (msg && msg.type === "PHISHLENS_GET_DATA") {
    const tabId = msg.tabId || sender?.tab?.id;
    if (!tabId) {
      sendResponse({ data: null, backendOnline: false });
      return false;
    }
    chrome.storage.session.get([
      `phishlens:${tabId}`,
      `phishlens:${tabId}:error`,
      "phishlens:backend_online",
    ]).then((res) => {
      sendResponse({
        data: res[`phishlens:${tabId}`] || null,
        error: res[`phishlens:${tabId}:error`] || null,
        backendOnline: res["phishlens:backend_online"] !== false,
      });
    });
    return true;
  }

  // Called from content.js banner "Explain" button — resolves active tab automatically
  if (msg && msg.type === "PHISHLENS_GET_CURRENT_TAB_DATA") {
    chrome.tabs.query({ active: true, currentWindow: true }).then((tabs) => {
      const tabId = tabs?.[0]?.id;
      if (!tabId) { sendResponse({ data: null }); return; }
      chrome.storage.session.get([`phishlens:${tabId}`]).then((res) => {
        sendResponse({ data: res[`phishlens:${tabId}`] || null });
      });
    });
    return true;
  }
});
