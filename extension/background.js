// Service worker: owns the one WebSocket connection to the local Python service
// and relays messages between it and any YouTube Shorts tab.
//
// The content script deliberately does NOT open the socket itself. A page-context
// connection to ws://127.0.0.1 from an https page runs into mixed-content rules,
// and every tab would open its own socket. Here there is exactly one.

const WS_URL = "ws://127.0.0.1:8765";
const RECONNECT_MIN = 1000;
const RECONNECT_MAX = 15000;
const STATE_RELAY_HZ = 5; // throttle HUD updates; actions are never throttled

let socket = null;
let reconnectDelay = RECONNECT_MIN;
let reconnectTimer = null;
let lastRelay = 0;

let settings = { enabled: true, sensitivity: 0.5 };
let lastState = { connected: false };

// --------------------------------------------------------------------------
// settings
// --------------------------------------------------------------------------

async function loadSettings() {
  const stored = await chrome.storage.local.get(["enabled", "sensitivity"]);
  settings = {
    enabled: stored.enabled !== undefined ? stored.enabled : true,
    sensitivity: stored.sensitivity !== undefined ? stored.sensitivity : 0.5,
  };
}

function pushSettingsToBackend() {
  send({ type: "enabled", value: settings.enabled });
  send({ type: "sensitivity", value: settings.sensitivity });
}

// --------------------------------------------------------------------------
// socket
// --------------------------------------------------------------------------

function setBadge(connected) {
  chrome.action.setBadgeText({ text: connected ? "on" : "" });
  chrome.action.setBadgeBackgroundColor({ color: connected ? "#1f8b4c" : "#8b1f1f" });
}

function connect() {
  if (socket && (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING)) {
    return;
  }
  clearTimeout(reconnectTimer);

  try {
    socket = new WebSocket(WS_URL);
  } catch (err) {
    scheduleReconnect();
    return;
  }

  socket.onopen = () => {
    reconnectDelay = RECONNECT_MIN;
    lastState = { ...lastState, connected: true };
    setBadge(true);
    pushSettingsToBackend();
    broadcastToTabs({ type: "connection", connected: true });
  };

  socket.onmessage = (event) => {
    let msg;
    try {
      msg = JSON.parse(event.data);
    } catch (err) {
      return;
    }

    if (msg.type === "state") {
      lastState = { ...msg, connected: true };
      const now = Date.now();
      if (now - lastRelay >= 1000 / STATE_RELAY_HZ) {
        lastRelay = now;
        broadcastToTabs({ type: "state", state: msg });
      }
    } else if (msg.type === "action") {
      if (settings.enabled) broadcastToTabs({ type: "action", action: msg });
    }
  };

  socket.onclose = () => {
    lastState = { ...lastState, connected: false };
    setBadge(false);
    broadcastToTabs({ type: "connection", connected: false });
    scheduleReconnect();
  };

  socket.onerror = () => {
    try {
      socket.close();
    } catch (err) {
      /* already closing */
    }
  };
}

function scheduleReconnect() {
  clearTimeout(reconnectTimer);
  reconnectTimer = setTimeout(connect, reconnectDelay);
  reconnectDelay = Math.min(RECONNECT_MAX, Math.round(reconnectDelay * 1.6));
}

function send(payload) {
  if (socket && socket.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify(payload));
    return true;
  }
  return false;
}

// --------------------------------------------------------------------------
// tab relay
// --------------------------------------------------------------------------

async function broadcastToTabs(message) {
  const tabs = await chrome.tabs.query({ url: "*://www.youtube.com/*" });
  for (const tab of tabs) {
    chrome.tabs.sendMessage(tab.id, message).catch(() => {
      /* tab has no content script yet - fine */
    });
  }
}

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  switch (msg.type) {
    case "getStatus":
      sendResponse({ settings, state: lastState });
      return true;

    case "setEnabled":
      settings.enabled = !!msg.value;
      chrome.storage.local.set({ enabled: settings.enabled });
      send({ type: "enabled", value: settings.enabled });
      broadcastToTabs({ type: "settings", settings });
      sendResponse({ ok: true, settings });
      return true;

    case "setSensitivity":
      settings.sensitivity = Number(msg.value);
      chrome.storage.local.set({ sensitivity: settings.sensitivity });
      send({ type: "sensitivity", value: settings.sensitivity });
      broadcastToTabs({ type: "settings", settings });
      sendResponse({ ok: true, settings });
      return true;

    case "videoChanged":
      // dwell = seconds spent on the video we just left; auto = we skipped it
      // ourselves, so it says nothing about what the user actually wanted.
      send({ type: "video_changed", dwell: msg.dwell, auto: !!msg.auto });
      sendResponse({ ok: true });
      return true;

    case "reconnect":
      reconnectDelay = RECONNECT_MIN;
      connect();
      sendResponse({ ok: true });
      return true;

    default:
      return false;
  }
});

// --------------------------------------------------------------------------
// lifecycle
// --------------------------------------------------------------------------

// An open WebSocket keeps a modern MV3 worker alive, but the alarm is cheap
// insurance in case the socket dies while the worker is idle.
chrome.alarms.create("keepalive", { periodInMinutes: 0.5 });
chrome.alarms.onAlarm.addListener(() => {
  if (!socket || socket.readyState !== WebSocket.OPEN) connect();
});

chrome.runtime.onStartup.addListener(async () => {
  await loadSettings();
  connect();
});
chrome.runtime.onInstalled.addListener(async () => {
  await loadSettings();
  connect();
});

(async () => {
  await loadSettings();
  setBadge(false);
  connect();
})();
