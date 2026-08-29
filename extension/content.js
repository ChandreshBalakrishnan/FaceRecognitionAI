// Runs on youtube.com. Draws a heads-up display, reports how long you spend on
// each video, and performs the actual "go to the next Short" when the background
// worker relays a skip.

(() => {
  if (window.__emotionScrollLoaded) return;
  window.__emotionScrollLoaded = true;

  const HUD_ID = "emotion-scroll-hud";
  let hud = null;
  let settings = { enabled: true, sensitivity: 0.5 };
  let connected = false;
  let lastHref = location.href;
  let lastSkipAt = 0;
  let hudHidden = false;

  // Dwell reporting: how long we have been on this video, and whether leaving it
  // was our own doing. Auto-skips must not teach the backend that you like
  // short videos - that would be a feedback loop.
  let videoStartedAt = Date.now();
  let leavingBecauseOfUs = false;

  const onShorts = () => location.pathname.startsWith("/shorts/");

  // ------------------------------------------------------------------ HUD

  function buildHud() {
    if (hud || !document.body) return;
    hud = document.createElement("div");
    hud.id = HUD_ID;
    hud.innerHTML = `
      <div class="es-row">
        <span class="es-dot"></span>
        <span class="es-title">Emotion Scroll</span>
        <span class="es-state">off</span>
      </div>
      <div class="es-reason">waiting for camera…</div>
      <div class="es-bars">
        <div class="es-bar"><span>away</span><i><b data-k="attention_loss"></b></i></div>
        <div class="es-bar"><span>drowsy</span><i><b data-k="drowsiness"></b></i></div>
        <div class="es-bar"><span>emotion</span><i><b data-k="emotion"></b></i></div>
      </div>
      <div class="es-meter"><div class="es-fill"></div><div class="es-mark"></div></div>
      <div class="es-foot"><span class="es-dwell"></span></div>
      <div class="es-hint">alt+E arm / disarm · alt+H hide</div>
    `;
    const style = document.createElement("style");
    style.textContent = `
      #${HUD_ID} {
        position: fixed; right: 16px; bottom: 16px; z-index: 2147483647;
        width: 216px; padding: 10px 12px; border-radius: 10px;
        background: rgba(18,18,20,.88); color: #eee; backdrop-filter: blur(8px);
        font: 12px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        box-shadow: 0 6px 24px rgba(0,0,0,.45); pointer-events: none;
        transition: opacity .2s ease;
      }
      #${HUD_ID}.es-hidden { opacity: 0; }
      #${HUD_ID} .es-row { display: flex; align-items: center; gap: 6px; }
      #${HUD_ID} .es-dot { width: 7px; height: 7px; border-radius: 50%; background: #b33; }
      #${HUD_ID}.es-connected .es-dot { background: #3b3; }
      #${HUD_ID} .es-title { font-weight: 600; letter-spacing: .2px; }
      #${HUD_ID} .es-state { margin-left: auto; font-size: 11px; color: #999; text-transform: uppercase; }
      #${HUD_ID}.es-armed .es-state { color: #7d7; }
      #${HUD_ID} .es-reason { margin: 7px 0 6px; font-size: 12px; color: #ddd; }
      #${HUD_ID} .es-bars { display: grid; gap: 3px; margin-bottom: 7px; }
      #${HUD_ID} .es-bar { display: flex; align-items: center; gap: 6px; font-size: 10px; color: #8e8e97; }
      #${HUD_ID} .es-bar span { width: 46px; flex: none; }
      #${HUD_ID} .es-bar i { flex: 1; height: 4px; border-radius: 2px; background: #33333c; overflow: hidden; }
      #${HUD_ID} .es-bar b { display: block; height: 100%; width: 0%; background: #5a9bd8; transition: width .2s linear; }
      #${HUD_ID} .es-meter { position: relative; height: 6px; border-radius: 3px; background: #333; overflow: hidden; }
      #${HUD_ID} .es-fill { height: 100%; width: 0%; background: linear-gradient(90deg,#4a8,#e85); transition: width .15s linear; }
      #${HUD_ID} .es-mark { position: absolute; top: -2px; width: 2px; height: 10px; background: #fff; left: 55%; }
      #${HUD_ID} .es-foot { margin-top: 6px; font-size: 10px; color: #8e8e97; min-height: 13px; }
      #${HUD_ID} .es-hint { margin-top: 5px; font-size: 10px; color: #6a6a72; }
      #${HUD_ID}.es-fired { box-shadow: 0 0 0 2px #e85, 0 6px 24px rgba(0,0,0,.45); }
    `;
    document.documentElement.appendChild(style);
    document.body.appendChild(hud);
  }

  function updateHud(state) {
    if (!hud) return;
    hud.classList.toggle("es-connected", connected);
    hud.classList.toggle("es-armed", connected && settings.enabled);
    hud.querySelector(".es-state").textContent = !connected
      ? "no camera"
      : settings.enabled
      ? "armed"
      : "paused";

    if (!state) return;

    const reasonEl = hud.querySelector(".es-reason");
    if (state.presence !== undefined && state.presence < 0.5) {
      reasonEl.textContent = "you're not at the camera";
    } else if (state.blocked_by) {
      reasonEl.textContent = state.blocked_by;
    } else if (state.holding > 0) {
      reasonEl.textContent = `${state.reason} · ${state.holding.toFixed(1)}s`;
    } else {
      reasonEl.textContent = state.reason || "watching";
    }

    const comp = state.components || {};
    hud.querySelectorAll(".es-bar b").forEach((bar) => {
      const v = comp[bar.dataset.k] || 0;
      bar.style.width = `${Math.min(100, v * 100)}%`;
    });

    hud.querySelector(".es-fill").style.width = `${Math.min(
      100,
      (state.disengagement || 0) * 100
    )}%`;
    hud.querySelector(".es-mark").style.left = `${Math.min(
      100,
      (state.threshold || 0.55) * 100
    )}%`;

    const blink = state.blink_rate == null ? "—" : `${Math.round(state.blink_rate)}/min`;
    hud.querySelector(".es-dwell").textContent =
      `${(state.dwell || 0).toFixed(0)}s / ${(state.dwell_gate || 0).toFixed(0)}s · ` +
      `blink ${blink}` +
      (state.dwell_learned ? "" : ` · learning (${state.dwell_samples || 0})`);
  }

  function flashHud() {
    if (!hud) return;
    hud.classList.add("es-fired");
    setTimeout(() => hud && hud.classList.remove("es-fired"), 450);
  }

  function setHudVisible(visible) {
    hudHidden = !visible;
    if (hud) hud.classList.toggle("es-hidden", hudHidden);
  }

  // -------------------------------------------------------------- skipping

  function findScroller() {
    let el = document.querySelector("ytd-reel-video-renderer video, video");
    while (el && el !== document.body) {
      const s = getComputedStyle(el);
      if (/(auto|scroll)/.test(s.overflowY) && el.scrollHeight > el.clientHeight + 10) {
        return el;
      }
      el = el.parentElement;
    }
    return document.scrollingElement || document.documentElement;
  }

  // Several strategies, tried in order, because YouTube changes its markup often.
  const strategies = [
    function navButton() {
      const btn = document.querySelector(
        "#navigation-button-down button, " +
          "#navigation-button-down .yt-spec-button-shape-next, " +
          'button[aria-label="Next video"]'
      );
      if (btn && true) {
        btn.click();
        return "nav-button";
      }
      return null;
    },
    function nextRenderer() {
      // Ask the next Short to bring itself into view. Usually the most reliable
      // of the three, because it does not depend on a button's markup or on the
      // page honouring a synthetic keypress.
      const items = [...document.querySelectorAll("ytd-reel-video-renderer")];
      if (items.length < 2) return null;
      const current = items.findIndex((el) => el.getBoundingClientRect().top > -50);
      const next = items[current + 1];
      if (!next) return null;
      next.scrollIntoView({ behavior: "smooth", block: "start" });
      return "next-renderer";
    },
    function scrollContainer() {
      const scroller = findScroller();
      if (!scroller) return null;
      const step = scroller.clientHeight || window.innerHeight;
      scroller.scrollBy({ top: step, behavior: "smooth" });
      return "scroll";
    },
    function arrowKey() {
      for (const type of ["keydown", "keyup"]) {
        document.dispatchEvent(
          new KeyboardEvent(type, {
            key: "ArrowDown",
            code: "ArrowDown",
            keyCode: 40,
            which: 40,
            bubbles: true,
            cancelable: true,
          })
        );
      }
      return "arrow-key";
    },
  ];

  async function goNext() {
    const before = location.href;
    for (const strategy of strategies) {
      const used = strategy();
      if (!used) continue;
      const moved = await waitForChange(before, 900);
      if (moved) return used;
    }
    return null;
  }

  function waitForChange(before, timeout) {
    return new Promise((resolve) => {
      const start = Date.now();
      const tick = () => {
        if (location.href !== before) return resolve(true);
        if (Date.now() - start > timeout) return resolve(false);
        setTimeout(tick, 80);
      };
      tick();
    });
  }

  function dropSkip(reason) {
    // Never fail silently. The backend logs every skip it decides on, so if one
    // does not happen the console has to say why - otherwise you are left
    // comparing a terminal that says "skip" against a feed that did not move.
    console.info(`[Emotion Scroll] skip ignored: ${reason}`);
    if (hud) hud.querySelector(".es-reason").textContent = `ignored: ${reason}`;
  }

  async function handleSkip() {
    if (!onShorts()) return dropSkip("not on a Short");
    if (!settings.enabled) return dropSkip("auto-skip is off");

    // Deliberately NOT checking document.hasFocus(). visibilityState already
    // rules out background tabs, and requiring focus breaks the whole point:
    // you looking away from the browser is exactly when this should fire. It
    // also silently disabled every skip whenever DevTools had focus.
    if (document.visibilityState !== "visible") return dropSkip("tab is in the background");
    if (Date.now() - lastSkipAt < 2000) return dropSkip("too soon after the last skip");

    lastSkipAt = Date.now();
    leavingBecauseOfUs = true;
    flashHud();

    const used = await goNext();
    if (used) {
      console.info(`[Emotion Scroll] advanced via ${used}`);
    } else {
      leavingBecauseOfUs = false;
      console.warn(
        "[Emotion Scroll] could not advance - all navigation strategies failed, " +
          "so YouTube's layout has probably changed"
      );
      if (hud) hud.querySelector(".es-reason").textContent = "could not scroll";
    }
  }

  // ---------------------------------------------------------- video change

  setInterval(() => {
    if (location.href === lastHref) return;
    lastHref = location.href;

    const dwell = (Date.now() - videoStartedAt) / 1000;
    videoStartedAt = Date.now();
    const auto = leavingBecauseOfUs;
    leavingBecauseOfUs = false;

    if (onShorts()) {
      chrome.runtime.sendMessage({ type: "videoChanged", dwell, auto }).catch(() => {});
    }
    if (hud) hud.style.display = onShorts() ? "" : "none";
  }, 400);

  // -------------------------------------------------------------- messages

  chrome.runtime.onMessage.addListener((msg) => {
    if (msg.type === "state") {
      updateHud(msg.state);
    } else if (msg.type === "action") {
      handleSkip();
    } else if (msg.type === "connection") {
      connected = msg.connected;
      updateHud(null);
    } else if (msg.type === "settings") {
      settings = msg.settings;
      updateHud(null);
    }
  });

  document.addEventListener("keydown", (e) => {
    if (!e.altKey || e.metaKey || e.ctrlKey) return;
    if (e.code === "KeyE") {
      e.preventDefault();
      chrome.runtime.sendMessage({ type: "setEnabled", value: !settings.enabled }).catch(() => {});
    } else if (e.code === "KeyH") {
      e.preventDefault();
      setHudVisible(hudHidden);
    }
  });

  // -------------------------------------------------------------- start up

  function boot() {
    buildHud();
    if (hud) hud.style.display = onShorts() ? "" : "none";
    chrome.runtime
      .sendMessage({ type: "getStatus" })
      .then((res) => {
        if (!res) return;
        settings = res.settings || settings;
        connected = !!(res.state && res.state.connected);
        updateHud(res.state);
      })
      .catch(() => {});
  }

  if (document.body) boot();
  else document.addEventListener("DOMContentLoaded", boot, { once: true });
})();
