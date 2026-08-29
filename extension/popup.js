const el = (id) => document.getElementById(id);

function render({ settings, state }) {
  const connected = state && state.connected;
  el("dot").classList.toggle("on", !!connected);
  el("status").textContent = connected
    ? settings.enabled
      ? "Armed"
      : "Connected, auto-skip off"
    : "Service not running";

  if (!connected) {
    el("detail").textContent = "Run  python server.py  in the backend folder.";
    el("dwell").textContent = "";
    el("fill").style.width = "0%";
    el("mark").style.left = "55%";
    document.querySelectorAll(".bar b").forEach((b) => (b.style.width = "0%"));
  } else {
    if (state.presence !== undefined && state.presence < 0.5) {
      el("detail").textContent = "No face in view";
    } else if (state.blocked_by) {
      el("detail").textContent = state.blocked_by;
    } else {
      el("detail").textContent = `${state.reason} · ${Number(state.disengagement || 0).toFixed(
        2
      )} of ${Number(state.threshold || 0).toFixed(2)}`;
    }

    const comp = state.components || {};
    document.querySelectorAll(".bar b").forEach((b) => {
      b.style.width = `${Math.min(100, (comp[b.dataset.k] || 0) * 100)}%`;
    });

    el("fill").style.width = `${Math.min(100, (state.disengagement || 0) * 100)}%`;
    el("mark").style.left = `${Math.min(100, (state.threshold || 0.55) * 100)}%`;

    const blink = state.blink_rate == null ? "—" : `${Math.round(state.blink_rate)}/min`;
    el("dwell").textContent =
      `watched ${Number(state.dwell || 0).toFixed(0)}s · ` +
      `your median ${Number(state.dwell_median || 0).toFixed(0)}s · blink ${blink}` +
      (state.dwell_learned ? "" : ` · learning (${state.dwell_samples || 0}/5)`);
  }

  el("enabled").checked = !!settings.enabled;
  el("sensitivity").value = Math.round((settings.sensitivity ?? 0.5) * 100);
}

async function refresh() {
  try {
    const res = await chrome.runtime.sendMessage({ type: "getStatus" });
    if (res) render(res);
  } catch (err) {
    /* worker still waking up */
  }
}

el("enabled").addEventListener("change", async (e) => {
  await chrome.runtime.sendMessage({ type: "setEnabled", value: e.target.checked });
  refresh();
});

el("sensitivity").addEventListener("input", async (e) => {
  await chrome.runtime.sendMessage({ type: "setSensitivity", value: e.target.value / 100 });
});

el("reconnect").addEventListener("click", async () => {
  await chrome.runtime.sendMessage({ type: "reconnect" });
  setTimeout(refresh, 400);
});

refresh();
setInterval(refresh, 500);
