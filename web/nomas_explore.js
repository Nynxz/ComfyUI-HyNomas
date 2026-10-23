// Nomas Explore viewer: record a camera move with WASD / arrows / mouse drag at 24 fps, run the
// next turn, play it back. The recording is written into the node's `controls` widget as
// {session, op, nonce, ticks: [[keys, dx, dy], ...]} and the graph is queued.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const FPS = 24;
const KEYMAP = {
  KeyW: "w", KeyA: "a", KeyS: "s", KeyD: "d",
  ArrowUp: "up", ArrowDown: "down", ArrowLeft: "left", ArrowRight: "right",
};

const STYLE = `
.nomas-explore { display:flex; flex-direction:column; gap:6px; width:100%; height:100%; font:12px sans-serif; color:#ddd; }
.nomas-view { position:relative; flex:1; min-height:160px; background:#111; border-radius:6px; overflow:hidden; outline:none; cursor:crosshair; }
.nomas-view:focus { box-shadow:0 0 0 2px #4a9eff inset; }
.nomas-view video { width:100%; height:100%; object-fit:contain; display:block; }
.nomas-hud { position:absolute; left:8px; bottom:8px; display:grid; grid-template-columns:repeat(3,22px) 10px repeat(3,22px); gap:3px; pointer-events:none; }
.nomas-key { width:22px; height:22px; border-radius:4px; background:rgba(0,0,0,.55); border:1px solid rgba(255,255,255,.25);
  display:flex; align-items:center; justify-content:center; font-size:11px; }
.nomas-key.on { background:#4a9eff; border-color:#9cf; color:#fff; }
.nomas-msg { position:absolute; left:0; right:0; top:8px; text-align:center; pointer-events:none; text-shadow:0 1px 2px #000; }
.nomas-bar { position:absolute; left:0; top:0; height:3px; background:#4a9eff; width:0; }
.nomas-row { display:flex; gap:6px; align-items:center; flex-wrap:wrap; }
.nomas-row button { background:#333; color:#ddd; border:1px solid #555; border-radius:4px; padding:3px 8px; cursor:pointer; }
.nomas-row button:hover { background:#444; }
.nomas-row select { background:#222; color:#ddd; border:1px solid #555; border-radius:4px; }
`;

function addStyle() {
  if (document.getElementById("nomas-explore-style")) return;
  const s = document.createElement("style");
  s.id = "nomas-explore-style";
  s.textContent = STYLE;
  document.head.appendChild(s);
}

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}

function setupNode(node) {
  addStyle();
  const controls = node.widgets.find((w) => w.name === "controls");
  const change = node.widgets.find((w) => w.name === "change");
  const action = node.widgets.find((w) => w.name === "action");
  // a new seed starts a new world, so the seed must not change between turns
  const seedControl = node.widgets.find((w) => w.name === "control_after_generate");
  if (seedControl) seedControl.value = "fixed";
  let session = node.properties?.nomas_session;
  if (!session) {
    session = (crypto.randomUUID?.() || String(Math.random())).slice(0, 12);
    node.properties = node.properties || {};
    node.properties.nomas_session = session;
  }

  const root = el("div", "nomas-explore");
  const view = el("div", "nomas-view");
  view.tabIndex = 0;
  const video = el("video");
  video.loop = true; video.muted = true; video.playsInline = true;
  const msg = el("div", "nomas-msg", "Click here, then hold W A S D / arrows (or drag) to move");
  const bar = el("div", "nomas-bar");
  const hud = el("div", "nomas-hud");
  const keys = {};
  for (const k of ["", "w", "", "|", "", "up", "", "a", "s", "d", "|", "left", "down", "right"]) {
    const label = { left: "◀", right: "▶", up: "▲", down: "▼" }[k] || k.toUpperCase();
    const key = el("div", k === "|" ? "" : "nomas-key", k === "|" ? "" : label);
    if (!k) key.style.visibility = "hidden";
    else if (k !== "|") keys[k] = key;
    hud.appendChild(key);
  }
  view.append(video, msg, bar, hud);

  const row = el("div", "nomas-row");
  const length = el("select");
  for (const s of [1, 2, 3, 4, 5, 8]) {
    const o = el("option", "", `${s} s turn`);
    o.value = s;
    if (s === 3) o.selected = true;
    length.appendChild(o);
  }
  const textTurn = el("button", "", "Text turn");
  textTurn.title = "Run the next turn without a camera move (uses 'change' / 'action')";
  const undo = el("button", "", "Undo");
  const reset = el("button", "", "New world");
  const playAll = el("button", "", "Play all");
  playAll.title = "Play the whole world from the start";
  const save = el("button", "", "Save video");
  save.title = "Save the whole world as an mp4 in output/video";
  const state = el("span", "", "");
  row.append(length, textTurn, undo, reset, playAll, save, state);
  root.append(view, row);

  node.addDOMWidget("nomas_viewer", "nomas_viewer", root, { serialize: false, hideOnZoom: false });
  node.setSize([Math.max(node.size[0], 480), Math.max(node.size[1], 560)]);

  const held = new Set();
  let drag = null, mdx = 0, mdy = 0;
  let recording = null, busy = false;

  const paint = () => {
    for (const [k, e] of Object.entries(keys)) e.classList.toggle("on", held.has(k));
  };

  const submit = (op, ticks) => {
    controls.value = JSON.stringify({ session, op, nonce: Date.now(), ticks: ticks || [] });
    busy = true;
    msg.textContent = { undo: "Undoing…", view: "Loading the whole world…", save: "Saving the video…" }[op] || "Generating the next turn…";
    app.queuePrompt(0, 1);
  };

  const stop = () => {
    if (!recording) return;
    clearInterval(recording.timer);
    const ticks = recording.ticks;
    recording = null;
    bar.style.width = "0";
    if (ticks.length) submit("turn", ticks);
  };

  const start = () => {
    if (recording || busy) return;
    const total = Number(length.value) * FPS;
    recording = { ticks: [], timer: setInterval(() => {
      recording.ticks.push([[...held], mdx, mdy]);
      mdx = 0; mdy = 0;
      bar.style.width = `${(100 * recording.ticks.length) / total}%`;
      if (recording.ticks.length >= total) stop();
    }, 1000 / FPS) };
    msg.textContent = "Recording… (Space or Enter to finish early)";
  };

  view.addEventListener("keydown", (e) => {
    const k = KEYMAP[e.code];
    if (k) {
      e.preventDefault(); e.stopPropagation();
      held.add(k); paint(); start();
    } else if (e.code === "Space" || e.code === "Enter") {
      e.preventDefault(); e.stopPropagation(); stop();
    }
  });
  view.addEventListener("keyup", (e) => {
    const k = KEYMAP[e.code];
    if (k) { e.preventDefault(); e.stopPropagation(); held.delete(k); paint(); }
  });
  view.addEventListener("blur", () => { held.clear(); paint(); });
  view.addEventListener("contextmenu", (e) => e.preventDefault());
  view.addEventListener("pointerdown", (e) => {
    view.focus();
    drag = [e.clientX, e.clientY];
    view.setPointerCapture(e.pointerId);
    e.stopPropagation();
  });
  view.addEventListener("pointermove", (e) => {
    if (!drag) return;
    const dx = e.clientX - drag[0], dy = e.clientY - drag[1];
    drag = [e.clientX, e.clientY];
    if (dx || dy) { mdx += dx; mdy += dy; start(); }
    e.stopPropagation();
  });
  view.addEventListener("pointerup", () => { drag = null; });
  // keep the canvas from treating keys typed into the viewer as shortcuts
  for (const ev of ["keypress", "wheel"]) view.addEventListener(ev, (e) => e.stopPropagation());

  textTurn.onclick = () => { if (!busy) submit("turn", []); };
  undo.onclick = () => { if (!busy) submit("undo", []); };
  reset.onclick = () => { if (!busy) submit("reset", []); };
  playAll.onclick = () => { if (!busy) submit("view", []); };
  save.onclick = () => { if (!busy) submit("save", []); };

  node.nomasShow = (clip, info) => {
    busy = false;
    if (clip) {
      const p = new URLSearchParams({ filename: clip.filename, subfolder: clip.subfolder || "", type: clip.type || "temp" });
      video.src = api.apiURL(`/view?${p}`);
      video.play().catch(() => {});
    }
    if (info) {
      state.textContent = `${info.turns} turns · ${info.seconds} s`;
      undo.disabled = !info.can_undo;
    }
    if (info?.saved) {
      msg.textContent = `Saved output/${info.saved}`;
      return;
    }
    msg.textContent = "Hold W A S D / arrows (or drag) for the next move";
    // a text turn's words are used once
    if (info?.op === "turn") {
      if (change) change.value = "";
      if (action) action.value = "";
    }
  };
  node.nomasFailed = () => { busy = false; msg.textContent = "The turn failed — see the console. Try again."; };
}

app.registerExtension({
  name: "nynxz.Nomas.Explore",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "NomasExplore") return;
    const created = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = created?.apply(this, arguments);
      setupNode(this);
      return r;
    };
    const executed = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      executed?.apply(this, arguments);
      this.nomasShow?.(message?.nomas_clip?.[0], message?.nomas_state?.[0]);
    };
  },
  setup() {
    api.addEventListener("execution_error", (e) => {
      for (const node of app.graph?._nodes || []) {
        if (node.type === "NomasExplore" && String(node.id) === String(e.detail?.node_id)) node.nomasFailed?.();
      }
    });
  },
});
