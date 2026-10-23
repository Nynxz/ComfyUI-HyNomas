// Nomas Explore viewer.
//
// Record a camera move with W A S D / the arrow keys / a mouse drag at 24 fps; it is written into
// the node's `controls` widget as {session, op, nonce, ticks: [[keys, dx, dy], ...]} and the graph
// is queued. The server answers with the world's turns (frame ranges, kind, label, camera path and
// a clip each), which the viewer plays back-to-back on one timeline, with a top-down map of where
// the camera has been.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const FPS = 24;
const KEYMAP = {
  KeyW: "w", KeyA: "a", KeyS: "s", KeyD: "d",
  ArrowUp: "up", ArrowDown: "down", ArrowLeft: "left", ArrowRight: "right",
};
const KIND = {
  opening: { color: "#8b7cf6", name: "Opening" },
  camera: { color: "#3fa7ff", name: "Camera" },
  text: { color: "#f5a524", name: "Text" },
  both: { color: "#2dd4a0", name: "Camera + text" },
};

const STYLE = `
.nx { display:flex; flex-direction:column; gap:8px; width:100%; height:100%; font:12px/1.35 system-ui, sans-serif; color:#d8dbe2; user-select:none; }
.nx-view { position:relative; flex:1; min-height:200px; background:#0b0d12; border-radius:8px; overflow:hidden; outline:none; cursor:crosshair; }
.nx-view.nx-focus { box-shadow:0 0 0 2px #3fa7ff inset; }
.nx-view video { position:absolute; inset:0; width:100%; height:100%; object-fit:contain; background:#0b0d12; }
.nx-view video.nx-hidden { visibility:hidden; }
.nx-empty { position:absolute; inset:0; display:flex; flex-direction:column; align-items:center; justify-content:center; gap:6px; color:#7d8494; text-align:center; padding:20px; }
.nx-empty b { color:#d8dbe2; font-size:14px; }
.nx-top { position:absolute; left:10px; right:220px; top:10px; display:flex; gap:8px; align-items:flex-start; pointer-events:none; }
.nx-pill { background:rgba(10,12,18,.72); backdrop-filter:blur(6px); border:1px solid rgba(255,255,255,.08); border-radius:999px; padding:4px 10px; white-space:nowrap; }
.nx-status { max-width:100%; overflow:hidden; text-overflow:ellipsis; }
.nx-status.nx-busy::before, .nx-status.nx-rec::before { content:""; display:inline-block; width:8px; height:8px; margin-right:6px; border-radius:50%; vertical-align:0; animation:nxpulse 1s infinite; }
.nx-status.nx-busy::before { background:#3fa7ff; }
.nx-status.nx-rec::before { background:#ff4d5e; }
@keyframes nxpulse { 50% { opacity:.25 } }
.nx-progress { position:absolute; left:0; bottom:0; height:3px; width:0; background:linear-gradient(90deg,#3fa7ff,#8b7cf6); transition:width .25s; }
.nx-rec-bar { position:absolute; left:0; top:0; height:3px; width:0; background:#ff4d5e; }
.nx-fly { position:absolute; inset:0; width:100%; height:100%; display:none; pointer-events:none; }
.nx-fly.nx-on { display:block; }
.nx-map { position:absolute; right:10px; top:10px; width:200px; height:130px; border-radius:8px; background:rgba(10,12,18,.72); border:1px solid rgba(255,255,255,.08); pointer-events:none; }
.nx-hud { position:absolute; left:10px; bottom:12px; display:flex; gap:10px; pointer-events:none; }
.nx-keys { display:grid; grid-template-columns:repeat(3,22px); gap:3px; }
.nx-key { width:22px; height:22px; border-radius:5px; background:rgba(10,12,18,.65); border:1px solid rgba(255,255,255,.18); display:flex; align-items:center; justify-content:center; font-size:10px; font-weight:600; }
.nx-key.on { background:#3fa7ff; border-color:#9fd2ff; color:#fff; }
.nx-gap { visibility:hidden; }
.nx-timeline { display:flex; flex-direction:column; gap:4px; }
.nx-transport { display:flex; align-items:center; gap:8px; }
.nx-time { font-variant-numeric:tabular-nums; color:#9aa1b1; min-width:84px; text-align:right; }
.nx-track { position:relative; flex:1; height:26px; cursor:pointer; }
.nx-segs { position:absolute; left:0; right:0; top:7px; bottom:7px; display:flex; gap:2px; }
.nx-seg { height:100%; border-radius:3px; opacity:.55; min-width:3px; transition:opacity .15s; }
.nx-seg:hover, .nx-seg.nx-cur { opacity:1; }
.nx-head { position:absolute; top:2px; bottom:2px; width:2px; margin-left:-1px; background:#fff; border-radius:1px; box-shadow:0 0 0 1px rgba(0,0,0,.5); pointer-events:none; }
.nx-meta { display:flex; gap:12px; align-items:center; }
.nx-label { flex:1; color:#aeb4c2; font-size:11px; min-height:15px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.nx-legend { display:flex; gap:10px; color:#7d8494; font-size:11px; }
.nx-legend i { display:inline-block; width:8px; height:8px; border-radius:2px; margin-right:4px; vertical-align:0; }
.nx-bar { display:flex; gap:6px; align-items:center; flex-wrap:wrap; }
.nx-btn { display:inline-flex; align-items:center; gap:5px; background:#232733; color:#d8dbe2; border:1px solid #343a48; border-radius:6px; padding:4px 9px; cursor:pointer; font:inherit; }
.nx-btn:hover:not(:disabled) { background:#2c3140; border-color:#4a5164; }
.nx-btn:disabled { opacity:.4; cursor:default; }
.nx-btn.nx-primary { background:#1f4f86; border-color:#2f6db5; }
.nx-btn.nx-icon { padding:4px 8px; min-width:30px; justify-content:center; }
.nx-sel { background:#232733; color:#d8dbe2; border:1px solid #343a48; border-radius:6px; padding:3px 6px; font:inherit; }
.nx-spacer { flex:1; }
`;

function addStyle() {
  if (document.getElementById("nx-style")) return;
  const s = document.createElement("style");
  s.id = "nx-style";
  s.textContent = STYLE;
  document.head.appendChild(s);
}

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}

function button(label, title, cls = "") {
  const b = el("button", `nx-btn ${cls}`, label);
  b.title = title;
  return b;
}

const clock = (frames) => {
  const s = Math.max(0, frames) / FPS;
  return `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
};

// ---- a tiny wireframe renderer (Three.js axes: y up, the camera looks along -z, Euler 'YXZ') ----
const TURN = (20 * Math.PI) / 180 / FPS; // arrow keys, per 24 fps tick (as nomas/camera.py)
const MOVE = 2.8 / FPS;
const MOUSE = 0.0024;

function basis(yaw, pitch) {
  const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
  return { right: [cy, 0, -sy], up: [sy * sp, cp, cy * sp], back: [sy * cp, -sp, cy * cp] };
}

// one 24 fps control tick, exactly as Pose.step on the server
function stepPose(p, keys, dx, dy) {
  const { back } = basis(p.yaw, p.pitch);
  const f = back.map((v) => -v);
  let r = [-f[2], 0, f[0]];
  const n = Math.hypot(r[0], r[2]);
  r = n > 1e-8 ? r.map((v) => v / n) : basis(p.yaw, p.pitch).right;
  const fw = (keys.has("w") ? 1 : 0) - (keys.has("s") ? 1 : 0);
  const st = (keys.has("d") ? 1 : 0) - (keys.has("a") ? 1 : 0);
  return {
    x: p.x + (f[0] * fw + r[0] * st) * MOVE,
    y: p.y + (f[1] * fw + r[1] * st) * MOVE,
    z: p.z + (f[2] * fw + r[2] * st) * MOVE,
    yaw: p.yaw + TURN * ((keys.has("left") ? 1 : 0) - (keys.has("right") ? 1 : 0)) - dx * MOUSE,
    pitch: p.pitch + TURN * ((keys.has("up") ? 1 : 0) - (keys.has("down") ? 1 : 0)) - dy * MOUSE,
  };
}

function viewer3d(canvas, eye, fovDeg) {
  const ctx = canvas.getContext("2d");
  const w = canvas.width, h = canvas.height;
  const { right, up, back } = basis(eye.yaw, eye.pitch);
  const f = h / 2 / Math.tan((fovDeg * Math.PI) / 360);
  const near = 0.05;
  const cam = (p) => {
    const d = [p[0] - eye.x, p[1] - eye.y, p[2] - eye.z];
    return [d[0] * right[0] + d[1] * right[1] + d[2] * right[2], d[0] * up[0] + d[1] * up[1] + d[2] * up[2],
      -(d[0] * back[0] + d[1] * back[1] + d[2] * back[2])];
  };
  const screen = ([x, y, z]) => [w / 2 + (f * x) / z, h / 2 - (f * y) / z];
  return {
    ctx,
    line(a, b, color, alpha = 1, width = 1, fog = 0) {
      let p = cam(a), q = cam(b);
      if (p[2] < near && q[2] < near) return;
      if (p[2] < near || q[2] < near) {
        const [i, o] = p[2] < near ? [q, p] : [p, q];
        const t = (i[2] - near) / (i[2] - o[2]);
        const c = [i[0] + (o[0] - i[0]) * t, i[1] + (o[1] - i[1]) * t, near];
        if (p[2] < near) p = c; else q = c;
      }
      const depth = (p[2] + q[2]) / 2;
      ctx.globalAlpha = alpha * (fog ? Math.exp(-fog * depth) : 1);
      ctx.strokeStyle = color; ctx.lineWidth = width;
      const [u0, v0] = screen(p), [u1, v1] = screen(q);
      ctx.beginPath(); ctx.moveTo(u0, v0); ctx.lineTo(u1, v1); ctx.stroke();
    },
    dot(p, color, r) {
      const c = cam(p);
      if (c[2] < near) return;
      const [u, v] = screen(c);
      ctx.globalAlpha = 1; ctx.fillStyle = color;
      ctx.beginPath(); ctx.arc(u, v, r, 0, 2 * Math.PI); ctx.fill();
    },
  };
}

// a camera frustum at pose p (for the path view)
function frustum(v, p, color, size) {
  const { right, up, back } = basis(p.yaw, p.pitch);
  const o = [p.x, p.y, p.z];
  const at = (s, t) => [0, 1, 2].map((i) => o[i] - back[i] * size + right[i] * s * size * 0.7 + up[i] * t * size * 0.45);
  const c = [at(-1, -1), at(1, -1), at(1, 1), at(-1, 1)];
  for (let i = 0; i < 4; i++) { v.line(o, c[i], color, 0.9, 1.5); v.line(c[i], c[(i + 1) % 4], color, 0.9, 1.5); }
}

const ICO = (() => { // the demo's subject marker: an icosahedron at (0, 1.5, 0)
  const t = (1 + Math.sqrt(5)) / 2, s = 1.15 / Math.hypot(1, t);
  const v = [[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t], [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]]
    .map(([x, y, z]) => [x * s, y * s + 1.5, z * s]);
  const edges = [];
  for (let i = 0; i < 12; i++) for (let j = i + 1; j < 12; j++) {
    if (Math.abs(Math.hypot(v[i][0] - v[j][0], v[i][1] - v[j][1], v[i][2] - v[j][2]) - 2 * s) < 1e-3) edges.push([v[i], v[j]]);
  }
  return edges;
})();

function setupNode(node) {
  addStyle();
  const controls = node.widgets.find((w) => w.name === "controls");
  const change = node.widgets.find((w) => w.name === "change");
  const action = node.widgets.find((w) => w.name === "action");
  // a new seed starts a new world, so the seed must not change between turns
  const seedControl = node.widgets.find((w) => w.name === "control_after_generate");
  if (seedControl) seedControl.value = "fixed";
  node.properties = node.properties || {};
  let session = node.properties.nomas_session;
  if (!session) {
    session = (crypto.randomUUID?.() || String(Math.random())).replace(/-/g, "").slice(0, 12);
    node.properties.nomas_session = session;
  }

  // ---- DOM ----
  const root = el("div", "nx");
  const view = el("div", "nx-view");
  view.tabIndex = 0;
  const videos = [el("video"), el("video")];
  for (const v of videos) { v.muted = true; v.playsInline = true; v.preload = "auto"; v.classList.add("nx-hidden"); }
  const empty = el("div", "nx-empty");
  empty.append(el("b", "", "No world yet"), el("div", "", "Queue the graph (or press New world) to generate the opening turn."));
  const top = el("div", "nx-top");
  const status = el("div", "nx-pill nx-status", "Click the view, then hold W A S D / arrows or drag to move");
  top.append(status);
  const map = el("canvas", "nx-map");
  map.width = 400; map.height = 260;
  map.title = "The camera's path through the world";
  const fly = el("canvas", "nx-fly");
  const progress = el("div", "nx-progress");
  const recBar = el("div", "nx-rec-bar");
  const hud = el("div", "nx-hud");
  const keyEls = {};
  const keyGrid = (layout) => {
    const g = el("div", "nx-keys");
    for (const k of layout) {
      const label = { left: "◀", right: "▶", up: "▲", down: "▼" }[k] || k.toUpperCase();
      const e = el("div", k ? "nx-key" : "nx-key nx-gap", label);
      if (k) keyEls[k] = e;
      g.appendChild(e);
    }
    return g;
  };
  hud.append(keyGrid(["", "w", "", "a", "s", "d"]), keyGrid(["", "up", "", "left", "down", "right"]));
  view.append(...videos, fly, empty, top, map, hud, progress, recBar);

  const timeline = el("div", "nx-timeline");
  const transport = el("div", "nx-transport");
  const play = button("▶", "Play / pause the whole world (Space)", "nx-icon");
  const live = button("⏭", "Jump to the newest turn", "nx-icon");
  const time = el("span", "nx-time", "0:00 / 0:00");
  const track = el("div", "nx-track");
  const segs = el("div", "nx-segs");
  const head = el("div", "nx-head");
  track.append(segs, head);
  transport.append(play, live, track, time);
  const meta = el("div", "nx-meta");
  const label = el("div", "nx-label", "");
  const legend = el("div", "nx-legend");
  for (const k of Object.values(KIND)) {
    const s = el("span"); const i = el("i"); i.style.background = k.color; s.append(i, k.name); legend.appendChild(s);
  }
  meta.append(label, legend);
  timeline.append(transport, meta);

  const bar = el("div", "nx-bar");
  const length = el("select", "nx-sel");
  for (const s of [1, 2, 3, 4, 5, 8]) {
    const o = el("option", "", `${s} s move`); o.value = s; if (s === 3) o.selected = true; length.appendChild(o);
  }
  length.title = "Longest recording; Space / Enter ends a move early";
  const textTurn = button("✎ Text turn", "Run the next turn from 'change' / 'action' without moving");
  const undo = button("↶ Undo", "Drop the last turn");
  const reset = button("✦ New world", "Start over from the prompt");
  const save = button("⤓ Save video", "Save the whole world to output/video", "nx-primary");
  bar.append(length, textTurn, undo, el("span", "nx-spacer"), reset, save);
  root.append(view, timeline, bar);

  node.addDOMWidget("nomas_viewer", "nomas_viewer", root, { serialize: false, hideOnZoom: false, getMinHeight: () => 520 });
  node.setSize([Math.max(node.size[0], 760), Math.max(node.size[1], 820)]);

  // ---- world state & playback ----
  let world = null; // {turns:[{start, frames, kind, label, path, clip}], frames, can_undo}
  let cur = -1; // the turn on screen
  let active = 0; // which <video> is visible
  let playing = false;
  let busy = false;
  const clipURL = (c) => api.apiURL(`/view?${new URLSearchParams({ filename: c.filename, subfolder: c.subfolder || "", type: c.type || "temp" })}`);

  const position = () => {
    if (!world || cur < 0) return 0;
    return world.turns[cur].start + Math.round((videos[active].currentTime || 0) * FPS);
  };

  const paintTimeline = () => {
    const total = world?.frames || 0;
    const pos = Math.min(position(), total);
    time.textContent = `${clock(pos)} / ${clock(total)}`;
    head.style.left = total ? `${(100 * pos) / total}%` : "0";
    [...segs.children].forEach((s, i) => s.classList.toggle("nx-cur", i === cur));
    if (world && cur >= 0) {
      const t = world.turns[cur];
      label.textContent = `Turn ${cur + 1} of ${world.turns.length} · ${KIND[t.kind]?.name || t.kind} · ${t.label}`;
    }
    play.textContent = playing ? "❚❚" : "▶";
  };

  const poseAt = (frame) => { // the camera at a world frame, from the turns' paths (one point per 4 frames)
    if (!world || !world.turns.length) return null;
    let i = world.turns.findIndex((t) => frame < t.start + t.frames);
    if (i < 0) i = world.turns.length - 1;
    const t = world.turns[i];
    const k = Math.max(0, Math.min((frame - t.start) / 4, t.path.length - 1));
    const a = t.path[Math.floor(k)], b = t.path[Math.ceil(k)], u = k - Math.floor(k);
    const [x, y, z, yaw, pitch] = a.map((v, j) => v + (b[j] - v) * u);
    return { x, y, z, yaw, pitch };
  };

  const drawMap = () => {
    const ctx = map.getContext("2d");
    ctx.globalAlpha = 1;
    ctx.clearRect(0, 0, map.width, map.height);
    const pts = (world?.turns || []).flatMap((t) => t.path);
    if (!pts.length) return;
    let lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    for (const p of pts) for (let i = 0; i < 3; i++) { lo[i] = Math.min(lo[i], p[i]); hi[i] = Math.max(hi[i], p[i]); }
    const c = lo.map((v, i) => (v + hi[i]) / 2);
    const r = Math.max(hi[0] - lo[0], hi[2] - lo[2], hi[1] - lo[1], 3) / 2;
    // an elevated three-quarter view of the path, like the demo's playback overlay
    const dist = r * 2.6 + 4, pitch = -0.62, yaw = 0.5;
    const { back } = basis(yaw, pitch);
    const eye = { x: c[0] + back[0] * dist, y: c[1] + back[1] * dist, z: c[2] + back[2] * dist, yaw, pitch };
    const v = viewer3d(map, eye, 40);
    const step = Math.pow(10, Math.floor(Math.log10(r))) / 2 || 1, n = Math.ceil((r * 1.6) / step);
    const gx = Math.round(c[0] / step) * step, gz = Math.round(c[2] / step) * step, gy = Math.min(lo[1], 0);
    for (let i = -n; i <= n; i++) {
      v.line([gx + i * step, gy, gz - n * step], [gx + i * step, gy, gz + n * step], "#3b424d", 0.7);
      v.line([gx - n * step, gy, gz + i * step], [gx + n * step, gy, gz + i * step], "#3b424d", 0.7);
    }
    world.turns.forEach((t, i) => {
      for (let j = 1; j < t.path.length; j++) {
        v.line(t.path[j - 1], t.path[j], KIND[t.kind]?.color || "#888", i === cur ? 1 : 0.45, 2.5);
      }
    });
    const p = poseAt(position());
    if (p) { frustum(v, p, "#ffffff", r * 0.35 + 0.4); v.dot([p.x, p.y, p.z], "#ffffff", 3.5); }
    ctx.globalAlpha = 1;
  };

  // the live fly-through while a move is recorded: the demo's proxy scene from the camera's pose
  const drawFly = (p) => {
    const rect = view.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    fly.width = Math.max(1, Math.round(rect.width * dpr)); fly.height = Math.max(1, Math.round(rect.height * dpr));
    const v = viewer3d(fly, p, 55);
    const ctx = v.ctx;
    ctx.globalAlpha = 1; ctx.fillStyle = "rgba(8,10,14,0.62)"; ctx.fillRect(0, 0, fly.width, fly.height);
    for (let i = -40; i <= 40; i++) {
      const major = i % 5 === 0;
      v.line([i, 0, -40], [i, 0, 40], major ? "#c8f259" : "#3b424d", major ? 0.5 : 0.6, dpr, 0.03);
      v.line([-40, 0, i], [40, 0, i], major ? "#c8f259" : "#3b424d", major ? 0.5 : 0.6, dpr, 0.03);
    }
    for (let i = 0; i <= 28; i++) { // the back wall grid makes pitch readable
      const u = -21 + (42 * i) / 28;
      v.line([u, -11, -15], [u, 31, -15], "#303640", 0.5, dpr, 0.02);
      v.line([-21, -11 + (42 * i) / 28, -15], [21, -11 + (42 * i) / 28, -15], "#303640", 0.5, dpr, 0.02);
    }
    for (const [a, b] of ICO) v.line(a, b, "#c8f259", 0.95, 1.5 * dpr);
    ctx.globalAlpha = 1;
  };

  // show turn i from frame `from` in the hidden <video>, then swap, so turn changes don't flash
  let pending = 0;
  const load = (i, from = 0, autoplay = playing) => {
    if (!world || !world.turns[i]) return;
    cur = i;
    const token = ++pending;
    const next = videos[1 - active];
    const src = clipURL(world.turns[i].clip);
    const swap = () => {
      if (token !== pending) return;
      next.currentTime = from / FPS;
      const shown = videos[active];
      next.classList.remove("nx-hidden");
      shown.classList.add("nx-hidden");
      shown.pause();
      active = 1 - active;
      if (autoplay) next.play().catch(() => {});
      paintTimeline(); drawMap();
    };
    if (next.getAttribute("src") === src && next.readyState >= 2) swap();
    else { next.setAttribute("src", src); next.addEventListener("loadeddata", swap, { once: true }); }
  };

  const seek = (frame) => {
    if (!world || !world.turns.length) return;
    frame = Math.max(0, Math.min(frame, world.frames - 1));
    let i = world.turns.findIndex((t) => frame < t.start + t.frames);
    if (i < 0) i = world.turns.length - 1;
    if (i === cur && videos[active].getAttribute("src") === clipURL(world.turns[i].clip)) {
      videos[active].currentTime = (frame - world.turns[i].start) / FPS;
      if (playing) videos[active].play().catch(() => {});
      paintTimeline(); drawMap();
    } else load(i, frame - world.turns[i].start);
  };

  for (const v of videos) {
    v.addEventListener("timeupdate", () => { if (v === videos[active]) { paintTimeline(); drawMap(); } });
    v.addEventListener("ended", () => {
      if (v !== videos[active]) return;
      if (cur < world.turns.length - 1) load(cur + 1, 0, true);
      else { playing = false; paintTimeline(); }
    });
  }

  const setWorld = (state, showFrom) => {
    world = state;
    empty.style.display = world.turns.length ? "none" : "";
    segs.replaceChildren(...world.turns.map((t, i) => {
      const s = el("div", "nx-seg");
      s.style.flex = `${t.frames} 0 0`;
      s.style.background = KIND[t.kind]?.color || "#888";
      s.title = `Turn ${i + 1} · ${clock(t.frames)} · ${t.label}`;
      return s;
    }));
    undo.disabled = !state.can_undo;
    playing = true;
    cur = -1;
    seek(showFrom);
  };

  play.onclick = () => {
    if (!world) return;
    playing = !playing;
    if (playing) {
      if (cur === world.turns.length - 1 && videos[active].ended) seek(0);
      else videos[active].play().catch(() => {});
    } else videos[active].pause();
    paintTimeline();
  };
  live.onclick = () => { if (world) { playing = true; seek(world.turns[world.turns.length - 1].start); } };
  let scrubbing = false;
  const scrubTo = (e) => {
    const r = track.getBoundingClientRect();
    seek(Math.round(((e.clientX - r.left) / r.width) * (world?.frames || 0)));
  };
  track.addEventListener("pointerdown", (e) => { scrubbing = true; track.setPointerCapture(e.pointerId); scrubTo(e); e.stopPropagation(); });
  track.addEventListener("pointermove", (e) => { if (scrubbing) scrubTo(e); });
  track.addEventListener("pointerup", () => { scrubbing = false; });

  // ---- recording ----
  const held = new Set();
  let drag = null, mdx = 0, mdy = 0, recording = null;
  const paintKeys = () => { for (const [k, e] of Object.entries(keyEls)) e.classList.toggle("on", held.has(k)); };
  const say = (text, mode = "") => { status.textContent = text; status.className = `nx-pill nx-status ${mode}`; };
  const lock = (on) => { for (const b of [textTurn, reset, save]) b.disabled = on; undo.disabled = on || !world?.can_undo; };

  const submit = (op, ticks = []) => {
    controls.value = JSON.stringify({ session, op, nonce: Date.now(), ticks });
    busy = true;
    lock(true);
    say({ undo: "Undoing…", save: "Saving the video…", reset: "Starting a new world…" }[op] || "Generating the next turn…", "nx-busy");
    progress.style.width = "0";
    app.queuePrompt(0, 1);
  };

  const stop = () => {
    if (!recording) return;
    clearInterval(recording.timer);
    const ticks = recording.ticks;
    recording = null;
    recBar.style.width = "0";
    fly.classList.remove("nx-on");
    if (ticks.length) submit("turn", ticks);
  };

  const start = () => {
    if (recording || busy) return;
    if (!world) { say("Generate the opening turn first (queue, or New world)"); return; }
    const total = Number(length.value) * FPS;
    // continue from the end of the world, where the camera is
    if (cur !== world.turns.length - 1) seek(world.frames - 1);
    videos[active].pause(); playing = false; paintTimeline();
    const last = world.turns[world.turns.length - 1].path.slice(-1)[0];
    let pose = { x: last[0], y: last[1], z: last[2], yaw: last[3], pitch: last[4] };
    fly.classList.add("nx-on");
    drawFly(pose);
    recording = {
      ticks: [],
      timer: setInterval(() => {
        recording.ticks.push([[...held], mdx, mdy]);
        pose = stepPose(pose, held, mdx, mdy);
        drawFly(pose);
        mdx = 0; mdy = 0;
        recBar.style.width = `${(100 * recording.ticks.length) / total}%`;
        if (recording.ticks.length >= total) stop();
      }, 1000 / FPS),
    };
    say("Recording a move · Space / Enter to finish", "nx-rec");
  };

  view.addEventListener("keydown", (e) => {
    const k = KEYMAP[e.code];
    if (k) { e.preventDefault(); e.stopPropagation(); held.add(k); paintKeys(); start(); }
    else if (e.code === "Space" || e.code === "Enter") { e.preventDefault(); e.stopPropagation(); if (recording) stop(); else play.onclick(); }
  });
  view.addEventListener("keyup", (e) => {
    const k = KEYMAP[e.code];
    if (k) { e.preventDefault(); e.stopPropagation(); held.delete(k); paintKeys(); }
  });
  view.addEventListener("focus", () => view.classList.add("nx-focus"));
  view.addEventListener("blur", () => { view.classList.remove("nx-focus"); held.clear(); paintKeys(); });
  view.addEventListener("contextmenu", (e) => e.preventDefault());
  view.addEventListener("pointerdown", (e) => { view.focus(); drag = [e.clientX, e.clientY]; view.setPointerCapture(e.pointerId); e.stopPropagation(); });
  view.addEventListener("pointermove", (e) => {
    if (!drag) return;
    const dx = e.clientX - drag[0], dy = e.clientY - drag[1];
    drag = [e.clientX, e.clientY];
    if (dx || dy) { mdx += dx; mdy += dy; start(); }
    e.stopPropagation();
  });
  view.addEventListener("pointerup", () => { drag = null; });
  for (const ev of ["keypress", "wheel"]) view.addEventListener(ev, (e) => e.stopPropagation());

  textTurn.onclick = () => {
    if (busy) return;
    if (!(change?.value || "").trim() && !(action?.value || "").trim()) { say("Type a change or an action first"); return; }
    submit("turn");
  };
  undo.onclick = () => { if (!busy) submit("undo"); };
  reset.onclick = () => { if (!busy) submit("reset"); };
  save.onclick = () => { if (!busy) submit("save"); };

  // ---- server events ----
  node.nomasProgress = (value, max) => {
    if (!busy) return;
    progress.style.width = `${(100 * value) / max}%`;
    say(`Generating turn ${(world?.turns.length || 0) + 1} · step ${value}/${max}`, "nx-busy");
  };
  node.nomasShow = (state) => {
    busy = false;
    progress.style.width = "0";
    if (!state) { lock(false); return; }
    if (state.op === "save") {
      world.can_undo = state.can_undo;
      lock(false);
      say(`Saved output/${state.saved}`);
      return;
    }
    const newest = state.turns[state.turns.length - 1];
    setWorld(state, state.op === "undo" ? newest.start : Math.max(0, newest.start - 12));
    lock(false);
    say(`${state.turns.length} turn${state.turns.length > 1 ? "s" : ""} · ${clock(state.frames)} · hold W A S D / arrows or drag for the next move`);
    if (state.op === "turn") { if (change) change.value = ""; if (action) action.value = ""; }
  };
  node.nomasFailed = (message) => {
    busy = false;
    progress.style.width = "0";
    lock(false);
    say(message ? `Failed: ${message}` : "The turn failed — see the console");
  };
}

const nomasNode = (id) => (app.graph?._nodes || []).find((n) => n.type === "NomasExplore" && String(n.id) === String(id));

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
      this.nomasShow?.(message?.nomas_state?.[0]);
    };
  },
  setup() {
    api.addEventListener("progress", (e) => nomasNode(e.detail?.node)?.nomasProgress?.(e.detail.value, e.detail.max));
    api.addEventListener("execution_error", (e) => nomasNode(e.detail?.node_id)?.nomasFailed?.(e.detail?.exception_message));
    api.addEventListener("execution_interrupted", (e) => nomasNode(e.detail?.node_id)?.nomasFailed?.("interrupted"));
  },
});
