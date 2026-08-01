const $ = (id) => document.getElementById(id);

const fpsEl = $("fps");
const liveEl = $("live");
const zoneEl = $("zone");
const catEl = $("cat");
const detailEl = $("detail");
const stream = $("stream");
const stageFrame = $("stage-frame");
const roiSvg = $("roi-svg");
const roiPoly = $("roi-poly");
const handles = [...document.querySelectorAll(".roi-handle")];
const mmPerPxEl = $("mm-per-px");
const rulerMmEl = $("ruler-mm");
const calibrateBtn = $("calibrate");
const calibStatusEl = $("calib-status");
const calibViewEl = $("calib-view");
const sideStateEl = $("side-state");
const sideLiveEl = $("side-live");
const sideRotBtn = $("side-rot");
const sideScaleBtn = $("side-scale");
const sideRefMmEl = $("side-ref-mm");
const sideStatusEl = $("side-status");
const torchBtns = [...document.querySelectorAll(".torch")];
const torchStatusEl = $("torch-status");
const viewBtns = [...document.querySelectorAll(".view-btn")];

const DEFAULT_QUAD = [
  [0.301, 0.0],
  [0.748, 0.0],
  [0.721, 0.554],
  [0.285, 0.547],
];
const DEFAULT_SIDE_QUAD = [
  [0.06, 0.3],
  [0.94, 0.3],
  [0.94, 0.86],
  [0.06, 0.86],
];

let view = "top";
let roiQuad = DEFAULT_QUAD.map((p) => [...p]);
let sideQuad = DEFAULT_SIDE_QUAD.map((p) => [...p]);
let dragIndex = -1;
let saveTimer = null;
let roiSyncedFromServer = false;
let sideRoiSynced = false;
let frameSize = null;
let sideFrameSize = null;
let sideRot = 0;

const activeQuad = () => (view === "top" ? roiQuad : sideQuad);
const activeFrameSize = () => (view === "top" ? frameSize : sideFrameSize);

function renderResult(latest) {
  if (!latest) {
    zoneEl.dataset.zone = "";
    zoneEl.textContent = "—";
    catEl.textContent = "—";
    detailEl.textContent = "";
    return;
  }
  const [a, b] = latest.size_mm;
  zoneEl.dataset.zone = latest.zone;
  zoneEl.textContent = latest.pending ? `…${latest.zone}` : latest.zone;
  catEl.textContent = latest.pending ? "анализ…" : latest.category;
  const dims = latest.height_mm ? `${a}×${b}×${latest.height_mm} mm` : `${a}×${b} mm`;
  const kParts = [`K<sub>xy</sub> ${(latest.k_xy ?? 0).toFixed(2)}`];
  if (latest.k_side != null) kParts.push(`K<sub>side</sub> ${latest.k_side.toFixed(2)}`);
  detailEl.innerHTML = latest.pending ? dims : `${dims} · ${kParts.join(" · ")}`;
}

function displayMapping() {
  const fs = activeFrameSize();
  const nw = (fs && fs[0]) || stream.naturalWidth || 1280;
  const nh = (fs && fs[1]) || stream.naturalHeight || 720;
  const cw = stageFrame.clientWidth || 1;
  const ch = stageFrame.clientHeight || 1;
  const scale = Math.min(cw / nw, ch / nh);
  return {
    nw,
    nh,
    cw,
    ch,
    scale,
    ox: (cw - nw * scale) / 2,
    oy: (ch - nh * scale) / 2,
  };
}

function syncSvgViewBox() {
  const { cw, ch } = displayMapping();
  roiSvg.setAttribute("viewBox", `0 0 ${cw} ${ch}`);
}

function normToSvg(nx, ny) {
  const m = displayMapping();
  return [m.ox + nx * m.nw * m.scale, m.oy + ny * m.nh * m.scale];
}

function svgToNorm(sx, sy) {
  const m = displayMapping();
  return [
    Math.min(1, Math.max(0, (sx - m.ox) / m.scale / m.nw)),
    Math.min(1, Math.max(0, (sy - m.oy) / m.scale / m.nh)),
  ];
}

function pointerToSvg(e) {
  const pt = roiSvg.createSVGPoint();
  pt.x = e.clientX;
  pt.y = e.clientY;
  const ctm = roiSvg.getScreenCTM();
  if (!ctm) return [0, 0];
  const p = pt.matrixTransform(ctm.inverse());
  return [p.x, p.y];
}

function renderRoi() {
  syncSvgViewBox();
  const m = displayMapping();
  const pts = activeQuad().map(([x, y]) => normToSvg(x, y));
  roiPoly.setAttribute("points", pts.map(([x, y]) => `${x},${y}`).join(" "));
  const r = Math.max(10, Math.min(m.cw, m.ch) * 0.018);
  handles.forEach((h, i) => {
    const [x, y] = pts[i];
    h.setAttribute("cx", String(x));
    h.setAttribute("cy", String(y));
    h.setAttribute("r", String(r));
  });
}

function scheduleSaveRoi() {
  clearTimeout(saveTimer);
  const isTop = view === "top";
  const url = isTop ? "/api/config" : "/api/side_config";
  const body = isTop ? { roi_quad: roiQuad } : { side_roi_quad: sideQuad };
  saveTimer = setTimeout(async () => {
    try {
      await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
    } catch {
      /* ignore */
    }
  }, 120);
}

function setView(next) {
  if (next === view) return;
  view = next;
  viewBtns.forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.view === view)));
  calibViewEl.textContent = view === "top" ? "" : " · сбоку";
  stream.src = view === "top" ? "/video" : "/side_video";
  renderRoi();
}

function onPointerDown(e) {
  const handle = e.target.closest(".roi-handle");
  if (!handle) return;
  e.preventDefault();
  dragIndex = Number(handle.dataset.i);
  handle.setPointerCapture?.(e.pointerId);
}

function onPointerMove(e) {
  if (dragIndex < 0) return;
  e.preventDefault();
  const [sx, sy] = pointerToSvg(e);
  activeQuad()[dragIndex] = svgToNorm(sx, sy);
  renderRoi();
  scheduleSaveRoi();
}

function onPointerUp() {
  dragIndex = -1;
}

async function poll() {
  try {
    const res = await fetch("/api/status", { cache: "no-store" });
    const data = await res.json();
    fpsEl.textContent = `${data.fps ?? "—"} fps`;
    const live = data.running && !data.black_frames;
    liveEl.dataset.state = live ? "on" : "off";
    liveEl.textContent = data.black_frames
      ? "black"
      : data.running
        ? "live"
        : data.error
          ? "err"
          : "off";

    if (Array.isArray(data.frame_size) && data.frame_size.length === 2) {
      const w = Number(data.frame_size[0]);
      const h = Number(data.frame_size[1]);
      if (w > 1 && h > 1) {
        const changed = !frameSize || frameSize[0] !== w || frameSize[1] !== h;
        frameSize = [w, h];
        if (changed) renderRoi();
      }
    }
    if (!roiSyncedFromServer && Array.isArray(data.roi_quad) && data.roi_quad.length === 4) {
      roiQuad = data.roi_quad.map((p) => [Number(p[0]), Number(p[1])]);
      roiSyncedFromServer = true;
      renderRoi();
    }
    const side = data.side || {};
    if (Array.isArray(side.frame_size) && side.frame_size.length === 2) {
      const w = Number(side.frame_size[0]);
      const h = Number(side.frame_size[1]);
      if (w > 1 && h > 1) {
        const changed = !sideFrameSize || sideFrameSize[0] !== w || sideFrameSize[1] !== h;
        sideFrameSize = [w, h];
        if (changed && view === "side") renderRoi();
      }
    }
    if (!sideRoiSynced && Array.isArray(side.roi_quad) && side.roi_quad.length === 4) {
      sideQuad = side.roi_quad.map((p) => [Number(p[0]), Number(p[1])]);
      sideRoiSynced = true;
      if (view === "side") renderRoi();
    }
    if (typeof side.rot === "number" && side.rot !== sideRot) {
      sideRot = side.rot;
      sideRotBtn.textContent = `поворот ${sideRot}°`;
    }
    if (!side.enabled) {
      sideStateEl.textContent = "камера сбоку не подключена";
      sideLiveEl.textContent = "K проверяется только по XY — лежащий цилиндр не виден";
    } else {
      sideStateEl.textContent = `сбоку: ${side.fps ?? 0} fps · ${Number(
        side.mm_per_px || 0
      ).toFixed(3)} mm/px`;
      const m = side.measure;
      sideLiveEl.textContent =
        side.fresh && m
          ? `H ${m.height_mm} mm · K_side ${m.k_side}`
          : side.error || "объект не виден";
    }
    renderTorch(data.torch, side.enabled);
    const scale = view === "top" ? data.mm_per_px : side.mm_per_px;
    if (typeof scale === "number") mmPerPxEl.textContent = scale.toFixed(4);
    renderResult(data.latest);
  } catch {
    liveEl.dataset.state = "off";
    liveEl.textContent = "off";
  }
}

function renderTorch(torch, sideEnabled) {
  if (!torch) return;
  const notes = [];
  for (const btn of torchBtns) {
    const cam = btn.dataset.cam;
    const t = torch[cam] || {};
    btn.dataset.on = t.want ? "1" : "0";
    btn.dataset.off = cam === "side" && !sideEnabled ? "1" : "0";
    if (t.want && t.fresh && !t.on) notes.push(`${cam}: Safari не пустил фонарик`);
    else if (t.want && !t.fresh) notes.push(`${cam}: страница телефона закрыта`);
  }
  torchStatusEl.dataset.ok = notes.length ? "0" : "";
  torchStatusEl.textContent = notes.join(" · ");
}

async function toggleTorch(btn) {
  const on = btn.dataset.on !== "1";
  btn.dataset.on = on ? "1" : "0";
  try {
    await fetch("/api/torch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ cam: btn.dataset.cam, on }),
    });
  } catch {
    torchStatusEl.dataset.ok = "0";
    torchStatusEl.textContent = "нет связи";
  }
}

async function runCalibrate() {
  const rulerMm = Number(rulerMmEl.value) || 200;
  calibrateBtn.disabled = true;
  calibStatusEl.dataset.ok = "";
  calibStatusEl.textContent = "ищу шкалу 0→конец…";
  try {
    const res = await fetch(view === "top" ? "/api/calibrate" : "/api/side_calibrate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ruler_mm: rulerMm }),
    });
    const data = await res.json();
    if (!res.ok || !data.ok) {
      calibStatusEl.dataset.ok = "0";
      calibStatusEl.textContent = data.error || "ошибка";
      return;
    }
    data.mm_per_px = data.mm_per_px ?? data.side_mm_per_px;
    mmPerPxEl.textContent = Number(data.mm_per_px).toFixed(4);
    calibStatusEl.dataset.ok = "1";
    const via = data.method === "scale" ? `шкала · ${data.ticks} совп.` : "корпус−поля";
    calibStatusEl.textContent = `${data.mm_per_px} mm/px · ${via} · ${data.scale_px}px = ${data.ruler_mm}mm`;
  } catch {
    calibStatusEl.dataset.ok = "0";
    calibStatusEl.textContent = "нет связи";
  } finally {
    calibrateBtn.disabled = false;
  }
}

$("roi-reset").addEventListener("click", () => {
  if (view === "top") roiQuad = DEFAULT_QUAD.map((p) => [...p]);
  else sideQuad = DEFAULT_SIDE_QUAD.map((p) => [...p]);
  renderRoi();
  scheduleSaveRoi();
});

viewBtns.forEach((b) => b.addEventListener("click", () => setView(b.dataset.view)));

for (const btn of torchBtns) {
  btn.addEventListener("click", () => toggleTorch(btn));
}

sideScaleBtn.addEventListener("click", async () => {
  const refMm = Number(sideRefMmEl.value) || 200;
  sideScaleBtn.disabled = true;
  sideStatusEl.dataset.ok = "";
  sideStatusEl.textContent = "меряю эталон…";
  try {
    const res = await fetch("/api/side_scale", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ height_mm: refMm }),
    });
    const data = await res.json();
    if (!res.ok || !data.ok) {
      sideStatusEl.dataset.ok = "0";
      sideStatusEl.textContent = data.error || "ошибка";
      return;
    }
    sideStatusEl.dataset.ok = "1";
    sideStatusEl.textContent = `${data.side_mm_per_px} mm/px · ${data.height_mm}mm = ${data.height_px}px`;
  } catch {
    sideStatusEl.dataset.ok = "0";
    sideStatusEl.textContent = "нет связи";
  } finally {
    sideScaleBtn.disabled = false;
  }
});

sideRotBtn.addEventListener("click", async () => {
  sideRot = (sideRot + 90) % 360;
  sideRotBtn.textContent = `поворот ${sideRot}°`;
  if (view !== "side") setView("side");
  try {
    await fetch("/api/side_config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ side_rot: sideRot }),
    });
  } catch {
    /* ignore */
  }
});

calibrateBtn.addEventListener("click", () => {
  runCalibrate();
});

roiSvg.addEventListener("pointerdown", onPointerDown);
window.addEventListener("pointermove", onPointerMove);
window.addEventListener("pointerup", onPointerUp);
window.addEventListener("pointercancel", onPointerUp);
window.addEventListener("resize", renderRoi);
stream.addEventListener("load", renderRoi);

renderRoi();
poll();
setInterval(poll, 400);
