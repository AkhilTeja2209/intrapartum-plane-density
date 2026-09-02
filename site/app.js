/* In-browser inference for the intrapartum standard-plane classifier.
 *
 * The preprocessing here must mirror src/datasets.py build_transforms(train=False)
 * exactly, or the demo silently reports numbers the model never produced:
 *
 *     Grayscale(3) -> Resize(short side = round(img_size * 1.14))
 *                  -> CenterCrop(img_size) -> ToTensor -> Normalize(mean, std)
 *
 * Every constant comes from model_meta.json, written by tools/export_onnx.py at
 * export time, so the page cannot drift away from the checkpoint it serves.
 */
"use strict";

const $ = (id) => document.getElementById(id);
const state = { session: null, meta: null, busy: false };

function setStatus(msg, isError = false) {
  const el = $("status");
  el.textContent = msg || "";
  el.classList.toggle("err", !!isError);
}

/* ---------------------------------------------------------------- model --- */

async function init() {
  try {
    setStatus("Loading model metadata…");
    state.meta = await (await fetch("model/model_meta.json")).json();
    renderModelLine(state.meta);

    if (window.ort) {
      ort.env.wasm.wasmPaths =
        "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/";
      ort.env.wasm.numThreads = 1;
    } else {
      throw new Error("onnxruntime-web failed to load");
    }

    setStatus(`Loading model (${state.meta.onnx_mb} MB)… first load only.`);
    state.session = await ort.InferenceSession.create("model/model.onnx", {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
    setStatus("Model ready.");
  } catch (err) {
    console.error(err);
    setStatus(`Could not load the model: ${err.message}`, true);
  }

  loadExamples();
}

function renderModelLine(m) {
  const t = m.test_metrics || {};
  const ci = m.test_macro_f1_ci95;
  const parts = [
    `${m.arch} · trained on "${m.condition}" · seed ${m.seed}`,
    t.macro_f1 != null
      ? `test macro-F1 ${t.macro_f1.toFixed(3)}${
          ci ? ` [${ci[0].toFixed(3)}–${ci[1].toFixed(3)}]` : ""
        }`
      : null,
    t.n != null ? `${t.n.toLocaleString()} held-out frames` : null,
    `threshold ${Number(m.threshold).toFixed(2)}`,
  ].filter(Boolean);
  $("model-line").textContent = parts.join("  ·  ");
}

/* --------------------------------------------------------- preprocessing --- */

/** ITU-R 601-2 luma, the same conversion PIL uses for "L". */
function toGrayscaleInPlace(data) {
  for (let i = 0; i < data.length; i += 4) {
    const l = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
    data[i] = data[i + 1] = data[i + 2] = l;
  }
}

/**
 * Precompute the bilinear resample weights for one axis, using Pillow's
 * ImagingResample geometry.
 *
 * Canvas drawImage() is NOT a substitute. Its scaling filter is
 * implementation-defined, and measured against torchvision on a real test
 * frame it moved the output probability by 0.041 — small, but it is exactly
 * the kind of silent divergence between the reported model and the shipped one
 * that this project keeps finding elsewhere. This reproduces Pillow's geometry
 * to within one 8-bit level on ~0.15% of pixels.
 */
function bilinearWeights(inSize, outSize) {
  const scale = inSize / outSize;
  const filterScale = Math.max(scale, 1.0);
  const support = filterScale;          // bilinear support is 1.0, scaled
  const ss = 1.0 / filterScale;
  const rows = [];
  for (let i = 0; i < outSize; i++) {
    const center = (i + 0.5) * scale;
    const xmin = Math.max(0, Math.trunc(center - support + 0.5));
    const xmax = Math.min(inSize, Math.trunc(center + support + 0.5));
    const w = new Float64Array(Math.max(0, xmax - xmin));
    let sum = 0;
    for (let x = xmin; x < xmax; x++) {
      const t = Math.abs((x - center + 0.5) * ss);
      const v = t < 1 ? 1 - t : 0;
      w[x - xmin] = v;
      sum += v;
    }
    if (sum > 0) for (let k = 0; k < w.length; k++) w[k] /= sum;
    rows.push({ xmin, w });
  }
  return rows;
}

/** One separable pass. Pillow rounds to 8-bit between passes, so we do too. */
function resamplePass(src, inW, inH, outLen, rows, horizontal) {
  const outW = horizontal ? outLen : inW;
  const outH = horizontal ? inH : outLen;
  const dst = new Float64Array(outW * outH);
  for (let y = 0; y < outH; y++) {
    for (let x = 0; x < outW; x++) {
      const r = rows[horizontal ? x : y];
      let acc = 0;
      for (let k = 0; k < r.w.length; k++) {
        acc += r.w[k] * (horizontal
          ? src[y * inW + (r.xmin + k)]
          : src[(r.xmin + k) * inW + x]);
      }
      dst[y * outW + x] = Math.min(255, Math.max(0, Math.round(acc)));
    }
  }
  return { data: dst, w: outW, h: outH };
}

/**
 * Returns { tensor: Float32Array(3*S*S), cropCanvas } for an HTMLImageElement,
 * mirroring build_transforms(train=False):
 *   Grayscale(3) -> Resize(short side = R) -> CenterCrop(S) -> ToTensor -> Normalize
 */
function preprocess(img, meta) {
  const S = meta.img_size;
  const R = meta.resize_short_side;

  // 1. decode at native size and convert to luma first, matching the
  //    torchvision order (Grayscale before Resize).
  const c0 = document.createElement("canvas");
  c0.width = img.naturalWidth;
  c0.height = img.naturalHeight;
  const x0 = c0.getContext("2d", { willReadFrequently: true });
  x0.drawImage(img, 0, 0);
  const nat = x0.getImageData(0, 0, c0.width, c0.height);
  toGrayscaleInPlace(nat.data);

  const W = c0.width, H = c0.height;
  let plane0 = new Float64Array(W * H);
  for (let i = 0, p = 0; i < nat.data.length; i += 4, p++) plane0[p] = nat.data[i];

  // 2. resize so the SHORT side is R -- torchvision Resize(int) semantics
  const scale = R / Math.min(W, H);
  const rw = Math.max(1, Math.round(W * scale));
  const rh = Math.max(1, Math.round(H * scale));
  const hz = resamplePass(plane0, W, H, rw, bilinearWeights(W, rw), true);
  const vt = resamplePass(hz.data, hz.w, hz.h, rh, bilinearWeights(H, rh), false);

  // 3. centre crop S x S
  const left = Math.round((vt.w - S) / 2);
  const top = Math.round((vt.h - S) / 2);

  // 4. ToTensor + Normalize, laid out NCHW; plus a canvas for the preview
  const mean = meta.mean, std = meta.std;
  const out = new Float32Array(3 * S * S);
  const area = S * S;
  const crop = document.createElement("canvas");
  crop.width = crop.height = S;
  const xc = crop.getContext("2d", { willReadFrequently: true });
  const prevImg = xc.createImageData(S, S);

  for (let y = 0; y < S; y++) {
    for (let x = 0; x < S; x++) {
      const sy = Math.min(vt.h - 1, Math.max(0, top + y));
      const sx = Math.min(vt.w - 1, Math.max(0, left + x));
      const g = vt.data[sy * vt.w + sx];
      const p = y * S + x;
      const v = g / 255;
      out[p] = (v - mean[0]) / std[0];
      out[area + p] = (v - mean[1]) / std[1];
      out[2 * area + p] = (v - mean[2]) / std[2];
      const q = p * 4;
      prevImg.data[q] = prevImg.data[q + 1] = prevImg.data[q + 2] = g;
      prevImg.data[q + 3] = 255;
    }
  }
  xc.putImageData(prevImg, 0, 0);
  return { tensor: out, cropCanvas: crop };
}

function softmax2(a, b) {
  const m = Math.max(a, b);
  const ea = Math.exp(a - m), eb = Math.exp(b - m);
  return eb / (ea + eb); // P(class 1) = P(standard plane)
}

/* ------------------------------------------------------------- inference --- */

async function classify(img, truthLabel) {
  if (!state.session) { setStatus("Model is still loading…", true); return; }
  if (state.busy) return;
  state.busy = true;
  setStatus("Running…");

  try {
    const meta = state.meta;
    const S = meta.img_size;
    const { tensor, cropCanvas } = preprocess(img, meta);

    const feeds = { input: new ort.Tensor("float32", tensor, [1, 3, S, S]) };
    const t0 = performance.now();
    const out = await state.session.run(feeds);
    const dt = performance.now() - t0;

    const logits = out[state.session.outputNames[0]].data;
    const p = softmax2(logits[0], logits[1]);
    render(p, cropCanvas, truthLabel, dt);
    setStatus(`Done in ${dt.toFixed(0)} ms, locally.`);
  } catch (err) {
    console.error(err);
    setStatus(`Inference failed: ${err.message}`, true);
  } finally {
    state.busy = false;
  }
}

function render(p, cropCanvas, truthLabel) {
  const thr = Number(state.meta.threshold);
  const isStd = p >= thr;

  const prev = $("preview");
  prev.width = cropCanvas.width;
  prev.height = cropCanvas.height;
  prev.getContext("2d").drawImage(cropCanvas, 0, 0);

  const v = $("verdict");
  v.textContent = isStd ? "Standard plane" : "Not a standard plane";
  v.classList.toggle("pos", isStd);
  v.classList.toggle("neg", !isStd);

  $("prob").textContent = p.toFixed(3);
  $("gauge").style.width = `${(p * 100).toFixed(1)}%`;
  $("gauge-thr").style.left = `${(thr * 100).toFixed(1)}%`;
  $("thr-line").textContent =
    `Decision threshold ${thr.toFixed(2)}, chosen on the validation split and ` +
    `frozen before test.`;

  const truth = $("truth");
  if (truthLabel === 0 || truthLabel === 1) {
    const right = (truthLabel === 1) === isStd;
    truth.textContent = `Ground truth: ${
      truthLabel === 1 ? "standard plane" : "non-standard"
    } — model was ${right ? "correct" : "wrong"} on this frame.`;
    truth.hidden = false;
  } else {
    truth.hidden = true;
  }
  $("result").hidden = false;
}

/* ------------------------------------------------------------------ input -- */

function imageFromBlobOrURL(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("could not decode that image"));
    img.src = src;
  });
}

async function handleFile(file) {
  if (!file) return;
  if (!file.type.startsWith("image/")) {
    setStatus("That is not an image file.", true);
    return;
  }
  const url = URL.createObjectURL(file);
  try {
    const img = await imageFromBlobOrURL(url);
    await classify(img, null);
  } catch (err) {
    setStatus(err.message, true);
  } finally {
    URL.revokeObjectURL(url);
  }
}

function wireInput() {
  const drop = $("drop");
  const file = $("file");

  drop.addEventListener("click", () => file.click());
  drop.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); file.click(); }
  });
  file.addEventListener("change", () => handleFile(file.files[0]));

  ["dragenter", "dragover"].forEach((ev) =>
    drop.addEventListener(ev, (e) => {
      e.preventDefault();
      drop.classList.add("over");
    })
  );
  ["dragleave", "drop"].forEach((ev) =>
    drop.addEventListener(ev, (e) => {
      e.preventDefault();
      drop.classList.remove("over");
    })
  );
  drop.addEventListener("drop", (e) => {
    const f = e.dataTransfer?.files?.[0];
    if (f) handleFile(f);
  });
}

async function loadExamples() {
  try {
    const list = await (await fetch("examples/examples.json")).json();
    const box = $("example-buttons");
    list.forEach((ex, i) => {
      const b = document.createElement("button");
      b.className = "ex";
      b.type = "button";
      b.textContent = `${ex.label_name === "standard" ? "standard" : "non-standard"} #${
        (i % 2) + 1
      }`;
      b.addEventListener("click", async () => {
        try {
          const img = await imageFromBlobOrURL(ex.file);
          await classify(img, ex.label);
        } catch (err) {
          setStatus(err.message, true);
        }
      });
      box.appendChild(b);
    });
  } catch {
    /* examples are optional */
  }
}

wireInput();
init();
