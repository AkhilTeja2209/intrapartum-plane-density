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
const state = { session: null, meta: null, busy: false, loadError: null,
                showCam: true, lastCam: null, lastCrop: null,
                // stage two: the segmenter is a second model and a second
                // download, so it is fetched on request and then reused
                segSession: null, segMeta: null, segLoading: null,
                showSeg: true, lastSeg: null, lastSegBase: null,
                lastImg: null, lastRef: null, lastRefProb: null,
                lastIsStd: null };

function setStatus(msg, isError = false) {
  const el = $("status");
  el.textContent = msg || "";
  el.classList.toggle("err", !!isError);
}

/* ---------------------------------------------------------------- model --- */

/* The runtime is served from this origin. It used to be pulled from a public
 * CDN, which put a third party on the critical path of a page whose selling
 * point is that nothing leaves the machine: a visitor whose network, DNS or
 * extensions did not reach that host saw "Failed to fetch" and had no way to
 * tell which of three downloads had died. vendor/onnxruntime-web/ holds the
 * 1.19.2 wasm build -- see the README there for provenance and hashes.
 *
 * Absolute, not relative: onnxruntime-web passes this through to a dynamic
 * import() of its .mjs loader, and "vendor/..." is not a resolvable module
 * specifier -- only "./vendor/..." or a full URL is. A bare relative path
 * loads the graph and then fails at instantiation with "no available
 * backend found". */
const RUNTIME_DIR = new URL("vendor/onnxruntime-web/", document.baseURI).href;

/** Fetch bytes with progress, failing with a message that names the file.
 *  The Fetch API reports every network failure as the bare string "Failed to
 *  fetch", which is unactionable on a page that pulls a 44.7 MB graph, a
 *  57.3 MB graph and a WebAssembly runtime: it does not even say which. */
async function fetchBytes(url, label, onProgress) {
  let res;
  try {
    res = await fetch(url);
  } catch (err) {
    throw new Error(`${label} (${url}) could not be fetched: ${err.message}`);
  }
  if (!res.ok) throw new Error(`${label} (${url}) returned HTTP ${res.status}`);
  if (!res.body) return new Uint8Array(await res.arrayBuffer());

  // 1e6, not 1<<20: model_meta.json's onnx_mb is decimal, and a progress line
  // counting up to 42.6 MB for a download the page calls 44.7 MB reads like a
  // bug in whichever number the visitor happens to trust.
  const mb = (n) => (n / 1e6).toFixed(1);
  const total = Number(res.headers.get("content-length")) || 0;
  const reader = res.body.getReader();
  const chunks = [];
  let got = 0;

  for (;;) {
    let piece;
    try {
      piece = await reader.read();
    } catch (err) {
      // A download cut off part-way is the failure this page is most exposed
      // to, so say how far it got rather than reporting it like a dead host.
      throw new Error(
        `${label} (${url}) stopped after ${mb(got)} MB` +
        (total ? ` of ${mb(total)} MB` : "") + `: ${err.message}`);
    }
    if (piece.done) break;
    chunks.push(piece.value);
    got += piece.value.length;
    if (onProgress) onProgress(got, total);
  }
  // No completeness check against content-length: the server compresses, so
  // the header is the size on the wire and `got` is the size after decoding.
  // Comparing them failed every fetch on GitHub Pages while passing against a
  // local server that does not compress. A transfer cut short raises at the
  // reader above, and a graph that arrives damaged fails to parse in
  // sessionFrom, so nothing is lost by dropping it.
  const bytes = new Uint8Array(got);
  let at = 0;
  for (const c of chunks) { bytes.set(c, at); at += c.length; }
  return bytes;
}

async function fetchJson(url, label) {
  return JSON.parse(new TextDecoder().decode(await fetchBytes(url, label)));
}

/** A percentage, so that a slow link looks slow rather than broken. The
 *  classifier is 44.7 MB and the segmenter 57.3 MB; on a 2 MB/s connection
 *  that is half a minute of a status line that used to say nothing.
 *
 *  `expectMb` comes from the export metadata, not from content-length: the
 *  server gzips, so the header counts bytes on the wire (41.5 MB for the
 *  classifier) while the reader counts bytes after decoding (44.7 MB), and a
 *  bar fed the header runs past 100%. */
function progressStatus(what, expectMb) {
  const expect = expectMb ? expectMb * 1e6 : 0;
  return (got, total) => {
    const mb = (got / 1e6).toFixed(1);
    const end = expect || total;
    setStatus(end
      ? `${what}: ${mb} of ${(end / 1e6).toFixed(1)} MB ` +
        `(${Math.min(100, Math.round(100 * got / end))}%)…`
      : `${what}: ${mb} MB…`);
  };
}

/** Build a session from bytes already in hand, so that a network failure is
 *  attributed to the graph and a compile failure to the runtime. Letting
 *  InferenceSession.create fetch the URL itself conflates the two. */
async function sessionFrom(bytes, label) {
  try {
    return await ort.InferenceSession.create(bytes, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
  } catch (err) {
    throw new Error(`${label} downloaded, but the WebAssembly runtime in ` +
                    `${RUNTIME_DIR} could not start it: ${err.message}`);
  }
}

async function init() {
  const retry = $("retry");
  if (retry) retry.hidden = true;
  state.loadError = null;

  try {
    setStatus("Loading model metadata…");
    state.meta = await fetchJson("model/model_meta.json",
                                 "the model metadata");
    renderModelLine(state.meta);
    // Start this now rather than after the 44.7 MB classifier has downloaded:
    // it is a few kB, it is independent, and it is what tells the visitor that
    // a second stage exists and what it would cost them.
    loadSegMeta();

    if (!window.ort) {
      throw new Error(
        `the inference runtime (${RUNTIME_DIR}ort.wasm.min.js) did not load`);
    }
    ort.env.wasm.wasmPaths = RUNTIME_DIR;
    ort.env.wasm.numThreads = 1;

    const label = `the ${state.meta.onnx_mb} MB classifier`;
    setStatus(`Loading model (${state.meta.onnx_mb} MB)… first load only.`);
    const bytes = await fetchBytes(
      "model/model.onnx", label,
      progressStatus("Loading model", state.meta.onnx_mb));
    state.session = await sessionFrom(bytes, label);
    setStatus("Model ready.");
  } catch (err) {
    console.error(err);
    // Before the retry button this was a dead end: the only way out of a
    // dropped download was for the visitor to work out that reloading the
    // page by hand would start it again.
    state.loadError = err.message;
    setStatus(`Could not load the model: ${err.message}`, true);
    if (retry) retry.hidden = false;
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
    // quantise once, here, exactly as Pillow's normalize_coeffs_8bpc does
    const k = new Float64Array(w.length);
    for (let j = 0; j < w.length; j++) k[j] = Math.round(w[j] * PRECISION_SCALE);
    rows.push({ xmin, w, k });
  }
  return rows;
}

/* Pillow does not accumulate 8-bit resampling in floating point. It quantises
 * each normalised weight to a 22-bit fixed-point integer, sums in integers
 * with a half-LSB offset, and shifts back down:
 *
 *     k[i]  = round(w[i] * (1 << 22))
 *     acc   = (1 << 21) + sum(pixel[i] * k[i])
 *     out   = clip8(acc >> 22)
 *
 * Rounding the float sum instead is not equivalent, and the difference is not
 * academic. On a 256 -> 384 upscale a third of the output pixels land on an
 * exact half-way blend of two neighbours, so float rounding and fixed-point
 * rounding disagree on **14.5% of pixels** by up to two 8-bit levels, which
 * moved the segmenter's area score by 2.6e-3 relative against PyTorch.
 * Reproducing the fixed-point path matches Pillow on every one of the 147,456
 * pixels. A downscale hides this -- ties are rare there -- which is why the
 * classifier path looked fine with float accumulation.
 */
const PRECISION_BITS = 22;
const PRECISION_SCALE = 1 << PRECISION_BITS;      // 4194304

/** One separable pass. Pillow rounds to 8-bit between passes, so we do too. */
function resamplePass(src, inW, inH, outLen, rows, horizontal) {
  const outW = horizontal ? outLen : inW;
  const outH = horizontal ? inH : outLen;
  const dst = new Float64Array(outW * outH);
  const half = PRECISION_SCALE / 2;
  for (let y = 0; y < outH; y++) {
    for (let x = 0; x < outW; x++) {
      const r = rows[horizontal ? x : y];
      // r.k holds the already-quantised weights; the products reach ~3e9,
      // which overflows int32 but is exact in a double, so accumulate there.
      let acc = half;
      for (let k = 0; k < r.k.length; k++) {
        acc += r.k[k] * (horizontal
          ? src[y * inW + (r.xmin + k)]
          : src[(r.xmin + k) * inW + x]);
      }
      const v = Math.floor(acc / PRECISION_SCALE);
      dst[y * outW + x] = v < 0 ? 0 : v > 255 ? 255 : v;
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


/* ------------------------------------------------------------ Grad-CAM --- */

/**
 * Class activation map from the exported feature map.
 *
 * For a head of global-average-pool -> linear, Grad-CAM reduces exactly to
 * CAM: the logit is a weighted sum of channel means, so the gradient of the
 * logit with respect to a feature cell is the classifier weight divided by the
 * spatial area -- constant over space. The Grad-CAM channel weight is then the
 * classifier weight up to a positive scale that normalisation removes.
 *
 * That is what makes this possible in a browser. ONNX Runtime Web has no
 * backward pass, but none is needed: the map computed here matches PyTorch
 * Grad-CAM to a correlation of 1.000000 on the bundled examples.
 */
function computeCAM(feat, dims, weights, cls) {
  const [, C, H, W] = dims;
  const w = weights[cls];
  const cam = new Float32Array(H * W);
  for (let k = 0; k < C; k++) {
    const wk = w[k];
    if (wk === 0) continue;
    const base = k * H * W;
    for (let i = 0; i < H * W; i++) cam[i] += wk * feat[base + i];
  }
  let mn = Infinity, mx = -Infinity;
  for (let i = 0; i < cam.length; i++) {
    if (cam[i] < 0) cam[i] = 0;                 // ReLU, as in Grad-CAM
    if (cam[i] < mn) mn = cam[i];
    if (cam[i] > mx) mx = cam[i];
  }
  const span = mx - mn || 1;
  for (let i = 0; i < cam.length; i++) cam[i] = (cam[i] - mn) / span;
  return { data: cam, h: H, w: W };
}

/** Bilinear upsample of the 7x7 map to the displayed frame. */
function upsampleCAM(cam, S) {
  const { data, h, w } = cam;
  const out = new Float32Array(S * S);
  for (let y = 0; y < S; y++) {
    const fy = Math.min(h - 1, Math.max(0, (y + 0.5) * h / S - 0.5));
    const y0 = Math.floor(fy), y1 = Math.min(h - 1, y0 + 1), ty = fy - y0;
    for (let x = 0; x < S; x++) {
      const fx = Math.min(w - 1, Math.max(0, (x + 0.5) * w / S - 0.5));
      const x0 = Math.floor(fx), x1 = Math.min(w - 1, x0 + 1), tx = fx - x0;
      const a = data[y0 * w + x0], b = data[y0 * w + x1];
      const c = data[y1 * w + x0], d = data[y1 * w + x1];
      out[y * S + x] = (a * (1 - tx) + b * tx) * (1 - ty) +
                       (c * (1 - tx) + d * tx) * ty;
    }
  }
  return out;
}

/** Perceptually ordered blue -> cyan -> yellow -> red, like matplotlib jet. */
function heatColour(v) {
  const stops = [[0, 0, 0, 140], [0.25, 0, 190, 220], [0.5, 90, 215, 90],
                 [0.75, 255, 200, 40], [1, 225, 40, 30]];
  for (let i = 1; i < stops.length; i++) {
    if (v <= stops[i][0]) {
      const [p0, r0, g0, b0] = stops[i - 1], [p1, r1, g1, b1] = stops[i];
      const t = (v - p0) / (p1 - p0 || 1);
      return [r0 + (r1 - r0) * t, g0 + (g1 - g0) * t, b0 + (b1 - b0) * t];
    }
  }
  return [225, 40, 30];
}

/** Paint the frame with the CAM blended over it, weighted by activation. */
function paintOverlay(canvas, cropCanvas, camUp, alphaMax = 0.62) {
  const S = canvas.width;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(cropCanvas, 0, 0);
  if (!camUp) return;
  const img = ctx.getImageData(0, 0, S, S);
  const d = img.data;
  for (let i = 0, p = 0; i < d.length; i += 4, p++) {
    const v = camUp[p];
    // fade the overlay out where the model is not attending, so the
    // underlying anatomy stays readable instead of being tinted everywhere
    const a = alphaMax * Math.pow(v, 1.5);
    const [r, g, b] = heatColour(v);
    d[i] = d[i] * (1 - a) + r * a;
    d[i + 1] = d[i + 1] * (1 - a) + g * a;
    d[i + 2] = d[i + 2] * (1 - a) + b * a;
  }
  ctx.putImageData(img, 0, 0);
}

/* ------------------------------------------------- stage two: the U-Net --- */

/**
 * The segmenter's preprocessing is NOT the classifier's.
 *
 *   classifier   Grayscale -> Resize(short side 255) -> CenterCrop(224)
 *   segmenter    Grayscale -> Resize(S, S)            (no crop at all)
 *
 * and the segmenter normalises with the single-channel grey statistics rather
 * than the three ImageNet ones. Reusing the classifier's crop here would
 * produce masks that look plausible and sit in the wrong place, so the two
 * pipelines are kept separate and every constant comes from unet_meta.json.
 */
function preprocessSeg(img, sm) {
  const S = sm.img_size;
  const { mean, std } = sm.preprocess;

  const c0 = document.createElement("canvas");
  c0.width = img.naturalWidth;
  c0.height = img.naturalHeight;
  const x0 = c0.getContext("2d", { willReadFrequently: true });
  x0.drawImage(img, 0, 0);
  const nat = x0.getImageData(0, 0, c0.width, c0.height);
  toGrayscaleInPlace(nat.data);

  const W = c0.width, H = c0.height;
  const plane = new Float64Array(W * H);
  for (let i = 0, p = 0; i < nat.data.length; i += 4, p++) plane[p] = nat.data[i];

  // square resize, both axes independently -- Pillow's own two-pass geometry
  const hz = resamplePass(plane, W, H, S, bilinearWeights(W, S), true);
  const vt = resamplePass(hz.data, hz.w, hz.h, S, bilinearWeights(H, S), false);

  const tensor = new Float32Array(S * S);
  const base = document.createElement("canvas");
  base.width = base.height = S;
  const xc = base.getContext("2d", { willReadFrequently: true });
  const bi = xc.createImageData(S, S);
  for (let p = 0; p < S * S; p++) {
    const g = vt.data[p];
    tensor[p] = (g / 255 - mean) / std;
    const q = p * 4;
    bi.data[q] = bi.data[q + 1] = bi.data[q + 2] = g;
    bi.data[q + 3] = 255;
  }
  xc.putImageData(bi, 0, 0);
  return { tensor, baseCanvas: base };
}

/** Per-pixel softmax over the three classes, plus argmax and soft areas. */
function readSegLogits(logits, S) {
  const area = S * S;
  const cls = new Uint8Array(area);
  let sumPS = 0, sumFH = 0;
  for (let p = 0; p < area; p++) {
    const a = logits[p], b = logits[area + p], c = logits[2 * area + p];
    const m = Math.max(a, b, c);
    const ea = Math.exp(a - m), eb = Math.exp(b - m), ec = Math.exp(c - m);
    const z = ea + eb + ec;
    const pa = ea / z, pb = eb / z, pc = ec / z;
    sumPS += pb;
    sumFH += pc;
    cls[p] = pa >= pb && pa >= pc ? 0 : (pb >= pc ? 1 : 2);
  }
  // Soft areas, i.e. mean probability per class -- the same quantity
  // src/train_unet.py scores frames with, so the number shown here is the
  // number the decision was made on.
  return { cls, ps: sumPS / area, fh: sumFH / area, S };
}

/** The aggregation rule chosen on the validation split, not on test. */
function planeScore(seg, rule) {
  switch (rule) {
    case "geomean": return Math.sqrt(Math.max(0, seg.ps * seg.fh));
    case "ps": return seg.ps;
    case "fh": return seg.fh;
    case "sum": return seg.ps + seg.fh;
    default: return Math.min(seg.ps, seg.fh);
  }
}

/** Translucent fill plus a hard boundary, so small structures stay visible. */
function paintSeg(canvas, baseCanvas, seg, colours, show) {
  const S = canvas.width;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(baseCanvas, 0, 0);
  if (!seg || !show) return;

  const col = [null, colours.pubic_symphysis, colours.fetal_head];
  const img = ctx.getImageData(0, 0, S, S);
  const d = img.data;
  const { cls } = seg;
  for (let y = 0; y < S; y++) {
    for (let x = 0; x < S; x++) {
      const p = y * S + x;
      const c = cls[p];
      if (!c) continue;
      // boundary if any 4-neighbour belongs to a different class
      const edge =
        (x === 0 || cls[p - 1] !== c) || (x === S - 1 || cls[p + 1] !== c) ||
        (y === 0 || cls[p - S] !== c) || (y === S - 1 || cls[p + S] !== c);
      const a = edge ? 1.0 : 0.3;
      const [r, g, b] = col[c];
      const q = p * 4;
      d[q] = d[q] * (1 - a) + r * a;
      d[q + 1] = d[q + 1] * (1 - a) + g * a;
      d[q + 2] = d[q + 2] * (1 - a) + b * a;
    }
  }
  ctx.putImageData(img, 0, 0);
}

/** Metadata only: a few kB, so the header can describe stage two before the
 *  visitor decides whether to download it. */
async function loadSegMeta() {
  try {
    state.segMeta = await fetchJson("model/unet_meta.json",
                                    "the segmenter metadata");
    renderSegModelLine(state.segMeta);
    const b = $("seg-run");
    if (b) b.textContent =
      `Outline the anatomy — fetches a ${state.segMeta.onnx_mb} MB model`;
  } catch {
    /* stage two is optional; stage one still works without it */
  }
}

async function ensureSegSession() {
  if (state.segSession) return state.segSession;
  if (state.segLoading) return state.segLoading;

  state.segLoading = (async () => {
    if (!state.segMeta) await loadSegMeta();
    const label = `the ${state.segMeta.onnx_mb} MB segmenter`;
    setStatus(`Fetching the segmentation model (${state.segMeta.onnx_mb} MB)…`);
    const bytes = await fetchBytes(
      "model/unet.onnx", label,
      progressStatus("Fetching the segmenter", state.segMeta.onnx_mb));
    state.segSession = await sessionFrom(bytes, label);
    return state.segSession;
  })();
  try {
    return await state.segLoading;
  } finally {
    state.segLoading = null;
  }
}

function renderSegModelLine(sm) {
  const d = sm.test_dice || {};
  const m = sm.derived_metrics || {};
  const parts = [
    `stage 2: ${sm.arch} U-Net · ${sm.img_size}px`,
    d.mean_foreground != null
      ? `test Dice ${d.mean_foreground.toFixed(3)} (PS ${
          d.pubic_symphysis.toFixed(3)}, FH ${d.fetal_head.toFixed(3)})`
      : null,
    d.n_masks != null ? `${d.n_masks} held-out masks` : null,
    // Labelled "unreliable" deliberately. Across eleven arms this rule scores
    // 0.530 +/- 0.039 and its choice flips between seeds, so presenting the
    // number bare invites reading it as a second accuracy figure.
    m.macro_f1 != null
      ? `area-derived macro-F1 ${m.macro_f1.toFixed(3)} (unreliable)` : null,
    sm.quantised ? `${sm.onnx_mb} MB int8` : `${sm.onnx_mb} MB`,
  ].filter(Boolean);
  const el = $("seg-model-line");
  if (el) el.textContent = parts.join("  ·  ");
}

async function segment(img) {
  const sess = await ensureSegSession();
  const sm = state.segMeta;
  const S = sm.img_size;
  const { tensor, baseCanvas } = preprocessSeg(img, sm);

  const t0 = performance.now();
  const out = await sess.run({
    input: new ort.Tensor("float32", tensor, [1, 1, S, S]),
  });
  const dt = performance.now() - t0;

  const seg = readSegLogits(out[sess.outputNames[0]].data, S);
  state.lastSeg = seg;
  state.lastSegBase = baseCanvas;
  renderSeg(seg, dt);
  return seg;
}

function renderSeg(seg, dt) {
  const sm = state.segMeta;
  const cv = $("segcanvas");
  cv.width = cv.height = seg.S;
  paintSeg(cv, state.lastSegBase, seg, sm.colours, state.showSeg);

  const pct = (v) => `${(100 * v).toFixed(2)}% of frame`;
  $("area-ps").textContent = pct(seg.ps);
  $("area-fh").textContent = pct(seg.fh);

  const m0 = sm.derived_metrics || {};
  const rule = sm.derived_rule || "min";
  const score = planeScore(seg, rule);
  const thr = Number(sm.derived_threshold);
  const isStd = score >= thr;
  const v = $("seg-verdict");
  v.textContent = isStd
    ? "Enough of both structures to measure"
    : "Not enough of both structures";
  v.classList.toggle("pos", isStd);
  v.classList.toggle("neg", !isStd);

  const ruleText = { min: "the smaller of the two areas",
                     geomean: "the geometric mean of the two areas",
                     ps: "the symphysis area alone",
                     fh: "the head area alone",
                     sum: "the two areas added" }[rule] || rule;
  const f1Text = m0.macro_f1 != null
    ? ` On held-out frames this rule scores only ${m0.macro_f1.toFixed(3)} ` +
      `macro-F1, so treat it as a readout of the drawing, not as a verdict.`
    : "";
  $("seg-note").textContent =
    `Made only out of what was drawn: ${ruleText} is ` +
    `${score.toExponential(2)} against a threshold of ` +
    `${thr.toExponential(2)} picked on the validation split.${f1Text}`;

  // The two stages are trained on different supervision and can disagree.
  // Saying so is the point of showing both: the cascade's error is the sum of
  // its stages', and a visitor should see where that comes from.
  const m = sm.derived_metrics || {};
  const disagree = state.lastIsStd != null && state.lastIsStd !== isStd;
  $("seg-note").textContent += disagree
    ? ` The two stages disagree on this frame — stage one says ${
        state.lastIsStd ? "standard" : "not standard"} and stage two says ${
        isStd ? "standard" : "not standard"}.`
    : "";
  $("seg-caveat").textContent = m.recall_standard != null
    ? `Read it with its error profile in mind: every mask in the training set ` +
      `is a standard plane, so the segmenter has never been shown a negative ` +
      `and draws anatomy on almost every frame — recall ` +
      `${m.recall_standard.toFixed(2)} against specificity ` +
      `${Number(m.specificity).toFixed(2)} on ` +
      `${Number(m.n).toLocaleString()} held-out frames, meaning it calls ` +
      `${(100 * (1 - Number(m.specificity))).toFixed(0)}% of non-standard ` +
      `frames standard.`
    : "";

  $("seg-out").hidden = false;
  if (dt != null) setStatus(`Segmented in ${dt.toFixed(0)} ms, locally.`);

  // The export records what PyTorch produced for each bundled example, so the
  // browser pipeline can be checked rather than trusted.
  const ref = state.lastRef && sm.example_refs
    ? sm.example_refs[state.lastRef] : null;
  if (ref) {
    console.log("segmentation vs PyTorch reference:",
      { ps: [seg.ps, ref.areas.pubic_symphysis],
        fh: [seg.fh, ref.areas.fetal_head],
        score: [score, ref.score],
        maxRelDiff: Math.max(
          Math.abs(seg.ps - ref.areas.pubic_symphysis) / ref.areas.pubic_symphysis,
          Math.abs(seg.fh - ref.areas.fetal_head) / ref.areas.fetal_head) });
  }
}

function wireSeg() {
  const run = $("seg-run");
  if (run) {
    run.addEventListener("click", async () => {
      if (!state.lastImg) { setStatus("Load a frame first.", true); return; }
      run.disabled = true;
      try {
        await segment(state.lastImg);
        $("seg-run").hidden = true;
        $("seg-intro").hidden = true;
      } catch (err) {
        console.error(err);
        setStatus(`Segmentation failed: ${err.message}`, true);
      } finally {
        run.disabled = false;
      }
    });
  }

  const tog = $("seg-toggle");
  if (tog) {
    tog.addEventListener("click", () => {
      state.showSeg = !state.showSeg;
      tog.textContent = state.showSeg ? "Hide the outline" : "Show the outline";
      tog.setAttribute("aria-pressed", String(state.showSeg));
      if (state.lastSegBase) {
        paintSeg($("segcanvas"), state.lastSegBase, state.lastSeg,
                 state.segMeta.colours, state.showSeg);
      }
    });
  }
}

/* ------------------------------------------------------------- inference --- */

async function classify(img, truthLabel, refKey = null, refProb = null) {
  if (!state.session) {
    setStatus(state.loadError
      ? `The model did not load: ${state.loadError}`
      : "Model is still loading…", true);
    return;
  }
  if (state.busy) return;
  state.busy = true;
  setStatus("Running…");
  state.lastImg = img;
  state.lastRef = refKey;
  state.lastRefProb = refProb;
  // a new frame invalidates the old masks; stage two re-runs below only if
  // its model is already in memory, so a visitor never pays for it twice
  state.lastSeg = null;
  $("seg-out").hidden = true;

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

    // Grad-CAM, when the export carried the feature map and weights.
    let camUp = null;
    const fo = out["features"];
    if (fo && state.meta.cam_weights) {
      const cam = computeCAM(fo.data, fo.dims, state.meta.cam_weights,
                             state.meta.cam_class ?? 1);
      camUp = upsampleCAM(cam, meta.img_size);
    }
    state.lastCam = camUp;
    state.lastCrop = cropCanvas;

    // examples.json carries the probability PyTorch produced for each bundled
    // frame, so the browser pipeline can be checked instead of trusted. A
    // preprocessing drift shows up here as a difference in the third decimal.
    if (state.lastRefProb != null) {
      console.log("classifier vs PyTorch reference:",
        { browser: p, pytorch: state.lastRefProb,
          absDiff: Math.abs(p - state.lastRefProb) });
    }
    render(p, cropCanvas, truthLabel, dt);
    setStatus(`Done in ${dt.toFixed(0)} ms, locally.`);

    if (state.segSession) {
      try {
        await segment(img);
      } catch (err) {
        console.error(err);
        setStatus(`Segmentation failed: ${err.message}`, true);
        $("seg-run").hidden = false;          // leave a way to retry
      }
    }
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
  state.lastIsStd = isStd;

  const prev = $("preview");
  prev.width = cropCanvas.width;
  prev.height = cropCanvas.height;
  paintOverlay(prev, cropCanvas, state.showCam ? state.lastCam : null);
  const btn = $("cam-toggle");
  if (btn) btn.hidden = !state.lastCam;

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
  // init() can run more than once now that loading can be retried, and the
  // buttons must not accumulate on each attempt.
  if (loadExamples.done) return;
  try {
    const list = await fetchJson("examples/examples.json", "the examples");
    const box = $("example-buttons");
    loadExamples.done = true;
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
          await classify(img, ex.label, ex.file, ex.reference_prob);
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

function wireCamToggle() {
  const btn = $("cam-toggle");
  if (!btn) return;
  btn.addEventListener("click", () => {
    state.showCam = !state.showCam;
    btn.textContent = state.showCam ? "Hide what the model looked at"
                                    : "Show what the model looked at";
    btn.setAttribute("aria-pressed", String(state.showCam));
    if (state.lastCrop) paintOverlay($("preview"), state.lastCrop,
                                     state.showCam ? state.lastCam : null);
  });
}

function wireRetry() {
  const btn = $("retry");
  if (!btn) return;
  btn.addEventListener("click", async () => {
    btn.disabled = true;
    try {
      await init();
    } finally {
      btn.disabled = false;
    }
  });
}

wireCamToggle();
wireSeg();
wireInput();
wireRetry();
init();
