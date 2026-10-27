# onnxruntime-web 1.19.2 (vendored)

These files are copied verbatim from the npm package `onnxruntime-web@1.19.2`,
MIT licensed, Copyright (c) Microsoft Corporation.

The demo used to load them from `cdn.jsdelivr.net` at run time. That put a
third party on the critical path of a page whose entire claim is that the image
never leaves the visitor's machine, and when the CDN was unreachable — blocked
host, DNS failure, an extension — the page reported the Fetch API's bare
"Failed to fetch" with no indication of which file was missing. Serving the
runtime from the same origin as the models removes both problems.

## What is here, and why only this

| File | Bytes | Purpose |
| --- | --- | --- |
| `ort.wasm.min.js` | 46,436 | the wasm-only bundle |
| `ort-wasm-simd-threaded.wasm` | 11,018,731 | the runtime itself |
| `ort-wasm-simd-threaded.mjs` | 24,503 | its loader |

`ort.min.js` (the default bundle) also carries the WebGL and WebGPU backends,
which this page does not use: `app.js` requests `executionProviders: ["wasm"]`
only. The wasm-only bundle is ~1 MB smaller for identical behaviour.

Threads are disabled (`ort.env.wasm.numThreads = 1`), because cross-origin
isolation is not available on GitHub Pages: serving `Cross-Origin-Opener-Policy`
and `Cross-Origin-Embedder-Policy` needs response headers, and Pages serves
static files without them. The threaded `.wasm` build is the one onnxruntime-web
1.19 ships in either case; it runs single-threaded when `numThreads` is 1.

## Provenance

Downloaded 2026-10-27 from
`https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/`.

```
98e831517582c08fc1b729def487a8b0152c83fc641746137eb4c7ad46032f1c  ort.wasm.min.js
1bf0b9ed7ad025cf9ca88ce6da29e54df3f128a169f8241d71823e81f078d578  ort-wasm-simd-threaded.wasm
d870a377322c3053fb97432d548423f165dd15e2af232947592fc07b0d2f3639  ort-wasm-simd-threaded.mjs
```

Verify with `sha256sum -c` against the list above, or re-fetch:

```bash
for f in ort.wasm.min.js ort-wasm-simd-threaded.wasm ort-wasm-simd-threaded.mjs; do
  curl -sS -o "$f" "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/$f"
done
```

To move to another version, replace all three files together — the `.js`
bundle and the `.wasm` binary are a matched pair — update the hashes here, and
bump the `?v=` on the script tag in `site/index.html`.
