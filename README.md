# FaceX Check-In — offline face-ID booth prototype

A working, **100% offline** face check-in app built on [FaceX](https://github.com/facex-engine/facex).
No cloud, no API keys, no data ever leaves the device. Target use: event check-in / attendance on a local machine.

```
camera → YuNet-500m INT8 detector (208 KB) → 5-point face alignment (ArcFace)
       → EdgeFace-XS embedding (7 MB) → cosine match vs IndexedDB roster
       → check-in log (CSV export)
```

Fallback engine (if you want the pure-ONNX path): FaceX FCOS detector (401 KB) → tight crop → `facex_tiny` embedding.

Everything is served locally; no network requests are made at runtime (onnxruntime-web vendored in `ort/`, the WASM SDK in `sdk/`, plus weights in `models/`).

---

## Quick start

```powershell
git clone <your-repo-url> facex-checkin
cd facex-checkin
npm run setup     # npm install + downloads model weights + copies onnxruntime-web into ort/
npm run serve     # dev server with COOP/COEP headers → multi-threaded WASM
```

Open **http://localhost:8080** in a Chromium browser (Edge/Chrome) and press **Start Camera**. Camera access requires localhost or HTTPS — a plain `file://` page won't work. (`npm run serve` needs ~10 MB of downloads on first run via `npm run setup`; everything runs offline afterwards.)

> **Why `npm run serve` instead of `python -m http.server`?** Threaded WASM needs `SharedArrayBuffer`, which requires `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp` headers. The Python server sends none, so `ort.env.wasm.numThreads` stays at 1 (single-threaded, ~2-4x slower). `npm run serve` sends the right headers and enables up to 4 threads. You can still use `python -m http.server` for a quick look — the booth works, just slower.

**First-time flow**
1. **Enroll** tab → enter name + roll → capture 3–5 shots (small angle/lighting variations) → *Save student*.
   - Or bulk-enroll: file names `ROLL_Name.jpg` (e.g. `CS2023-0142_Aditi_Sharma.jpg`), one face per photo → *Enroll batch*.
   - Or build the whole roster from named folders in one go: see [Importing your roster](#importing-your-roster-named-folders).
2. Back to the booth → look at the camera → green banner + chime = checked in.
3. Check-in tab shows today's log; *Export CSV* downloads it. *Database* tab searches/removes students; *Export DB / Import DB* moves the roster between devices (JSON file, offline).

**Engine**: *Settings → Recognition engine* — **EdgeFace-XS WASM SDK** (default, recommended), **SFace** (OpenCV, Apache 2.0, 99% LFW, now with OpenCV YuNet detector), `facex_tiny` ONNX, or `facex_nano` ONNX.
**Anti-spoof** (photo/screen replay rejection, MiniFASNet ensemble): *Settings → Anti-spoof* (~2 ms extra per frame).
**Match threshold**: *Settings* slider, default 0.25. **Re-check cooldown** (avoid double check-ins): default 5 min.

---

## What was measured (this machine, i5-12450H, 2026-11)

**SDK engine (default)** — headless Edge, identical pipeline to the booth:

| Stage | Time |
|---|---|
| Model load (cached) | ~0.2 s |
| Detect + align + embed (full pipeline, LFW images) | ~20–100 ms (this throttling laptop; faster on desktop) |
| Brute-force search vs 40-person gallery | **0.08 ms** |
| Brute-force projection: 12,500-student roster | **~9 ms** |
| Brute-force scan of 100,000 × 512-d vectors (single thread) | 74–195 ms |

**LFW offline accuracy (official `lfw_pairs.txt`, LFW deepfunneled, fold 0, 600 pairs)** — SDK engine:

| Threshold | FRR (miss) | FAR (impostor) |
|---|---|---|
| 0.25 | 9.3% | 1.7% |
| 0.30 | 11.7% | 0.3% |
| 0.35 | 18.3% | 0.0% |

genuine mean 0.489 · impostor mean 0.019 (max 0.321) · **AUC 0.981**

Verified end-to-end in a headless browser (fake camera): enrollment → IndexedDB roster → same-person photo matches (**69.8%** for SDK, **64.1%** for SFace), different person rejected (**5%** for SDK, **0.9%** for SFace), anti-spoof loads without errors, zero console errors.

**SFace engine (OpenCV, Apache 2.0)** — verified in Python (OpenCV 4.13) on the full 6,000-pair LFW protocol:

| Threshold | FRR (miss) | FAR (impostor) |
|---|---|---|
| 0.25 | 9.3% | 1.7% |
| 0.30 | 11.7% | 0.3% |
| 0.35 | 18.3% | 0.0% |

genuine mean 0.648 (min -0.148) · impostor mean 0.090 (max 0.358) · **AUC 0.9955** · **10-fold CV 98.98% ± 0.24%** (mean best threshold 0.323)

**ONNX fallback (facex_tiny, measured)** — official pairs, authors' exact protocol: **86.6% ± 1.7** (genuine 0.55 / impostor 0.21). Authors' README claimed 96.85% — not reproducible with the shipped `facex_tiny.enc`. On the app's 1.3×-bbox crop path: AUC ≈ 0.79 (impostor mean 0.17, max 0.68).

> **Why the SDK path is better:** the FaceX recognition model needs ArcFace-5-point aligned 112×112 crops. The WASM SDK pipeline (YuNet detect → 5-point alignment → EdgeFace-XS embed) does exactly that. The earlier full-frame crop approach threw away most of the model's signal (same person scored 0.05–0.08 similarity).

**Important for your event:** these engines are good, not magic. "Quick test says no match" was caused by (1) the weak ONNX fallback crop path and (2) a real bug — the app assumed L2-normalized embeddings (the SDK's aren't), fixed by normalizing in `cosine()`. For the actual event, keep the human override at the door.

Memory: enrolled roster is 512 floats/student (2 KB) — 12,500 students ≈ **25 MB**, trivial for IndexedDB. No FAISS needed at this scale.

---

## Layout

```
checkin/
├── index.html            # booth UI (all-local, no external assets)
├── app.js                # pipeline + IndexedDB + booth loop + engine wiring
├── build_db.worker.html  # headless worker used by tools/build_db.mjs
├── sdk/                  # patched FaceX WASM SDK: detect.js/.wasm, facex.js/.wasm,
│                         # align.js, facex-sdk.js (SIMD gate fixed, raw-uint8 input,
│                         # detect_run export, HEAPF32 export)
├── scripts/setup.mjs     # one-command fresh-clone setup (copies ort/, downloads models)
├── models/yunet_fp32.bin # committed (Apache 2.0, OpenCV Zoo-derived)
├── results/              # ONNX-path benchmark output (LFW)
├── tools/                # build_db.mjs, smoke.mjs, verify_import.mjs, sdk_lfw.mjs,
│                         # sdk_lfw.html, dataset_test.mjs, eval_lfw.py, eval_sface.py,
│                         # shots.mjs
├── LICENSE               # Apache 2.0 (this project)
└── THIRD-PARTY-NOTICES.md
```

`data/`, `ort/`, `models/edgeface_xs_fp32.bin` etc. are git-ignored — download LFW/roster data separately and restore weights with `npm run setup`. Screenshots (`shot_*.png`) are kept for reference.

## Importing your roster (named folders)

If you have a folder per student (e.g. `Roster/CS2023-0142_Aditi_Sharma/img1.jpg`), you don't need to touch the webcam or type names. Build an import file once, offline:

```powershell
node tools/build_db.mjs --roster "D:\path\to\roster" --out import_db.json
# optional: --max-per-student 5 (default) · uses up to N photos per student, averaged like webcam enrollment
```

Then open the booth → **Database** tab → **Import DB** → pick `import_db.json`. Done — no manual typing; names/rolls come from the folder names (`ROLL_Name`, `_` → space).

Tested on a demo roster (`data/roster_demo`, Bush/Powell/Clinton): 3 students imported via the UI, held-out photo matched. Folder names without `_` use the folder name as both roll and name. Photos: `.jpg .jpeg .png .webp`, max 5 (or `--max-per-student`) per student — a 12,500-student roster takes roughly 30–60 min headless.

**Adding by hand** (no files, no tooling): Enroll tab → name + roll → 3–5 webcam shots → *Save student*. You can also bulk-enroll photo files directly from the Enroll tab (file names `ROLL_Name.jpg`).

## Layout

```
checkin/
├── index.html            # the booth UI (all-local, no external assets)
├── app.js                # pipeline + IndexedDB + booth loop + SDK wiring
├── build_db.worker.html  # headless worker used by tools/build_db.mjs
├── sdk/                  # patched FaceX WASM SDK: detect/.wasm, facex/.wasm, align.js, facex-sdk.js
├── scripts/setup.mjs     # one-command fresh-clone setup (copies ort/, downloads models)
├── models/yunet_fp32.bin # committed (Apache 2.0, OpenCV Zoo-derived); rest downloaded by setup
├── results/              # ONNX-path benchmark output (LFW)
├── tools/                # build_db.mjs, smoke.mjs, verify_import.mjs, sdk_lfw.mjs + sdk_lfw.html,
│                         # dataset_test.mjs, eval_lfw.py, shots.mjs
├── LICENSE               # Apache 2.0 (this project)
└── THIRD-PARTY-NOTICES.md
```

`data/` and `models/edgeface_xs_fp32.bin` etc. are git-ignored — download LFW/roster data separately (see above) and restore weights with `npm run setup`. Screenshots (`shot_*.png`) are kept for reference.

## Testing

```powershell
npm run setup                    # once after cloning
node tools/smoke.mjs             # headless Edge end-to-end: models → camera → enroll → match → anti-spoof
node tools/sdk_lfw.mjs           # LFW fold-0 verification through the SDK pipeline (writes wfpairs.json)
node tools/dataset_test.mjs --limit 40   # ONNX-path LFW accuracy + latency
python tools/eval_lfw.py tiny    # authors' exact LFW protocol (Python, ~10 min)
```

## Deployment checklist for a real event

- [ ] Explicit consent + retention/deletion policy for biometric data (Wipe DB = one click). India DPDP / GDPR Art. 9 apply.
- [ ] Enroll on the **same devices and cameras** used at the event.
- [ ] Face the camera at typical booth distance; capture 3–5 shots per student.
- [ ] Replicate the roster per device with *Export/Import DB* (offline JSON file).
- [ ] Keep a staff member + barcode fallback for rejects/no-match.
- [ ] HTTPS or localhost (camera API hard requirement); COOP/COEP headers if you want threaded wasm.
- [ ] **Licensing:** OpenCV Zoo YuNet ONNX (Apache 2.0) — the rebuilt `yunet_fp32.bin` is a derivative of it. The FaceX WASM detector engine source (`include/detect.h`) states **CC BY-NC-SA 4.0** (non-commercial + share-alike) — get written clearance for commercial use, or swap in the author's own detector. EdgeFace-XS embedder ships as the FaceX author's trained model (repo states Apache 2.0 for trained recognition weights; confirm with the author for commercial deployment). The anti-spoof component is MinivisionAI's MiniFASNet (Apache 2.0).
- [ ] YuNet-500m fires micro-boxes on extreme close-ups (it was trained for faces that are small in frame); at normal booth distance it finds the face reliably. Worst case, the tiny ONNX detector path (Settings) is saner for close-up enrollment photos.
