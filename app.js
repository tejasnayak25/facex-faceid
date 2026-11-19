/* =====================================================================
   FaceX Check-In — offline face-ID booth
   Pipeline: wasm detect (FaceX FCOS, 320²) → 112² crop → facex_tiny/nano
   embedding (onnxruntime-web, fully local) → brute-force cosine vs IndexedDB.
   No network requests are made at runtime.
   ===================================================================== */
'use strict';

// ---------- encrypted weight decryption (same scheme as the FaceX demo) ----------
const _kp = new Uint8Array([
  0x42, 0x8a, 0x2f, 0x98, 0xd7, 0x28, 0xae, 0x22, 0x71, 0x37, 0x44, 0x91, 0x23, 0xef, 0x65, 0xcd,
  0xb5, 0xc0, 0xfb, 0xcf, 0xec, 0x4d, 0x3b, 0x2f, 0xe9, 0xb5, 0xdb, 0xa5, 0x81, 0x89, 0xdb, 0xbc,
]);
const _ko = new Uint8Array([
  0xb7, 0x4d, 0xb2, 0x24, 0xec, 0x76, 0x04, 0xe4, 0xcb, 0x46, 0xc8, 0x02, 0x97, 0x3c, 0x33, 0xfa,
  0x8b, 0xb0, 0x11, 0x60, 0x0c, 0xd5, 0x02, 0xf3, 0xac, 0xb8, 0x62, 0x0c, 0xf3, 0xad, 0x54, 0x01,
]);

async function loadEncryptedSession(url) {
  const K = new Uint8Array(32);
  for (let i = 0; i < 32; i++) K[i] = _kp[i] ^ _ko[i];
  const r = await fetch(url);
  if (!r.ok) throw new Error(`fetch ${url} → ${r.status}`);
  const buf = new Uint8Array(await r.arrayBuffer());
  const iv = buf.subarray(0, 12), data = buf.subarray(12);
  const ck = await crypto.subtle.importKey('raw', K, { name: 'AES-GCM' }, false, ['decrypt']);
  const plain = await crypto.subtle.decrypt({ name: 'AES-GCM', iv }, ck, data);
  K.fill(0);
  const sess = await ort.InferenceSession.create(new Uint8Array(plain), {
    executionProviders: ['wasm'],
  });
  return sess;
}

// ---------- detector pre/post processing ----------
const DET_HW = 320;
const detCv = document.createElement('canvas');
detCv.width = DET_HW; detCv.height = DET_HW;
const detCtx = detCv.getContext('2d', { willReadFrequently: true });
const detInput = new Float32Array(3 * DET_HW * DET_HW);

const recCv = document.createElement('canvas');
recCv.width = 112; recCv.height = 112;
const recCtx = recCv.getContext('2d', { willReadFrequently: true });
const recInput = new Float32Array(3 * 112 * 112);

const spoofCv = document.createElement('canvas');
spoofCv.width = 80; spoofCv.height = 80;
const spoofCtx = spoofCv.getContext('2d', { willReadFrequently: true });
const spoofInput27 = new Float32Array(3 * 80 * 80);
const spoofInput40 = new Float32Array(3 * 80 * 80);

const thumbCv = document.createElement('canvas');
thumbCv.width = 96; thumbCv.height = 96;
const thumbCtx = thumbCv.getContext('2d');

function sourceSize(src) {
  if (src instanceof HTMLVideoElement || src instanceof HTMLImageElement) {
    return { w: src.videoWidth || src.naturalWidth, h: src.videoHeight || src.naturalHeight };
  }
  return { w: src.width, h: src.height }; // canvas
}

function fillDetInput(src, W, H) {
  detCtx.fillStyle = '#000';
  detCtx.fillRect(0, 0, W, H);
  const { w: vw, h: vh } = sourceSize(src);
  if (!vw || !vh) return { scale: 1 };
  const scale = Math.min(W / vw, H / vh);
  const dw = Math.round(vw * scale), dh = Math.round(vh * scale);
  detCtx.drawImage(src, 0, 0, dw, dh);
  const img = detCtx.getImageData(0, 0, W, H).data;
  const N = W * H;
  for (let i = 0, p = 0; i < N; i++) {
    const r = img[p++], g = img[p++], b = img[p++]; p++;
    detInput[i] = (r - 127.5) / 128;
    detInput[N + i] = (g - 127.5) / 128;
    detInput[2 * N + i] = (b - 127.5) / 128;
  }
  return { scale };
}

function decodeFacex(out, W, H, scoreThr = 0.42, nmsThr = 0.4) {
  const strides = [8, 16, 32];
  const dets = [];
  for (let li = 0; li < 3; li++) {
    const s = strides[li];
    const cls = out['cls_p' + (li + 3)].data;
    const box = out['box_p' + (li + 3)].data;
    const Hs = H / s, Ws = W / s, N = Hs * Ws;
    for (let i = 0; i < N; i++) {
      const z = cls[i];
      const score = 1 / (1 + Math.exp(-z));
      if (score < scoreThr) continue;
      const y = Math.floor(i / Ws), x = i % Ws;
      const cx = (x + 0.5) * s, cy = (y + 0.5) * s;
      const l = Math.max(0, box[0 * N + i]) * s, t = Math.max(0, box[1 * N + i]) * s;
      const r = Math.max(0, box[2 * N + i]) * s, b = Math.max(0, box[3 * N + i]) * s;
      const x1 = cx - l, y1 = cy - t, x2 = cx + r, y2 = cy + b;
      if (x2 <= x1 || y2 <= y1) continue;
      const ar = (x2 - x1) / (y2 - y1);
      if (ar < 0.5 || ar > 1.75) continue;
      dets.push({ x1, y1, x2, y2, score });
    }
  }
  dets.sort((a, b) => b.score - a.score);
  const kept = [];
  while (dets.length) {
    const d = dets.shift();
    kept.push(d);
    for (let i = dets.length - 1; i >= 0; i--) {
      const e = dets[i];
      const ix1 = Math.max(d.x1, e.x1), iy1 = Math.max(d.y1, e.y1);
      const ix2 = Math.min(d.x2, e.x2), iy2 = Math.min(d.y2, e.y2);
      const iw = Math.max(0, ix2 - ix1), ih = Math.max(0, iy2 - iy1);
      const inter = iw * ih;
      const u = (d.x2 - d.x1) * (d.y2 - d.y1) + (e.x2 - e.x1) * (e.y2 - e.y1) - inter;
      if (inter / (u || 1e-6) > nmsThr) dets.splice(i, 1);
    }
  }
  return kept;
}

// ---------- embeddings ----------
async function detectFaces(src) {
  const { scale } = fillDetInput(src, DET_HW, DET_HW);
  const inp = new ort.Tensor('float32', detInput, [1, 3, DET_HW, DET_HW]);
  const out = await state.detSess.run({ input: inp });
  return decodeFacex(out, DET_HW, DET_HW).map(d => ({
    x1: d.x1 / scale, y1: d.y1 / scale, x2: d.x2 / scale, y2: d.y2 / scale, score: d.score,
  }));
}

function flipCHW(chw) {
  const N = 112 * 112, out = new Float32Array(chw.length);
  for (let c = 0; c < 3; c++) for (let y = 0; y < 112; y++) for (let x = 0; x < 112; x++) {
    out[c * N + y * 112 + (111 - x)] = chw[c * N + y * 112 + x];
  }
  return out;
}

async function embedFace(src, box) {
  const { w: vw, h: vh } = sourceSize(src);
  const bw = box.x2 - box.x1, bh = box.y2 - box.y1;
  const side = Math.max(bw, bh) * 1.3;
  const cx = (box.x1 + box.x2) / 2, cy = (box.y1 + box.y2) / 2;
  const sx = cx - side / 2, sy = cy - side / 2;
  recCtx.fillStyle = '#000';
  recCtx.fillRect(0, 0, 112, 112);
  recCtx.drawImage(src, sx, sy, side, side, 0, 0, 112, 112);
  const img = recCtx.getImageData(0, 0, 112, 112).data;
  const N = 112 * 112;
  for (let i = 0, p = 0; i < N; i++) {
    const r = img[p++], g = img[p++], b = img[p++]; p++;
    recInput[i] = (r - 127.5) / 128;
    recInput[N + i] = (g - 127.5) / 128;
    recInput[2 * N + i] = (b - 127.5) / 128;
  }
  // flip-averaged embedding: average of normal + mirrored pass, renormalized
  // (recovers part of the gap left by the browser-grade tiny model)
  const t1 = new ort.Tensor('float32', recInput, [1, 3, 112, 112]);
  const t2 = new ort.Tensor('float32', flipCHW(recInput), [1, 3, 112, 112]);
  const [o1, o2] = await Promise.all([state.recSess.run({ input: t1 }), state.recSess.run({ input: t2 })]);
  const d1 = o1['embedding'].data, d2 = o2['embedding'].data;
  const dim = d1.length;
  const emb = new Float32Array(dim);
  let norm = 0;
  for (let i = 0; i < dim; i++) { emb[i] = d1[i] + d2[i]; norm += emb[i] * emb[i]; }
  norm = Math.sqrt(norm) || 1;
  for (let i = 0; i < dim; i++) emb[i] /= norm;
  return emb;
}

function cropThumbDataURL(src, box) {
  const bw = box.x2 - box.x1, bh = box.y2 - box.y1;
  const side = Math.max(bw, bh) * 1.3;
  const cx = (box.x1 + box.x2) / 2, cy = (box.y1 + box.y2) / 2;
  thumbCtx.fillStyle = '#000';
  thumbCtx.fillRect(0, 0, 96, 96);
  thumbCtx.drawImage(src, cx - side / 2, cy - side / 2, side, side, 0, 0, 96, 96);
  return thumbCv.toDataURL('image/jpeg', 0.72);
}

function fillSpoofTensor(src, box, scale, dst) {
  const { w: vw, h: vh } = sourceSize(src);
  const bw = box.x2 - box.x1, bh = box.y2 - box.y1;
  let s = Math.min((vh - 1) / bh, Math.min((vw - 1) / bw, scale));
  let nw = bw * s, nh = bh * s;
  let cx = (box.x1 + box.x2) / 2, cy = (box.y1 + box.y2) / 2;
  let ltx = cx - nw / 2, lty = cy - nh / 2;
  let rbx = cx + nw / 2, rby = cy + nh / 2;
  if (ltx < 0) { rbx -= ltx; ltx = 0; }
  if (lty < 0) { rby -= lty; lty = 0; }
  if (rbx > vw - 1) { ltx -= rbx - vw + 1; rbx = vw - 1; }
  if (rby > vh - 1) { lty -= rby - vh + 1; rby = vh - 1; }
  spoofCtx.fillStyle = '#000';
  spoofCtx.fillRect(0, 0, 80, 80);
  spoofCtx.drawImage(src, ltx, lty, rbx - ltx, rby - lty, 0, 0, 80, 80);
  const img = spoofCtx.getImageData(0, 0, 80, 80).data;
  const N = 80 * 80;
  for (let i = 0, p = 0; i < N; i++) {
    const r = img[p++], g = img[p++], b = img[p++]; p++;
    dst[i] = b; dst[N + i] = g; dst[2 * N + i] = r; // BGR, raw [0,255]
  }
}

async function spoofLiveScore(src, box) {
  if (!state.spoof27Sess || !state.spoof40Sess) return 1;
  fillSpoofTensor(src, box, 2.7, spoofInput27);
  fillSpoofTensor(src, box, 4.0, spoofInput40);
  const t27 = new ort.Tensor('float32', spoofInput27, [1, 3, 80, 80]);
  const t40 = new ort.Tensor('float32', spoofInput40, [1, 3, 80, 80]);
  const [o27, o40] = await Promise.all([state.spoof27Sess.run({ input: t27 }), state.spoof40Sess.run({ input: t40 })]);
  return (o27['prob'].data[1] + o40['prob'].data[1]) / 2;
}

// ---------- WASM SDK engine (EdgeFace-XS embedder + YuNet-500m detector, 5-point aligned) ----------
let sdkFx = null;
function ensureSdk() {
  if (sdkFx) return sdkFx;
  if (typeof FaceXSDK === 'undefined') throw new Error('SDK scripts not loaded — check sdk/*.js paths');
  sdkFx = new FaceXSDK({
    detWeightsUrl: 'models/yunet_fp32.bin',
    embWeightsUrl: 'models/edgeface_xs_fp32.bin',
    detSize: 160,
    onProgress: (m) => setStat('#statState', m),
  });
  return sdkFx.load().then(() => sdkFx);
}

// Unified source → faces + aligned embeddings pipeline (both engines).
async function analyzeSource(src) {
  if (state.engine === 'sdk') {
    const t0 = performance.now();
    const fx = await ensureSdk();
    const r = fx.process(src);   // synchronous WASM; faces + embeddings aligned
    return {
      faces: r.faces.map(f => ({ x1: f.x1, y1: f.y1, x2: f.x2, y2: f.y2, score: f.score })),
      embeddings: r.embeddings,
      detMs: 0,
      totalMs: performance.now() - t0,
    };
  }
  // ONNX fallback (tiny/nano)
  const td0 = performance.now();
  const faces = await detectFaces(src);
  const detMs = performance.now() - td0;
  const embeddings = [];
  for (const f of faces) embeddings.push(await embedFace(src, f));
  return { faces, embeddings, detMs, totalMs: performance.now() - td0 };
}

// ---------- IndexedDB ----------
const DB_NAME = 'facex-checkin', DB_VER = 1;
function idbOpen() {
  return new Promise((res, rej) => {
    const r = indexedDB.open(DB_NAME, DB_VER);
    r.onupgradeneeded = () => {
      const db = r.result;
      if (!db.objectStoreNames.contains('students')) db.createObjectStore('students', { keyPath: 'id', autoIncrement: true });
      if (!db.objectStoreNames.contains('logs')) {
        const s = db.createObjectStore('logs', { keyPath: 'id', autoIncrement: true });
        s.createIndex('ts', 'ts');
      }
    };
    r.onsuccess = () => res(r.result);
    r.onerror = () => rej(r.error);
  });
}
const idb = {
  async _tx(store, mode, fn) {
    const db = await idbOpen();
    return new Promise((res, rej) => {
      const tx = db.transaction(store, mode);
      const req = fn(tx.objectStore(store));
      tx.oncomplete = () => { db.close(); res(req && req.result); };
      tx.onerror = () => { db.close(); rej(tx.error); };
    });
  },
  studentsAll() {
    return idbOpen().then(db => new Promise((res, rej) => {
      const rq = db.transaction('students', 'readonly').objectStore('students').getAll();
      rq.onsuccess = () => { db.close(); res(rq.result || []); };
      rq.onerror = () => rej(rq.error);
    }));
  },
  studentAdd(s) { return this._tx('students', 'readwrite', st => st.add(s)); },
  studentDelete(id) { return this._tx('students', 'readwrite', st => st.delete(id)); },
  studentsClear() { return this._tx('students', 'readwrite', st => st.clear()); },
  logsAll() {
    return idbOpen().then(db => new Promise((res, rej) => {
      const rq = db.transaction('logs', 'readonly').objectStore('logs').getAll();
      rq.onsuccess = () => { db.close(); res((rq.result || []).sort((a, b) => b.ts - a.ts)); };
      rq.onerror = () => rej(rq.error);
    }));
  },
  logAdd(l) { return this._tx('logs', 'readwrite', st => st.add(l)); },
  logsClear() { return this._tx('logs', 'readwrite', st => st.clear()); },
};

// ---------- app state ----------
const state = {
  detSess: null, recSess: null, spoof27Sess: null, spoof40Sess: null,
  students: [],           // cached in memory for fast matching
  running: false,
  processing: false,
  lastFrameAt: 0,
  threshold: 0.25,
  cooldownMin: 5,
  multiFrame: true,
  spoofEnabled: false,
  engine: 'sdk',              // 'sdk' = EdgeFace-XS aligned WASM | 'tiny'/'nano' = ONNX
  recognitionModel: 'tiny',
  consecutive: { id: null, count: 0 },
  cooldowns: new Map(),   // roll -> last check-in ts
  ewma: { frame: 0, det: 0, emb: 0, search: 0 },
  facesPerSec: 0,
  faceEvents: [],         // rolling window timestamps
  pendingShots: [],       // enrollment captures
  bannerUntil: 0,
};

const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];
const video = $('#video'), overlay = $('#overlay');
const octx = overlay.getContext('2d');

function toast(msg, ms = 2200) {
  const t = $('#toast');
  t.textContent = msg;
  t.style.display = 'block';
  clearTimeout(t._h);
  t._h = setTimeout(() => (t.style.display = 'none'), ms);
}
function setStat(id, v) { $(id).textContent = v; }
function fmtMs(v) { return v ? v.toFixed(1) + ' ms' : '—'; }
function ewma(key, v) { state.ewma[key] = state.ewma[key] ? state.ewma[key] * 0.85 + v * 0.15 : v; }

// true cosine similarity (handles non-unit embeddings, e.g. the SDK's EdgeFace-XS)
function cosine(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
  const denom = Math.sqrt(na * nb);
  return denom > 0 ? dot / denom : 0;
}

function bestMatch(emb) {
  let best = null, bestScore = -1;
  const t0 = performance.now();
  for (const s of state.students) {
    const sc = cosine(emb, s.embedding);
    if (sc > bestScore) { bestScore = sc; best = s; }
  }
  state.ewma.search = state.ewma.search ? state.ewma.search * 0.85 + (performance.now() - t0) * 0.15 : performance.now() - t0;
  return { student: best, score: bestScore };
}

// ---------- overlay drawing ----------
function fitTransform() {
  const cw = overlay.width, ch = overlay.height;
  const { w: vw, h: vh } = sourceSize(video);
  if (!vw) return null;
  const s = Math.max(cw / vw, ch / vh);
  const dx = (cw - vw * s) / 2, dy = (ch - vh * s) / 2;
  return { s, dx, dy, cw, ch };
}
function resizeOverlayToStage() {
  const stage = overlay.parentElement;
  overlay.width = stage.clientWidth;
  overlay.height = stage.clientHeight;
}
window.addEventListener('resize', resizeOverlayToStage);

function drawDetections(faces, results) {
  resizeOverlayToStage();
  const t = fitTransform();
  octx.clearRect(0, 0, overlay.width, overlay.height);
  if (!t) return;
  const { s, dx, dy, cw } = t;
  // video is mirrored (scaleX(-1)); mirror overlay drawing to match
  octx.save();
  octx.translate(cw, 0);
  octx.scale(-1, 1);
  octx.translate(dx, dy);
  octx.scale(s, s);
  for (let i = 0; i < faces.length; i++) {
    const f = faces[i];
    const ok = results[i] && results[i].matched;
    octx.lineWidth = 3 / s;
    octx.strokeStyle = ok ? '#10b981' : '#f59e0b';
    octx.strokeRect(f.x1, f.y1, f.x2 - f.x1, f.y2 - f.y1);
  }
  octx.restore();
  // labels in identity space, mirrored manually
  octx.font = '600 13px system-ui';
  octx.textBaseline = 'bottom';
  for (let i = 0; i < faces.length; i++) {
    const f = faces[i];
    const xMirror = cw - (dx + f.x2 * s);
    const y = dy + f.y1 * s - 6;
    const label = results[i] ? results[i].label : '';
    const w = octx.measureText(label).width + 16;
    octx.fillStyle = results[i] && results[i].matched ? 'rgba(6,46,26,.92)' : 'rgba(46,9,9,.92)';
    roundRect(octx, xMirror - 2, y - 22, w, 22, 6);
    octx.fill();
    octx.fillStyle = results[i] && results[i].matched ? '#6ee7b7' : '#fca5a5';
    octx.fillText(label, xMirror + 6, y - 4);
  }
}
function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

// ---------- check-in core ----------
async function processFrame(src, { forLog = true } = {}) {
  const t0 = performance.now();
  const res = await analyzeSource(src);
  const faces = res.faces;

  const results = [];
  for (let i = 0; i < faces.length; i++) {
    const f = faces[i];
    const emb = res.embeddings[i];

    // anti-spoof gate
    let live = 1;
    if (state.spoofEnabled) live = await spoofLiveScore(src, f);
    const liveOk = live >= 0.5;

    const { student, score } = bestMatch(emb);
    const matched = !!student && score >= state.threshold && liveOk;
    results.push({
      matched, score, live, student,
      label: liveOk
        ? (matched ? `${student.name} · ${score.toFixed(2)}` : 'UNKNOWN')
        : `SPOOF? (live ${(live * 100) | 0}%)`,
    });

    if (forLog) {
      state.faceEvents.push(performance.now());
      handleConsecutive(matched ? student.roll : null, matched, student, score, src, f);
    }
  }
  const total = performance.now() - t0;
  ewma('det', res.detMs);
  ewma('frame', total);
  return { faces, results, totalMs: total };
}

function handleConsecutive(roll, matched, student, score, src, face) {
  if (!matched) {
    if (!state.multiFrame || state.consecutive.id !== null) state.consecutive = { id: null, count: 0 };
    return;
  }
  if (!state.multiFrame) return doCheckIn(student, score, src, face);
  if (state.consecutive.id === roll) state.consecutive.count++;
  else state.consecutive = { id: roll, count: 1 };
  if (state.consecutive.count === 3) doCheckIn(student, score, src, face);
}

function doCheckIn(student, score, src, face) {
  const now = Date.now();
  const last = state.cooldowns.get(student.roll) || 0;
  if (now - last < state.cooldownMin * 60 * 1000) return; // cooldown — already checked in
  state.cooldowns.set(student.roll, now);
  const log = {
    ts: now, name: student.name, roll: student.roll,
    similarity: +score.toFixed(4), thumb: cropThumbDataURL(src, face),
  };
  idb.logAdd(log).then(refreshLogs);
  showBanner(student, score, log.thumb, false);
  chime(true);
}

function showBanner(student, score, thumb, unknown) {
  const b = $('#banner');
  b.classList.toggle('unknown', unknown);
  $('#bannerImg').src = thumb || '';
  $('#bannerName').textContent = unknown ? 'Unknown face' : student.name;
  $('#bannerSub').textContent = unknown
    ? 'Not in database — check enrollment'
    : `Checked in · roll ${student.roll} · similarity ${(score * 100).toFixed(1)}%`;
  b.style.display = 'flex';
  state.bannerUntil = performance.now() + 2600;
}

// ---------- audio cue (WebAudio, offline) ----------
let audioCtx = null;
function chime(success) {
  try {
    audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)();
    const t = audioCtx.currentTime;
    const notes = success ? [880, 1320] : [220, 180];
    notes.forEach((f, i) => {
      const o = audioCtx.createOscillator(), g = audioCtx.createGain();
      o.type = 'sine'; o.frequency.value = f;
      g.gain.setValueAtTime(0.0001, t + i * 0.09);
      g.gain.exponentialRampToValueAtTime(0.25, t + i * 0.09 + 0.02);
      g.gain.exponentialRampToValueAtTime(0.0001, t + i * 0.09 + 0.22);
      o.connect(g); g.connect(audioCtx.destination);
      o.start(t + i * 0.09); o.stop(t + i * 0.09 + 0.25);
    });
  } catch (e) { /* audio blocked — visual feedback still works */ }
}

// ---------- main booth loop ----------
async function boothLoop() {
  if (!state.running) return;
  requestAnimationFrame(boothLoop);
  resizeOverlayToStage();
  const now = performance.now();
  if (now - state.lastFrameAt < 250) return;   // ~4 booth scans/sec, plenty for check-in
  if (state.processing) return;
  state.lastFrameAt = now;
  if (video.readyState < 2) return;

  state.processing = true;
  $('#liveDot').classList.add('busy');
  try {
    const { faces, results } = await processFrame(video, { forLog: true });
    drawDetections(faces, results);
    // hide banner after timeout
    if (state.bannerUntil && now > state.bannerUntil) { $('#banner').style.display = 'none'; state.bannerUntil = 0; }
    // throttle UI counters
    setStat('#statMs', fmtMs(state.ewma.frame));
    setStat('#statDet', state.engine === 'sdk' ? 'wasm' : fmtMs(state.ewma.det));
    setStat('#statEmb', state.engine === 'sdk' ? 'wasm' : fmtMs(state.ewma.emb));
    setStat('#statSearch2', state.ewma.search ? (state.ewma.search * 1000).toFixed(0) + ' µs' : '—');
    setStat('#statSearch', state.ewma.search ? (state.ewma.search * 1000).toFixed(0) + ' µs' : '—');
    setStat('#statState', faces.length ? 'scanning' : 'waiting for face');
    // faces/sec over trailing 5 s
    const cutoff = now - 5000;
    state.faceEvents = state.faceEvents.filter(t => t >= cutoff);
    setStat('#statFps', (state.faceEvents.length / 5).toFixed(1));
  } catch (err) {
    console.error(err);
    setStat('#statState', 'error — see console');
  } finally {
    state.processing = false;
    $('#liveDot').classList.remove('busy');
  }
}

// ---------- enrollment ----------
function renderShots() {
  const strip = $('#shotStrip');
  strip.innerHTML = '';
  const TOTAL = 5;
  for (let i = 0; i < TOTAL; i++) {
    const d = document.createElement('div');
    d.className = 'shot';
    if (state.pendingShots[i]) {
      const img = document.createElement('img');
      img.src = state.pendingShots[i].thumb;
      d.appendChild(img);
    } else d.textContent = `shot ${i + 1}`;
    strip.appendChild(d);
  }
  $('#btnCapture').textContent = `📷 Capture shot (${state.pendingShots.length}/${TOTAL})`;
  $('#btnSaveStudent').disabled = state.pendingShots.length < 3 || !$('#fName').value.trim() || !$('#fRoll').value.trim();
}

async function captureShot() {
  if (!state.running) return toast('Start the camera first');
  const { faces, embeddings } = await analyzeSource(video);
  if (!faces.length) return toast('No face detected — look at the camera');
  let best = 0;
  for (let i = 1; i < faces.length; i++) {
    if ((faces[i].x2 - faces[i].x1) * (faces[i].y2 - faces[i].y1) > (faces[best].x2 - faces[best].x1) * (faces[best].y2 - faces[best].y1)) best = i;
  }
  state.pendingShots.push({ emb: embeddings[best], thumb: cropThumbDataURL(video, faces[best]) });
  chime(true);
  renderShots();
}

async function saveStudent() {
  const name = $('#fName').value.trim(), roll = $('#fRoll').value.trim();
  if (!name || !roll) return toast('Name and roll required');
  if (state.pendingShots.length < 3) return toast('Capture at least 3 shots');
  // average the (already L2-normalized) embeddings, then renormalize
  const dim = state.pendingShots[0].emb.length;
  const avg = new Float32Array(dim);
  for (const s of state.pendingShots) for (let i = 0; i < dim; i++) avg[i] += s.emb[i] / state.pendingShots.length;
  let norm = 0; for (let i = 0; i < dim; i++) norm += avg[i] * avg[i];
  norm = Math.sqrt(norm) || 1;
  for (let i = 0; i < dim; i++) avg[i] /= norm;
  await idb.studentAdd({ name, roll, embedding: avg, thumb: state.pendingShots[0].thumb, shots: state.pendingShots.length, enrolledAt: Date.now(), engine: state.engine });
  await reloadStudents();
  state.pendingShots = [];
  $('#fName').value = ''; $('#fRoll').value = '';
  renderShots();
  toast(`Saved ${name}`);
}

function parseRollName(filename) {
  const base = filename.replace(/\.[^.]+$/, '');
  const i = base.indexOf('_');
  if (i <= 0) return { roll: base, name: 'Unnamed' };
  return { roll: base.slice(0, i).trim(), name: base.slice(i + 1).replace(/[_-]+/g, ' ').trim() };
}

async function batchEnroll() {
  const files = [...$('#batchPhotos').files];
  if (!files.length) return toast('Pick photos first');
  const status = $('#batchStatus');
  let ok = 0, fail = 0;
  for (let i = 0; i < files.length; i++) {
    status.textContent = `Processing ${i + 1}/${files.length}…`;
    try {
      const img = await fileToImage(files[i]);
      const { faces, embeddings } = await analyzeSource(img);
      if (!faces.length) throw new Error('no face');
      let best = 0;
      for (let j = 1; j < faces.length; j++) if (faces[j].score > faces[best].score) best = j;
      const emb = embeddings[best];
      const f = faces[best];
      const { roll, name } = parseRollName(files[i].name);
      await idb.studentAdd({ name, roll, embedding: emb, thumb: cropThumbDataURL(img, f), shots: 1, enrolledAt: Date.now(), engine: state.engine });
      ok++;
    } catch (e) { console.warn(files[i].name, e); fail++; }
  }
  status.textContent = `Done — ${ok} enrolled, ${fail} failed`;
  await reloadStudents();
  toast(`Batch enrollment: ${ok} saved`);
}

function fileToImage(file) {
  return new Promise((res, rej) => {
    const img = new Image();
    img.onload = () => res(img);
    img.onerror = rej;
    img.src = URL.createObjectURL(file);
  });
}

// ---------- database tab ----------
async function reloadStudents() {
  state.students = await idb.studentsAll();
  setStat('#statDb', String(state.students.length));
  renderDbList($('#dbSearch').value.trim().toLowerCase());
  warnEngineMismatch();
}
// students enrolled with a different recognition engine will never match reliably
function rosterEngineMismatch() {
  return state.students.filter(s => (s.engine || 'tiny') !== state.engine).length;
}
function warnEngineMismatch() {
  const n = rosterEngineMismatch();
  if (n > 0) toast(`${n} enrolled face(s) use the old engine — delete & re-enroll them or they'll never match`, 6500);
}
function renderDbList(filter = '') {
  const list = $('#dbList');
  const rows = state.students.filter(s => !filter || s.name.toLowerCase().includes(filter) || s.roll.toLowerCase().includes(filter));
  if (!rows.length) { list.innerHTML = '<div class="empty">' + (filter ? 'No match.' : 'Database is empty.') + '</div>'; return; }
  list.innerHTML = '';
  for (const s of rows) {
    const d = document.createElement('div');
    d.className = 'db-item';
    d.innerHTML = `<img src="${s.thumb || ''}" alt=""/><div><div class="n">${escapeHtml(s.name)}</div><div class="m">${escapeHtml(s.roll)} · ${new Date(s.enrolledAt).toLocaleDateString()}</div></div><button class="icon-btn" title="Remove">✕</button>`;
    d.querySelector('.icon-btn').onclick = async () => {
      await idb.studentDelete(s.id);
      await reloadStudents();
      toast('Removed ' + s.name);
    };
    list.appendChild(d);
  }
}
const escapeHtml = (s) => s.replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

async function exportDb() {
  const students = state.students.map(s => ({ ...s, embedding: Array.from(s.embedding) }));
  const logs = await idb.logsAll();
  const blob = new Blob([JSON.stringify({ version: 1, exportedAt: Date.now(), students, logs })], { type: 'application/json' });
  downloadBlob(blob, `facex_db_${new Date().toISOString().slice(0, 10)}.json`);
}
async function importDb(file) {
  const data = JSON.parse(await file.text());
  let n = 0;
  for (const s of data.students || []) {
    const emb = s.embedding instanceof Array ? Float32Array.from(s.embedding) : null;
    if (!emb || (emb.length !== 512 && emb.length !== 256)) continue;
    await idb.studentAdd({ name: s.name || 'Unnamed', roll: s.roll || '?', embedding: emb, thumb: s.thumb || '', shots: s.shots || 1, enrolledAt: s.enrolledAt || Date.now(), engine: s.engine || 'tiny' });
    n++;
  }
  await reloadStudents();
  toast(`Imported ${n} students`);
}
function downloadBlob(blob, name) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
}

// ---------- logs ----------
function sameDay(a, b) { const x = new Date(a), y = new Date(b); return x.toDateString() === y.toDateString(); }
async function refreshLogs() {
  const logs = await idb.logsAll();
  const today = logs.filter(l => sameDay(l.ts, Date.now()));
  setStat('#todayCount', String(today.length));
  // rebuild cooldown map from today's logs
  for (const l of today) state.cooldowns.set(l.roll, Math.max(state.cooldowns.get(l.roll) || 0, l.ts));
  const list = $('#logList');
  if (!today.length) { list.innerHTML = '<div class="empty">No check-ins yet today.<br/>Stand in front of the camera and hold still.</div>'; return; }
  list.innerHTML = '';
  for (const l of today) {
    const d = document.createElement('div');
    d.className = 'log-item';
    const time = new Date(l.ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    d.innerHTML = `<img src="${l.thumb || ''}" alt=""/><div><div class="n">${escapeHtml(l.name)}</div><div class="m">${escapeHtml(l.roll)} · ${(l.similarity * 100).toFixed(1)}%</div></div><time>${time}</time>`;
    list.appendChild(d);
  }
}
async function exportCsv() {
  const logs = await idb.logsAll();
  const rows = [['timestamp_iso', 'name', 'roll', 'similarity']];
  for (const l of [...logs].reverse()) rows.push([new Date(l.ts).toISOString(), csvEsc(l.name), csvEsc(l.roll), l.similarity]);
  downloadBlob(new Blob([rows.map(r => r.join(',')).join('\n')], { type: 'text/csv' }), `facex_checkins_${new Date().toISOString().slice(0, 10)}.csv`);
}
const csvEsc = (s) => `"${String(s).replace(/"/g, '""')}"`;

// ---------- test image ----------
async function testPhoto(file) {
  const out = $('#testPhotoResult');
  out.textContent = 'Processing…';
  try {
    const img = await fileToImage(file);
    const { faces, results } = await processFrame(img, { forLog: false });
    if (!faces.length) { out.textContent = 'No face detected in that photo.'; return; }
    const r = results[0];
    out.innerHTML = r.matched
      ? `✓ <b style="color:#6ee7b7">${escapeHtml(r.student.name)}</b> — similarity <b>${(r.score * 100).toFixed(1)}%</b> (above ${(state.threshold * 100).toFixed(0)}% threshold)`
      : `✗ No match — best similarity ${(r.score * 100).toFixed(1)}% (threshold ${(state.threshold * 100).toFixed(0)}%)`;
  } catch (e) {
    out.textContent = 'Error: ' + e.message;
  }
}

// ---------- model loading ----------
async function loadCoreModels() {
  const t0 = performance.now();
  setStat('#statState', 'loading models…');
  $('#liveDot').classList.add('busy');
  if (state.engine === 'sdk') {
    await ensureSdk();
  } else {
    state.detSess = await loadEncryptedSession('models/facex_detect.enc');
    state.recSess = await loadEncryptedSession(`models/facex_${state.recognitionModel}.enc`);
  }
  setStat('#statLoad', ((performance.now() - t0) / 1000).toFixed(2) + ' s');
  $('#liveDot').classList.remove('busy');
}
async function loadSpoofModels() {
  if (state.spoof27Sess) return;
  state.spoof27Sess = await loadEncryptedSession('models/minifasnet_v2_27.enc');
  state.spoof40Sess = await loadEncryptedSession('models/minifasnet_v1se_40.enc');
}
async function switchEngine(value) {
  state.engine = value;
  localStorage.setItem('fx_engine', value);
  localStorage.setItem('fx_model', value);
  warnEngineMismatch();
  const t0 = performance.now();
  if (value === 'sdk') {
    await ensureSdk();
    setStat('#statLoad', ((performance.now() - t0) / 1000).toFixed(2) + ' s');
    toast('Recognition engine → EdgeFace-XS (aligned)');
  } else {
    if (!state.detSess) state.detSess = await loadEncryptedSession('models/facex_detect.enc');
    state.recognitionModel = value;
    state.recSess = await loadEncryptedSession(`models/facex_${value}.enc`);
    setStat('#statLoad', ((performance.now() - t0) / 1000).toFixed(2) + ' s (rec only)');
    toast(`Recognition engine → facex_${value}`);
  }
}

// ---------- wiring ----------
function bindUI() {
  $$('.tab').forEach(t => t.addEventListener('click', () => {
    $$('.tab').forEach(x => x.classList.toggle('active', x === t));
    $$('.tab-pane').forEach(p => p.classList.toggle('active', p.id === 'pane-' + t.dataset.tab));
  }));

  $('#btnStart').addEventListener('click', async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' }, audio: false });
      video.srcObject = stream;
      await video.play();
      $('#cta').style.display = 'none';
      state.running = true;
      setStat('#statState', 'scanning');
      boothLoop();
    } catch (e) {
      toast('Camera blocked: ' + e.message, 4000);
    }
  });

  $('#btnCapture').addEventListener('click', captureShot);
  $('#btnResetShots').addEventListener('click', () => { state.pendingShots = []; renderShots(); });
  $('#btnSaveStudent').addEventListener('click', saveStudent);
  $('#fName').addEventListener('input', renderShots);
  $('#fRoll').addEventListener('input', renderShots);
  $('#btnBatchEnroll').addEventListener('click', batchEnroll);

  $('#dbSearch').addEventListener('input', e => renderDbList(e.target.value.trim().toLowerCase()));
  $('#btnExportDb').addEventListener('click', exportDb);
  $('#btnImportDb').addEventListener('click', () => $('#importFile').click());
  $('#importFile').addEventListener('change', e => e.target.files[0] && importDb(e.target.files[0]).catch(err => toast('Import failed: ' + err.message, 4000)));
  $('#btnWipeDb').addEventListener('click', async () => {
    if (confirm('Wipe ALL enrolled students from this device?')) { await idb.studentsClear(); await reloadStudents(); toast('Database wiped'); }
  });

  $('#btnExportCsv').addEventListener('click', exportCsv);
  $('#btnClearLogs').addEventListener('click', async () => { if (confirm('Clear today\'s check-in log?')) { await idb.logsClear(); await refreshLogs(); } });

  const thr = $('#thrSlider');
  thr.addEventListener('input', () => { state.threshold = +thr.value; $('#thrVal').textContent = (+thr.value).toFixed(2); localStorage.setItem('fx_threshold', thr.value); });
  const cool = $('#cooldownSlider');
  cool.addEventListener('input', () => { state.cooldownMin = +cool.value; $('#cooldownVal').textContent = `${cool.value} min`; localStorage.setItem('fx_cooldown', cool.value); });
  $('#multiFrameToggle').addEventListener('change', e => { state.multiFrame = e.target.checked; localStorage.setItem('fx_multi', state.multiFrame ? '1' : '0'); });
  const engSel = $('#engineSelect');
  if (engSel) engSel.addEventListener('change', e => switchEngine(e.target.value).catch(err => toast('Engine switch failed: ' + err.message, 4000)));
  $('#spoofToggle').addEventListener('change', async e => {
    state.spoofEnabled = e.target.checked;
    localStorage.setItem('fx_spoof', e.target.checked ? '1' : '0');
    if (state.spoofEnabled) { setStat('#statState', 'loading anti-spoof…'); await loadSpoofModels(); toast('Anti-spoof ON (~2 ms/frame)'); setStat('#statState', 'scanning'); }
    else toast('Anti-spoof OFF');
  });
  $('#testPhoto').addEventListener('change', e => e.target.files[0] && testPhoto(e.target.files[0]));
}

function restoreSettings() {
  const thr = localStorage.getItem('fx_threshold');
  if (thr !== null) { $('#thrSlider').value = thr; state.threshold = +thr; }
  $('#thrVal').textContent = state.threshold.toFixed(2);
  const cool = localStorage.getItem('fx_cooldown');
  if (cool !== null) { $('#cooldownSlider').value = cool; state.cooldownMin = +cool; }
  $('#cooldownVal').textContent = `${state.cooldownMin} min`;
  if (localStorage.getItem('fx_multi') === '0') { $('#multiFrameToggle').checked = false; state.multiFrame = false; }
  // engine: prefer the fixed SDK engine unless the user explicitly chose one before.
  // Legacy `fx_model` was set by the OLD default (tiny) — force-upgrade it to sdk.
  const eng = localStorage.getItem('fx_engine')
    || (localStorage.getItem('fx_model') === 'nano' ? 'nano' : 'sdk');
  state.engine = eng;
  const engSel = $('#engineSelect');
  if (engSel) engSel.value = eng;
  if (eng !== 'sdk') state.recognitionModel = eng;
}

(async function init() {
  // offline wasm runtime — resolve against the page URL so sub-path hosting works too
  ort.env.wasm.wasmPaths = new URL('ort/', window.location.href).href;
  ort.env.wasm.numThreads = 1;   // threads need COOP/COEP headers; single-threaded wasm is plenty
  ort.env.wasm.simd = true;

  restoreSettings();
  bindUI();
  resizeOverlayToStage();
  renderShots();

  try {
    await loadCoreModels();
    await reloadStudents();
    await refreshLogs();
    setStat('#statState', 'press Start Camera');
    if (localStorage.getItem('fx_spoof') === '1') { $('#spoofToggle').checked = true; state.spoofEnabled = true; loadSpoofModels().catch(console.error); }
  } catch (err) {
    console.error(err);
    setStat('#statState', 'model load failed');
    toast('Model load failed — serve over http://localhost and use a Chromium browser. ' + err.message, 6000);
  }
})();
