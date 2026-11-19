/* =====================================================================
   FaceX dataset benchmark — measures the ACTUAL app pipeline (not the
   theoretical native numbers) on LFW deepfunneled.

   Usage:  node tools/dataset_test.mjs [--limit N] [--full]

   Protocol
   --------
   1. Top-1 identification: identities with >=3 images are split enroll
      (image 0) / query (images 1..k); a match requires nearest enrolled
      embedding == same identity. Reports top-1, top-5.
   2. Impostor test: identities with exactly 1 image (not enrolled) are
      used as "unknown" probes; reports false-accept rate at each of
      several cosine thresholds.
   3. Latency: p50/p95 per stage (detect / embed / search), plus a
      synthetic 100K-vector brute-force search to emulate booth-at-scale.
   ===================================================================== */
import * as fs from 'node:fs';
import * as path from 'node:path';
import * as crypto from 'node:crypto';
import { fileURLToPath } from 'node:url';
import jpeg from 'jpeg-js';
import * as ort from 'onnxruntime-node';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const LFW_DIR = path.join(root, 'data', 'lfw-deepfunneled');
const MODELS = path.join(root, 'models');
const RESULTS = path.join(root, 'results');
fs.mkdirSync(RESULTS, { recursive: true });

const args = process.argv.slice(2);
const FULL = args.includes('--full');
const LIMIT = (() => { const i = args.indexOf('--limit'); return i >= 0 ? +args[i + 1] : (FULL ? Infinity : 40); })(); // identities capped (quick pass by default)
const DET_HW = 320;

// ---------- decrypt (AES-256-GCM, iv||ct||tag, XOR-split master key) ----------
const _kp = [0x42,0x8a,0x2f,0x98,0xd7,0x28,0xae,0x22,0x71,0x37,0x44,0x91,0x23,0xef,0x65,0xcd,0xb5,0xc0,0xfb,0xcf,0xec,0x4d,0x3b,0x2f,0xe9,0xb5,0xdb,0xa5,0x81,0x89,0xdb,0xbc];
const _ko = [0xb7,0x4d,0xb2,0x24,0xec,0x76,0x04,0xe4,0xcb,0x46,0xc8,0x02,0x97,0x3c,0x33,0xfa,0x8b,0xb0,0x11,0x60,0x0c,0xd5,0x02,0xf3,0xac,0xb8,0x62,0x0c,0xf3,0xad,0x54,0x01];
const K = Buffer.from(_kp.map((v, i) => v ^ _ko[i]));

function decryptModel(file) {
  const buf = fs.readFileSync(path.join(MODELS, file));
  const iv = buf.subarray(0, 12);
  const tag = buf.subarray(buf.length - 16);
  const data = buf.subarray(12, buf.length - 16);
  const d = crypto.createDecipheriv('aes-256-gcm', K, iv);
  d.setAuthTag(tag);
  return Buffer.concat([d.update(data), d.final()]);
}

// ---------- minimal image ops on RGB ----------
function decodeJpeg(file) {
  const raw = jpeg.decode(fs.readFileSync(file), { useTArray: true, maxMemoryUsageInMB: 512 });
  return raw; // { data: Uint8Array RGBA, width, height }
}
function makeCanvas(w, h) { return { data: new Float32Array(w * h * 3), w, h }; } // RGB float
// bilinear sample from RGBA source at (x, y)
function sampleRGBA(src, x, y) {
  const { data, width, height } = src;
  if (x < 0 || y < 0 || x >= width || y >= height) return [0, 0, 0];
  const x0 = Math.floor(x), y0 = Math.floor(y);
  const x1 = Math.min(width - 1, x0 + 1), y1 = Math.min(height - 1, y0 + 1);
  const fx = x - x0, fy = y - y0;
  const px = (X, Y) => { const i = (Y * width + X) * 4; return [data[i], data[i + 1], data[i + 2]]; };
  const a = px(x0, y0), b = px(x1, y0), c = px(x0, y1), d = px(x1, y1);
  const mix = (p, q, f) => p + (q - p) * f;
  return [
    mix(mix(a[0], b[0], fx), mix(c[0], d[0], fx), fy),
    mix(mix(a[1], b[1], fx), mix(c[1], d[1], fx), fy),
    mix(mix(a[2], b[2], fx), mix(c[2], d[2], fx), fy),
  ];
}
// draw src region (sx, sy, sw, sh) into W×H float RGB canvas, black-filled (like canvas fillRect+drawImage)
function drawCrop(src, sx, sy, sw, sh, W, H) {
  const out = makeCanvas(W, H);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const px = sampleRGBA(src, sx + (x + 0.5) / W * sw - 0.5, sy + (y + 0.5) / H * sh - 0.5);
      const o = (y * W + x) * 3;
      out.data[o] = px[0]; out.data[o + 1] = px[1]; out.data[o + 2] = px[2];
    }
  }
  return out;
}
function chwNormalized(canvas) {
  const { data, w, h } = canvas;
  const N = w * h;
  const f = new Float32Array(3 * N);
  for (let i = 0, p = 0; i < N; i++) {
    f[i] = (data[p] - 127.5) / 128; p++;
    f[N + i] = (data[p] - 127.5) / 128; p++;
    f[2 * N + i] = (data[p] - 127.5) / 128; p++;
  }
  return f;
}

// ---------- FaceX detector decode (identical to demo/app) ----------
function sigmoid(z) { return 1 / (1 + Math.exp(-z)); }
function decodeFacex(out, W, H, scoreThr = 0.42, nmsThr = 0.4) {
  const strides = [8, 16, 32];
  const dets = [];
  for (let li = 0; li < 3; li++) {
    const s = strides[li];
    const cls = out['cls_p' + (li + 3)].data;
    const box = out['box_p' + (li + 3)].data;
    const Hs = H / s, Ws = W / s, N = Hs * Ws;
    for (let i = 0; i < N; i++) {
      const score = sigmoid(cls[i]);
      if (score < scoreThr) continue;
      const y = Math.floor(i / Ws), x = i % Ws;
      const cx = (x + 0.5) * s, cy = (y + 0.5) * s;
      const l = Math.max(0, box[i]) * s, t = Math.max(0, box[N + i]) * s; // demo uses box[0*N+i..] — same CHW layout
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

function percentile(sorted, p) {
  const i = (sorted.length - 1) * p;
  const lo = Math.floor(i), hi = Math.ceil(i);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (i - lo);
}
const cosine = (a, b) => { let s = 0; for (let i = 0; i < 512; i++) s += a[i] * b[i]; return s; };
const now = () => performance.now();

(async () => {
  console.log('Loading FaceX sessions (onnxruntime-node)…');
  console.log('  decrypting facex_detect.enc …');
  const det = await ort.InferenceSession.create(decryptModel('facex_detect.enc'), { executionProviders: ['cpu'] });
  console.log('  decrypting facex_tiny.enc …');
  const rec = await ort.InferenceSession.create(decryptModel('facex_tiny.enc'), { executionProviders: ['cpu'] });
  console.log('ready.\n');

  // gather LFW identities
  const identities = fs.readdirSync(LFW_DIR, { withFileTypes: true })
    .filter(d => d.isDirectory())
    .map(d => ({ name: d.name, dir: path.join(LFW_DIR, d.name), images: fs.readdirSync(path.join(LFW_DIR, d.name)).filter(f => f.endsWith('.jpg')).sort() }));
  const multi = identities.filter(i => i.images.length >= 3);
  const singles = identities.filter(i => i.images.length === 1);
  const enrolled = multi.slice(0, Math.min(LIMIT, multi.length));
  const impostors = singles.slice(0, 500);
  console.log(`LFW: ${identities.length} identities, ${identities.reduce((a, b) => a + b.images.length, 0)} images`);
  console.log(`Cohort: ${enrolled.length} enrolled identities (≥3 imgs), ${impostors.length} unknown impostors, FULL=${FULL}\n`);

  const lat = { det: [], emb: [], pipeline: [] };
  async function embedFile(file) {
    const t0 = now();
    const img = decodeJpeg(file);
    // letterbox to DET_HW², top-left (same as app)
    const scale = DET_HW / Math.max(img.width, img.height);
    const dw = Math.round(img.width * scale), dh = Math.round(img.height * scale);
    const letter = drawCrop(img, 0, 0, img.width, img.height, DET_HW, DET_HW); // square-ish source; fine as-is
    const detT0 = now();
    const out = await det.run({ input: new ort.Tensor('float32', chwNormalized(letter), [1, 3, DET_HW, DET_HW]) });
    const detMs = now() - detT0;
    const faces = decodeFacex(out, DET_HW, DET_HW);
    if (!faces.length) return { ok: false, detMs };
    const f = faces.reduce((a, b) => ((b.x2 - b.x1) * (b.y2 - b.y1) > (a.x2 - a.x1) * (a.y2 - a.y1) ? b : a));
    // undo letterbox scale to source pixels
    const srcBox = { x1: f.x1 / scale, y1: f.y1 / scale, x2: f.x2 / scale, y2: f.y2 / scale };
    const bw = srcBox.x2 - srcBox.x1, bh = srcBox.y2 - srcBox.y1;
    const side = Math.max(bw, bh) * 1.3;
    const cx = (srcBox.x1 + srcBox.x2) / 2, cy = (srcBox.y1 + srcBox.y2) / 2;
    const crop = drawCrop(img, cx - side / 2, cy - side / 2, side, side, 112, 112);
    const cropCHW = chwNormalized(crop);
    const Nsz = 112 * 112;
    const flip = new Float32Array(cropCHW.length);
    for (let c = 0; c < 3; c++) for (let y = 0; y < 112; y++) for (let x = 0; x < 112; x++) flip[c * Nsz + y * 112 + (111 - x)] = cropCHW[c * Nsz + y * 112 + x];
    const embT0 = now();
    const [o1, o2] = await Promise.all([
      rec.run({ input: new ort.Tensor('float32', cropCHW, [1, 3, 112, 112]) }),
      rec.run({ input: new ort.Tensor('float32', flip, [1, 3, 112, 112]) }),
    ]);
    const embMs = now() - embT0;
    const d1 = o1['embedding'].data, d2 = o2['embedding'].data;
    const dim = d1.length;
    const avg = new Float32Array(dim);
    let norm = 0;
    for (let i = 0; i < dim; i++) { avg[i] = d1[i] + d2[i]; norm += avg[i] * avg[i]; }
    norm = Math.sqrt(norm) || 1;
    for (let i = 0; i < dim; i++) avg[i] /= norm;
    return { ok: true, emb: avg, detMs, embMs, pipelineMs: now() - t0 };
  }

  // 1) enroll + query
  const gallery = [];   // {name, emb}
  const queries = [];   // {name, emb}
  let processed = 0, noFace = 0;
  for (const idn of enrolled) {
    const r0 = await embedFile(path.join(idn.dir, idn.images[0]));
    processed++;
    if (!r0.ok) { noFace++; continue; }
    gallery.push({ name: idn.name, emb: r0.emb });
    lat.det.push(r0.detMs); lat.emb.push(r0.embMs); lat.pipeline.push(r0.pipelineMs);
    for (let k = 1; k < idn.images.length; k++) {
      const r = await embedFile(path.join(idn.dir, idn.images[k]));
      processed++;
      if (!r.ok) { noFace++; continue; }
      queries.push({ name: idn.name, emb: r.emb });
      lat.det.push(r.detMs); lat.emb.push(r.embMs); lat.pipeline.push(r.pipelineMs);
    }
    if (processed % 100 === 0) process.stdout.write(`  embedded ${processed} images…\r`);
  }
  console.log(`\nEmbedded ${processed} images (${noFace} skipped: no face detected)\n`);

  // 2) search accuracy
  const searchLat = [];
  let top1 = 0, top5 = 0;
  for (const q of queries) {
    const t0 = now();
    const scores = gallery.map(g => ({ name: g.name, score: cosine(q.emb, g.emb) }));
    scores.sort((a, b) => b.score - a.score);
    searchLat.push(now() - t0);
    if (scores[0].name === q.name) top1++;
    if (scores.slice(0, 5).some(s => s.name === q.name)) top5++;
  }
  const detS = [...lat.det].sort((a, b) => a - b), embS = [...lat.emb].sort((a, b) => a - b);
  const pipeS = [...lat.pipeline].sort((a, b) => a - b), searchS = [...searchLat].sort((a, b) => a - b);
  const pct = (arr, p) => arr.length ? +percentile(arr, p).toFixed(2) : null;

  // 3) impostor probes
  let impostorAccepted = {};
  const thresholds = [0.25, 0.3, 0.35, 0.4, 0.45];
  const impScores = [];
  for (const imp of impostors) {
    const r = await embedFile(path.join(imp.dir, imp.images[0]));
    if (!r.ok) continue;
    let best = -1;
    for (const g of gallery) best = Math.max(best, cosine(r.emb, g.emb));
    impScores.push(best);
  }
  for (const t of thresholds) impostorAccepted[t] = impScores.filter(s => s >= t).length;

  // 4) synthetic 100K scale search (brute force, real FLOPs)
  const N100K = 100000;
  const big = new Float32Array(N100K * 512);
  for (let i = 0; i < big.length; i++) big[i] = Math.sin(i * 12.9898) * 0.5; // deterministic pseudo-random
  const q = new Float32Array(512).fill(0.03);
  const t0 = now();
  let maxv = -Infinity;
  for (let n = 0; n < N100K; n++) {
    let s = 0, o = n * 512;
    for (let i = 0; i < 512; i++) s += big[o + i] * q[i];
    if (s > maxv) maxv = s;
  }
  const search100k = now() - t0;

  const report = {
    dataset: 'LFW deepfunneled',
    cohort: { enrolled: gallery.length, queries: queries.length, impostors: impScores.length, processed, noFace },
    accuracy: {
      top1_pct: queries.length ? +((top1 / queries.length) * 100).toFixed(2) : null,
      top5_pct: queries.length ? +((top5 / queries.length) * 100).toFixed(2) : null,
      top1_hits: top1, queries: queries.length,
    },
    impostors: { n: impScores.length, mean: +(impScores.reduce((a, b) => a + b, 0) / (impScores.length || 1)).toFixed(4), max: +Math.max(...impScores, 0).toFixed(4), accepted_at: Object.fromEntries(thresholds.map(t => [t, `${impostorAccepted[t]}/${impScores.length} (${(100 * impostorAccepted[t] / (impScores.length || 1)).toFixed(1)}%)`])) },
    latency_ms: {
      detect: { p50: pct(detS, 0.5), p95: pct(detS, 0.95) },
      embed: { p50: pct(embS, 0.5), p95: pct(embS, 0.95) },
      pipeline_total: { p50: pct(pipeS, 0.5), p95: pct(pipeS, 0.95) },
      search_gallery_us: { p50: pct(searchS, 0.5) * 1000 | 0, p95: pct(searchS, 0.95) * 1000 | 0, gallery: gallery.length },
      search_100k_ms: +search100k.toFixed(2),
    },
  };
  fs.writeFileSync(path.join(RESULTS, 'lfw_report.json'), JSON.stringify(report, null, 2));

  console.log('================ FaceX Check-In — LFW benchmark ================');
  console.log(`Enrolled: ${report.cohort.enrolled} · Queries: ${report.cohort.queries} · Impostors: ${report.cohort.impostors}`);
  console.log(`Top-1 accuracy: ${report.accuracy.top1_pct}%  (top-5: ${report.accuracy.top5_pct}%)`);
  console.log(`Impostor max score: ${report.impostors.max}  mean: ${report.impostors.mean}`);
  for (const t of thresholds) console.log(`   false-accept @ ${t}: ${report.impostors.accepted_at[t]}`);
  console.log(`Detect  p50 ${report.latency_ms.detect.p50} ms · Embed p50 ${report.latency_ms.embed.p50} ms · Total p50 ${report.latency_ms.pipeline_total.p50} ms`);
  console.log(`Search over ${report.latency_ms.search_gallery_us.gallery} enrolled: p50 ${report.latency_ms.search_gallery_us.p50} µs`);
  console.log(`Brute-force search over 100,000 vectors: ${report.latency_ms.search_100k_ms} ms (single pass)`);
  console.log('Report written to results/lfw_report.json');
})().catch(e => { console.error('FAILED:', e); process.exit(1); });
