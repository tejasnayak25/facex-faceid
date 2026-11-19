/* worker_sface.js — SFace inference in a Web Worker (off main thread).
   Handles: YuNet ONNX detection (640×640) → 5-point alignment → SFace embedding.
   Protocol:
     Main → Worker: { type: 'init' }
     Main → Worker: { type: 'process', frame: ArrayBuffer (RGBA), w, h }
     Worker → Main: { type: 'ready' }
     Worker → Main: { type: 'result', faces: [...], embeddings: [...], ms }
     Worker → Main: { type: 'error', message }
*/
importScripts('ort/ort.all.min.js');

// AlignFace (ArcFace 5-point similarity transform) — pure JS, works with OffscreenCanvas
function getSimilarityTransform(src, dst) {
  const n = src.length;
  let sx = 0, sy = 0, dx = 0, dy = 0;
  for (let i = 0; i < n; i++) { sx += src[i][0]; sy += src[i][1]; dx += dst[i][0]; dy += dst[i][1]; }
  sx /= n; sy /= n; dx /= n; dy /= n;
  let num1 = 0, num2 = 0, den = 0;
  for (let i = 0; i < n; i++) {
    const sxc = src[i][0] - sx, syc = src[i][1] - sy;
    const dxc = dst[i][0] - dx, dyc = dst[i][1] - dy;
    num1 += dxc * sxc + dyc * syc;
    num2 += dxc * syc - dyc * sxc;
    den += sxc * sxc + syc * syc;
  }
  const a = num1 / den, b = num2 / den;
  const tx = dx - a * sx + b * sy;
  const ty = dy - b * sx - a * sy;
  return [a, b, tx, ty];
}

function alignFace(srcCtx, srcW, srcH, kps, dstCtx) {
  const REF = [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]];
  const [a, b, tx, ty] = getSimilarityTransform(kps, REF);
  const det = a * a + b * b;
  const ai = a / det, bi = b / det;
  const txi = -(ai * tx + bi * ty);
  const tyi = (bi * tx - ai * ty);
  const srcData = srcCtx.getImageData(0, 0, srcW, srcH);
  const dstData = dstCtx.createImageData(112, 112);
  const src = srcData.data, dst = dstData.data;
  for (let dy = 0; dy < 112; dy++) {
    for (let dx = 0; dx < 112; dx++) {
      const sx = ai * dx - bi * dy + txi;
      const sy = bi * dx + ai * dy + tyi;
      const x0 = Math.floor(sx), y0 = Math.floor(sy);
      const x1 = x0 + 1, y1 = y0 + 1;
      const fx = sx - x0, fy = sy - y0;
      if (x0 >= 0 && x1 < srcW && y0 >= 0 && y1 < srcH) {
        const i00 = (y0 * srcW + x0) * 4;
        const i10 = (y0 * srcW + x1) * 4;
        const i01 = (y1 * srcW + x0) * 4;
        const i11 = (y1 * srcW + x1) * 4;
        const di = (dy * 112 + dx) * 4;
        for (let c = 0; c < 3; c++) {
          dst[di + c] = Math.round(
            src[i00 + c] * (1 - fx) * (1 - fy) +
            src[i10 + c] * fx * (1 - fy) +
            src[i01 + c] * (1 - fx) * fy +
            src[i11 + c] * fx * fy
          );
        }
        dst[di + 3] = 255;
      }
    }
  }
  dstCtx.putImageData(dstData, 0, 0);
  return dstData;
}

let yunetSess = null;
let sfaceSess = null;

async function init() {
  ort.env.wasm.wasmPaths = new URL('ort/', self.location.href).href;
  ort.env.wasm.numThreads = Math.min(4, Math.max(1, (self.navigator?.hardwareConcurrency || 4) - 1));
  ort.env.wasm.simd = true;
  ort.env.logLevel = 'error';
  const yunetBuf = await (await fetch('models/yunet_2023mar_int8.onnx')).arrayBuffer();
  const sfaceBuf = await (await fetch('models/sface_int8.onnx')).arrayBuffer();
  yunetSess = await ort.InferenceSession.create(yunetBuf, { executionProviders: ['wasm'] });
  try {
    sfaceSess = await ort.InferenceSession.create(sfaceBuf.slice(0), { executionProviders: ['webgpu', 'wasm'] });
  } catch (e) {
    sfaceSess = await ort.InferenceSession.create(sfaceBuf, { executionProviders: ['wasm'] });
  }
  self.postMessage({ type: 'ready' });
}

function detectYuNet(imgData, srcW, srcH) {
  const detSize = 640;
  const cv = new OffscreenCanvas(detSize, detSize);
  const ctx = cv.getContext('2d');
  ctx.fillStyle = '#000'; ctx.fillRect(0, 0, detSize, detSize);
  const scale = Math.min(detSize / srcW, detSize / srcH);
  const nw = Math.round(srcW * scale), nh = Math.round(srcH * scale);
  const dx = 0, dy = 0;
  // Draw from ImageData (we'll pass a canvas instead — but worker has no Image)
  // Actually we receive raw RGBA pixels; draw via putImageData on a temp canvas
  const srcCv = new OffscreenCanvas(srcW, srcH);
  const srcCtx = srcCv.getContext('2d');
  srcCtx.putImageData(new ImageData(new Uint8ClampedArray(imgData), srcW, srcH), 0, 0);
  ctx.drawImage(srcCv, dx, dy, nw, nh);
  const data = ctx.getImageData(0, 0, detSize, detSize).data;
  const N = detSize * detSize;
  const input = new Float32Array(3 * N);
  for (let i = 0, p = 0; i < N; i++) {
    input[i] = data[p]; input[N + i] = data[p + 1]; input[2 * N + i] = data[p + 2];
    p += 4;
  }
  // Run synchronously (ORT session.run is async but we await)
  return { input, scale, dx, dy, srcW, srcH, srcCtx, srcCv };
}

// We need async for sess.run, so make process async
async function processFrame(rgba, w, h) {
  const t0 = performance.now();
  const detSize = 640;
  const srcCv = new OffscreenCanvas(w, h);
  const srcCtx = srcCv.getContext('2d');
  srcCtx.putImageData(new ImageData(new Uint8ClampedArray(rgba), w, h), 0, 0);

  // YuNet detection
  const cv = new OffscreenCanvas(detSize, detSize);
  const ctx = cv.getContext('2d');
  ctx.fillStyle = '#000'; ctx.fillRect(0, 0, detSize, detSize);
  const scale = Math.min(detSize / w, detSize / h);
  const nw = Math.round(w * scale), nh = Math.round(h * scale);
  const dx = 0, dy = 0;
  ctx.drawImage(srcCv, dx, dy, nw, nh);
  const imgData = ctx.getImageData(0, 0, detSize, detSize).data;
  const N = detSize * detSize;
  const input = new Float32Array(3 * N);
  for (let i = 0, p = 0; i < N; i++) {
    input[i] = imgData[p]; input[N + i] = imgData[p + 1]; input[2 * N + i] = imgData[p + 2];
    p += 4;
  }
  const t = new ort.Tensor('float32', input, [1, 3, detSize, detSize]);
  const o = await yunetSess.run({ input: t });

  const strides = [8, 16, 32];
  const dets = [];
  for (let si = 0; si < 3; si++) {
    const s = strides[si];
    const cls = o['cls_' + s].data;
    const obj = o['obj_' + s].data;
    const bbox = o['bbox_' + s].data;
    const kps = o['kps_' + s].data;
    const cols = detSize / s, rows = detSize / s;
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const idx = r * cols + c;
        const clsScore = Math.min(1, Math.max(0, cls[idx]));
        const objScore = Math.min(1, Math.max(0, obj[idx]));
        const score = Math.sqrt(clsScore * objScore);
        if (score < 0.5) continue;
        const cx = (c + bbox[idx * 4 + 0]) * s;
        const cy = (r + bbox[idx * 4 + 1]) * s;
        const bw = Math.exp(bbox[idx * 4 + 2]) * s;
        const bh = Math.exp(bbox[idx * 4 + 3]) * s;
        const x1 = (cx - bw / 2 - dx) / scale;
        const y1 = (cy - bh / 2 - dy) / scale;
        const x2 = (cx + bw / 2 - dx) / scale;
        const y2 = (cy + bh / 2 - dy) / scale;
        const kpsArr = [];
        for (let n = 0; n < 5; n++) {
          const kx = (kps[idx * 10 + 2 * n] + c) * s - dx;
          const ky = (kps[idx * 10 + 2 * n + 1] + r) * s - dy;
          kpsArr.push([kx / scale, ky / scale]);
        }
        dets.push({ x1, y1, x2, y2, score, kps: kpsArr.flat() });
      }
    }
  }
  // NMS
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
      if (inter / (u || 1e-6) > 0.3) dets.splice(i, 1);
    }
  }

  // Align + embed each face
  const faces = kept.map(f => ({ x1: f.x1, y1: f.y1, x2: f.x2, y2: f.y2, score: f.score }));
  const embeddings = [];
  const sfaceCv = new OffscreenCanvas(112, 112);
  const sfaceCtx = sfaceCv.getContext('2d');
  for (const f of kept) {
    const kps5 = [];
    for (let i = 0; i < 5; i++) kps5.push([f.kps[i * 2], f.kps[i * 2 + 1]]);
    alignFace(srcCtx, w, h, kps5, sfaceCtx);
    const imgData = sfaceCtx.getImageData(0, 0, 112, 112).data;
    const M = 112 * 112;
    const input = new Float32Array(3 * M);
    for (let i = 0, p = 0; i < M; i++) {
      input[i] = imgData[p]; input[M + i] = imgData[p + 1]; input[2 * M + i] = imgData[p + 2];
      p += 4;
    }
    const t = new ort.Tensor('float32', input, [1, 3, 112, 112]);
    const o = await sfaceSess.run({ data: t });
    embeddings.push(Array.from(o['fc1'].data));
  }
  return { faces, embeddings, ms: performance.now() - t0 };
}

self.onmessage = async (e) => {
  const { type } = e.data;
  if (type === 'init') {
    try {
      await init();
    } catch (err) {
      self.postMessage({ type: 'error', message: String(err && err.message || err) });
    }
    return;
  }
  if (type === 'process') {
    try {
      const { frame, w, h } = e.data;
      const rgba = new Uint8ClampedArray(frame);
      const result = await processFrame(rgba, w, h);
      // Transfer embeddings as-is (structured clone)
      self.postMessage({ type: 'result', faces: result.faces, embeddings: result.embeddings, ms: result.ms });
    } catch (err) {
      self.postMessage({ type: 'error', message: String(err && err.message || err) });
    }
    return;
  }
};
