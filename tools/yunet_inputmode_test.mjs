/* Test different input preprocessing for YuNet ONNX */
import { chromium } from 'playwright-core';
const browser = await chromium.launch({ channel: 'msedge', headless: true, args: ['--no-sandbox'] });
const page = await browser.newPage();
page.on('pageerror', e => console.log('pageerror:', e.message));
page.on('console', m => { if (m.type() === 'error') console.log('console error:', m.text()); });
await page.goto('http://127.0.0.1:8080/', { waitUntil: 'load', timeout: 30000 });
await page.waitForFunction(() => /press Start Camera|failed/.test(document.querySelector('#statState').textContent), null, { timeout: 90000 });

const r = await page.evaluate(async () => {
  const sess = await ort.InferenceSession.create(await (await fetch('models/yunet_2023mar_int8.onnx')).arrayBuffer(), { executionProviders: ['wasm'] });
  const img = await new Promise((res, rej) => { const i = new Image(); i.onload = () => res(i); i.onerror = rej; i.src = '/data/lfw-deepfunneled/George_W_Bush/George_W_Bush_0001.jpg'; });
  const srcW = img.naturalWidth, srcH = img.naturalHeight;
  const detSize = 640;
  const results = {};
  for (const mode of ['letterbox', 'stretch']) {
    const cv = document.createElement('canvas');
    cv.width = detSize; cv.height = detSize;
    const ctx = cv.getContext('2d', { willReadFrequently: true });
    ctx.fillStyle = '#000'; ctx.fillRect(0, 0, detSize, detSize);
    let scale, dx, dy, nw, nh;
    if (mode === 'letterbox') {
      scale = Math.min(detSize / srcW, detSize / srcH);
      nw = Math.round(srcW * scale); nh = Math.round(srcH * scale);
      dx = (detSize - nw) / 2; dy = (detSize - nh) / 2;
      ctx.drawImage(img, dx, dy, nw, nh);
    } else {
      scale = 1; dx = 0; dy = 0; nw = detSize; nh = detSize;
      ctx.drawImage(img, 0, 0, detSize, detSize);
    }
    const imgData = ctx.getImageData(0, 0, detSize, detSize).data;
    const N = detSize * detSize;
    const input = new Float32Array(3 * N);
    for (let i = 0, p = 0; i < N; i++) {
      input[i] = imgData[p]; input[N + i] = imgData[p + 1]; input[2 * N + i] = imgData[p + 2];
      p += 4;
    }
    const t = new ort.Tensor('float32', input, [1, 3, detSize, detSize]);
    const o = await sess.run({ input: t });
    const strides = [8, 16, 32];
    let bestScore = 0, bestKps = null;
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
          if (score > bestScore) {
            bestScore = score;
            bestKps = [];
            for (let n = 0; n < 5; n++) {
              bestKps.push([(kps[idx * 10 + 2 * n] + c) * s, (kps[idx * 10 + 2 * n + 1] + r) * s]);
            }
          }
        }
      }
    }
    results[mode] = { bestScore: +bestScore.toFixed(4), bestKps: bestKps ? bestKps.map(v => v.map(x => +x.toFixed(1))) : null };
  }
  return results;
});
console.log(JSON.stringify(r, null, 2));
await browser.close();
