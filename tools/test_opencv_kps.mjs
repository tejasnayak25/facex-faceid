/* Test: use OpenCV's exact landmarks in JS alignFace */
import { chromium } from 'playwright-core';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

const browser = await chromium.launch({ channel: 'msedge', headless: true, args: ['--no-sandbox'] });
const page = await browser.newPage();
page.on('pageerror', e => console.log('pageerror:', e.message));
await page.goto('http://127.0.0.1:8080/', { waitUntil: 'load', timeout: 30000 });
await page.waitForFunction(() => /press Start Camera|failed/.test(document.querySelector('#statState').textContent), null, { timeout: 90000 });

const r = await page.evaluate(async () => {
  const img = await new Promise((res, rej) => { const i = new Image(); i.onload = () => res(i); i.onerror = rej; i.src = '/data/lfw-deepfunneled/George_W_Bush/George_W_Bush_0001.jpg'; });
  const srcW = img.naturalWidth, srcH = img.naturalHeight;
  const srcCv = document.createElement('canvas');
  srcCv.width = srcW; srcCv.height = srcH;
  const srcCtx = srcCv.getContext('2d', { willReadFrequently: true });
  srcCtx.drawImage(img, 0, 0);
  const sfaceCv = document.createElement('canvas');
  sfaceCv.width = 112; sfaceCv.height = 112;
  const sfaceCtx = sfaceCv.getContext('2d', { willReadFrequently: true });
  // OpenCV's landmarks (re, le, nt, rcm, lcm) = already ArcFace order [left_eye, right_eye, nose, left_mouth, right_mouth]
  const opencvKps = [[103.9, 111.0], [147.8, 111.9], [121.8, 134.8], [108.2, 154.0], [144.7, 154.3]];
  alignFace(srcCtx, srcW, srcH, opencvKps, sfaceCtx);
  return sfaceCv.toDataURL('image/jpeg', 0.95);
});
fs.writeFileSync(path.join(root, 'data', 'aligned_bush1_opencvkps.png'), Buffer.from(r.split(',')[1], 'base64'));
await browser.close();
console.log('saved');
// Compare to Python's aligned crop
const a = (await import('sharp')).default ? null : null;
const { execSync } = await import('child_process');
console.log('done');
