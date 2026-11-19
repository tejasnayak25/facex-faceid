/* Headless end-to-end smoke test of the offline check-in booth.
   Uses Edge (fake camera) so the real pipeline runs: decrypt → detect → embed → match.
   Steps:
     1. load page, wait for models to decrypt+load
     2. click Start Camera (fake device), let the booth loop run
     3. enroll ONE student from an LFW photo via the hidden batch-enroll input
     4. run the SAME pipeline via "Test with image" using another photo of the same person
        → expect a match banner-style result
*/
import { chromium } from 'playwright-core';
import path from 'node:path';
import fs from 'node:fs';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const LFW = path.join(root, 'data', 'lfw-deepfunneled');
const pick = (idn, i) => path.join(LFW, idn, `${idn}_${String(i).padStart(4, '0')}.jpg`);
// two different photos of the same person + one of someone else
const enrollPhoto = pick('George_W_Bush', 1);
const queryPhoto = pick('George_W_Bush', 3);
const otherPhoto = pick('Colin_Powell', 1);
for (const f of [enrollPhoto, queryPhoto, otherPhoto]) if (!fs.existsSync(f)) throw new Error('missing ' + f);

const consoleErrors = [];
const browser = await chromium.launch({
  channel: 'msedge',
  headless: true,
  args: [
    '--use-fake-device-for-media-stream',
    '--use-fake-ui-for-media-stream',
    '--no-sandbox',
    '--disable-gpu',           // wasm does the heavy lifting anyway
    '--autoplay-policy=no-user-gesture-required',
  ],
});
const ctx = await browser.newContext({ viewport: { width: 1280, height: 720 }, permissions: ['camera'] });
const page = await ctx.newPage();
page.on('pageerror', e => consoleErrors.push('pageerror: ' + e.message));
page.on('console', m => { if (m.type() === 'error') consoleErrors.push('console: ' + m.text()); else if (/testPhoto|\[pf\]/.test(m.text())) console.log('log:', m.text()); });

await page.goto('http://127.0.0.1:8080/', { waitUntil: 'load', timeout: 30000 });
await page.waitForFunction(() => /press Start Camera|scanning|model load failed/.test(document.querySelector('#statState').textContent), null, { timeout: 60000 });
console.log('model state:', await page.textContent('#statState'));
console.log('model load time:', await page.textContent('#statLoad'));
console.log('toast:', await page.textContent('#toast'));
console.log('console errors so far:', consoleErrors);
if ((await page.textContent('#statState')).includes('failed')) throw new Error('model load failed — see diagnostics above');

// camera start (fake device)
await page.click('#btnStart');
await page.waitForTimeout(4000);
console.log('after start — state:', await page.textContent('#statState'), '· frame ms:', await page.textContent('#statMs'));

// enroll one student via batch photo (runs decrypt→detect→embed→IndexedDB)
await page.click('[data-tab="enroll"]');
await page.setInputFiles('#batchPhotos', { name: 'TEST-0001_George_W_Bush.jpg', mimeType: 'image/jpeg', buffer: fs.readFileSync(enrollPhoto) });
await page.click('#btnBatchEnroll');
await page.waitForFunction(() => /Done/.test(document.querySelector('#batchStatus').textContent || ''), null, { timeout: 120000 });
console.log('batch enroll:', await page.textContent('#batchStatus'));
console.log('db count:', await page.textContent('#statDb'));

// same-person test photo → should MATCH (switch to Settings tab first)
await page.click('[data-tab="settings"]');
let prev = await page.textContent('#testPhotoResult');
await page.setInputFiles('#testPhoto', { name: 'q.jpg', mimeType: 'image/jpeg', buffer: fs.readFileSync(queryPhoto) });
await page.waitForFunction((old) => { const t = document.querySelector('#testPhotoResult').textContent || ''; return t !== old && !/Processing/.test(t); }, prev, { timeout: 120000 });
console.log('same-person query:', (await page.textContent('#testPhotoResult')).trim());

// different-person test photo → should NOT match
prev = await page.textContent('#testPhotoResult');
await page.setInputFiles('#testPhoto', { name: 'q2.jpg', mimeType: 'image/jpeg', buffer: fs.readFileSync(otherPhoto) });
await page.waitForFunction((old) => { const t = document.querySelector('#testPhotoResult').textContent || ''; return t !== old && !/Processing/.test(t); }, prev, { timeout: 120000 });
console.log('other-person query:', (await page.textContent('#testPhotoResult')).trim());

// enable anti-spoof: decrypt + load both MiniFASNet sessions, watch for errors
await page.click('#spoofToggle');
await page.waitForFunction(() => /Anti-spoof ON/.test(document.querySelector('#toast').textContent || ''), null, { timeout: 120000 }).catch(() => {});
await page.waitForTimeout(4000);
console.log('spoof toggle checked:', await page.isChecked('#spoofToggle'), '· state:', await page.textContent('#statState'));

console.log('console errors:', consoleErrors.length ? consoleErrors : 'none');
await browser.close();
console.log('SMOKE TEST DONE');
