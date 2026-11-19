/* SFace engine: enroll Bush_0001, query Bush_0003 + Powell_0001 */
import { chromium } from 'playwright-core';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const LFW = path.join(root, 'data', 'lfw-deepfunneled');
const bush1 = fs.readFileSync(path.join(LFW, 'George_W_Bush', 'George_W_Bush_0001.jpg'));
const bush3 = fs.readFileSync(path.join(LFW, 'George_W_Bush', 'George_W_Bush_0003.jpg'));
const powell1 = fs.readFileSync(path.join(LFW, 'Colin_Powell', 'Colin_Powell_0001.jpg'));

const browser = await chromium.launch({ channel: 'msedge', headless: true, args: ['--no-sandbox'] });
const page = await browser.newPage({ viewport: { width: 1280, height: 720 } });
page.on('console', m => { if (m.type() === 'error') console.log('console error:', m.text()); });
page.on('pageerror', e => console.log('pageerror:', e.message));
await page.goto('http://127.0.0.1:8080/', { waitUntil: 'load', timeout: 30000 });
await page.waitForFunction(() => /press Start Camera|failed/.test(document.querySelector('#statState').textContent), null, { timeout: 90000 });
// switch to SFace engine before enrollment
await page.click('[data-tab="settings"]');
await page.selectOption('#engineSelect', 'sface');
await page.waitForFunction(() => /SFace \(OpenCV/.test(document.querySelector('#toast').textContent || '') || /press Start Camera|scanning|waiting for face/.test(document.querySelector('#statState').textContent), null, { timeout: 120000 });
console.log('engine after switch:', await page.evaluate(() => state.engine));
await page.waitForTimeout(2000);

await page.click('[data-tab="enroll"]');
await page.setInputFiles('#batchPhotos', { name: 'TEST-0001_George_W_Bush.jpg', mimeType: 'image/jpeg', buffer: bush1 });
await page.click('#btnBatchEnroll');
await page.waitForFunction(() => /Done/.test(document.querySelector('#batchStatus').textContent || ''), null, { timeout: 180000 });
console.log('enroll:', await page.textContent('#batchStatus'));

await page.click('[data-tab="settings"]');
for (const [label, buf] of [['same-person (Bush 0003)', bush3], ['other (Powell 0001)', powell1]]) {
  const prev = await page.textContent('#testPhotoResult');
  await page.setInputFiles('#testPhoto', { name: label.replace(/\W/g, '_') + '.jpg', mimeType: 'image/jpeg', buffer: buf });
  await page.waitForFunction((old) => { const t = document.querySelector('#testPhotoResult').textContent || ''; return t !== old && !/Processing/.test(t); }, prev, { timeout: 180000 });
  console.log(`${label}:`, (await page.textContent('#testPhotoResult')).trim());
}
await browser.close();
console.log('SFACE SMOKE DONE');
