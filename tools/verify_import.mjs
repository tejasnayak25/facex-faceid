/* End-to-end: import the built DB into the booth UI, then query a held-out photo */
import { chromium } from 'playwright-core';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const importFile = path.join(root, 'data', 'import_db_demo.json');
const queryPath = path.join(root, 'data', 'lfw-deepfunneled', 'George_W_Bush', 'George_W_Bush_0006.jpg');

const browser = await chromium.launch({ channel: 'msedge', headless: true, args: ['--no-sandbox'] });
const page = await browser.newPage({ viewport: { width: 1280, height: 720 } });
page.on('pageerror', e => console.log('pageerror:', e.message));
await page.goto('http://127.0.0.1:8080/', { waitUntil: 'load', timeout: 30000 });
await page.waitForFunction(() => /press Start Camera|failed/.test(document.querySelector('#statState').textContent), null, { timeout: 90000 });

await page.click('[data-tab="database"]');
await page.setInputFiles('#importFile', { name: 'import_db_demo.json', mimeType: 'application/json', buffer: fs.readFileSync(importFile) });
await page.waitForFunction(() => document.querySelector('#statDb').textContent === '3', null, { timeout: 60000 });
console.log('db count after import:', await page.textContent('#statDb'));
const rows = await page.locator('#dbList .db-item .n').allTextContents();
console.log('roster rows:', JSON.stringify(rows.map(s => s.trim())));

await page.click('[data-tab="settings"]');
const prev = await page.textContent('#testPhotoResult');
await page.setInputFiles('#testPhoto', { name: 'query.jpg', mimeType: 'image/jpeg', buffer: fs.readFileSync(queryPath) });
await page.waitForFunction((old) => { const t = document.querySelector('#testPhotoResult').textContent || ''; return t !== old && !/Processing/.test(t); }, prev, { timeout: 120000 });
console.log('query (held-out Bush_0006):', (await page.textContent('#testPhotoResult')).trim());
await browser.close();
console.log('IMPORT FLOW OK');
