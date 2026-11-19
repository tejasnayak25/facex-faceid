/* Build pairs JSON (fold 0) from official lfw_pairs.txt, then run sdk_lfw.html headless */
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright-core';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const lines = fs.readFileSync(path.join(root, '..', 'facex', 'lfw_pairs.txt'), 'utf8').trim().split('\n');
const [nFolds, nPerFold] = lines[0].trim().split(/\s+/).map(Number);
const pairs = [];
let idx = 1;
for (let fold = 0; fold < nFolds; fold++) {
  for (let k = 0; k < nPerFold; k++, idx++) {
    const [n, i, j] = lines[idx].trim().split(/\s+/);
    pairs.push([n, +i, n, +j, 1]);
  }
  for (let k = 0; k < nPerFold; k++, idx++) {
    const [n1, i, n2, j] = lines[idx].trim().split(/\s+/);
    pairs.push([n1, +i, n2, +j, 0]);
  }
}
// fold 0 only for the SDK eval (600 pairs)
fs.writeFileSync(path.join(root, 'wfpairs.json'), JSON.stringify(pairs.slice(0, nPerFold * 2)));
console.log(`wrote wfpairs.json (${nPerFold * 2} pairs, fold 0)`);

const browser = await chromium.launch({ channel: 'msedge', headless: true, args: ['--no-sandbox'] });
const page = await browser.newPage();
page.on('pageerror', e => console.log('pageerror:', e.message));
page.on('console', m => { if (m.type() === 'error') console.log('console:', m.text()); });
await page.goto('http://127.0.0.1:8080/sdk_lfw.html', { waitUntil: 'load', timeout: 30000 });
await page.waitForFunction('window.__done === true', null, { timeout: 30 * 60 * 1000 });
console.log(await page.textContent('#out'));
await browser.close();
