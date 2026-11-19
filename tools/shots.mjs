/* Visual pass over all four tabs */
import { chromium } from 'playwright-core';
const browser = await chromium.launch({ channel: 'msedge', headless: true, args: ['--no-sandbox'] });
const page = await browser.newPage({ viewport: { width: 1500, height: 800 } });
await page.goto('http://127.0.0.1:8080/', { waitUntil: 'load', timeout: 30000 });
await page.waitForFunction(() => /press Start Camera|failed/.test(document.querySelector('#statState').textContent), null, { timeout: 60000 });
for (const [tab, file] of [['checkin', 'shot_checkin.png'], ['enroll', 'shot_enroll.png'], ['database', 'shot_database.png'], ['settings', 'shot_settings.png']]) {
  await page.click(`[data-tab="${tab}"]`);
  await page.waitForTimeout(250);
  await page.screenshot({ path: file });
  console.log('saved', file);
}
await browser.close();
