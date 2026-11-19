/* build_db — walk a roster of named folders, embed every student with the
   exact SDK pipeline the booth uses, and write an Import-DB JSON.

   Usage:
     node tools/build_db.mjs --roster "D:\path\to\roster" [--out import_db.json] [--max-per-student 5]

   Roster layout: one folder per student, folder name "ROLL_Name" (e.g.
   "CS2023-0142_Aditi_Sharma"), any jpg/png inside. Folders without "_" get
   their folder name as both roll and name. Image extensions: .jpg .jpeg .png .webp

   Then in the booth: Database tab → Import DB → pick the generated file.
*/
import * as fs from 'node:fs';
import * as path from 'node:path';
import * as http from 'node:http';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright-core';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const args = process.argv.slice(2);
const opt = (name, def) => { const i = args.indexOf('--' + name); return i >= 0 ? args[i + 1] : def; };
const ROSTER = path.resolve(opt('roster', ''));
const OUT = opt('out', path.join(root, 'import_db.json'));
const MAX_PER = +opt('max-per-student', 5);
const PORT = 8123;

if (!ROSTER || !fs.existsSync(ROSTER) || !fs.statSync(ROSTER).isDirectory()) {
  console.error('Usage: node tools/build_db.mjs --roster "D:\\path\\to\\roster" [--out import_db.json] [--max-per-student 5]');
  process.exit(1);
}

const IMG_EXT = new Set(['.jpg', '.jpeg', '.png', '.webp']);
const MIME = { '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png', '.webp': 'image/webp' };

function parseFolder(name) {
  const i = name.indexOf('_');
  if (i > 0) return { roll: name.slice(0, i).trim(), name: name.slice(i + 1).replace(/[_-]+/g, ' ').trim() };
  return { roll: name, name: name.replace(/[_-]+/g, ' ').trim() };
}

const students = [];
for (const dir of fs.readdirSync(ROSTER, { withFileTypes: true })) {
  if (!dir.isDirectory()) continue;
  const dirPath = path.join(ROSTER, dir.name);
  const files = fs.readdirSync(dirPath)
    .filter(f => IMG_EXT.has(path.extname(f).toLowerCase()))
    .sort();
  if (!files.length) continue;
  // URLs relative to roster root as served by the built-in file server
  const urls = files.map(f => `http://127.0.0.1:${PORT}/` + [dir.name, f].map(encodeURIComponent).join('/'));
  const { roll, name } = parseFolder(dir.name);
  students.push({ roll, name, images: urls });
}
if (!students.length) { console.error('No student folders with images found in', ROSTER); process.exit(1); }
console.log(`Found ${students.length} students · ${students.reduce((a, s) => a + s.images.length, 0)} photos (max ${MAX_PER} used per student)`);

// tiny static file server for the roster folder (with CORS so canvas isn't tainted)
const serverRoot = ROSTER.endsWith(path.sep) ? ROSTER : ROSTER + path.sep;
const server = http.createServer((req, res) => {
  try {
    let rel = decodeURIComponent(new URL(req.url, 'http://x').pathname).replace(/^\/+/, '');
    const abs = path.normalize(path.join(ROSTER, rel));
    const prefix = path.normalize(ROSTER).toLowerCase() + path.sep;
    if (!abs.toLowerCase().startsWith(prefix) || !fs.existsSync(abs) || !fs.statSync(abs).isFile()) { res.writeHead(404); return res.end('nf'); }
    res.writeHead(200, { 'Content-Type': MIME[path.extname(abs).toLowerCase()] || 'application/octet-stream', 'Access-Control-Allow-Origin': '*' });
    fs.createReadStream(abs).pipe(res);
  } catch { res.writeHead(500); res.end(); }
});
await new Promise(r => server.listen(PORT, '127.0.0.1', r));
console.log(`serving roster on http://127.0.0.1:${PORT}/`);

const origin = `http://127.0.0.1:8080`;
const browser = await chromium.launch({ channel: 'msedge', headless: true, args: ['--no-sandbox', '--disable-gpu'] });
const page = await browser.newPage();
page.on('pageerror', e => console.log('pageerror:', e.message));
await page.goto(origin + '/build_db.worker.html', { waitUntil: 'load', timeout: 30000 });
await page.waitForFunction('window.__done === true', null, { timeout: 120000 });
console.log('worker ready — embedding roster (this takes a while for large rosters)…');

const result = await page.evaluate(async ({ students, maxPer }) => window.__run(students, maxPer), { students, maxPer: MAX_PER });
await browser.close();
server.close();

const out = {
  version: 1,
  exportedAt: Date.now(),
  source: 'build_db',
  students: result.students,
  logs: [],
};
fs.writeFileSync(OUT, JSON.stringify(out));
console.log(`\n✓ wrote ${OUT}`);
console.log(`  enrolled: ${result.students.length}`);
console.log(`  failed  : ${result.failed.length}${result.failed.length ? ' → ' + result.failed.map(f => f.name).join(', ') : ''}`);
console.log(`\nNow: open the booth → Database tab → Import DB → pick ${path.basename(OUT)}`);
