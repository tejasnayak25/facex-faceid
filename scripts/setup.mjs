/* scripts/setup.mjs — one-command setup for a fresh clone.
   1. Copies the onnxruntime-web runtime from node_modules into ort/
   2. Downloads the model weights (FaceX GitHub release + facex-engine/facex repo)
   3. Verifies sizes so you know everything is present and correct

   Usage:  npm run setup
*/
import * as fs from 'node:fs';
import * as path from 'node:path';
import * as https from 'node:https';
import * as http from 'node:http';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const dist = path.join(root, 'node_modules', 'onnxruntime-web', 'dist');
const ortDir = path.join(root, 'ort');
const modelsDir = path.join(root, 'models');
const ORT_FILES = ['ort.min.js', 'ort-wasm-simd-threaded.mjs', 'ort-wasm-simd-threaded.wasm', 'ort-wasm-simd-threaded.jsep.mjs', 'ort-wasm-simd-threaded.jsep.wasm'];

function fetchToFile(url, dest, expectedBytes) {
  return new Promise((resolve, reject) => {
    const get = (u, redirects = 0) => {
      const client = u.startsWith('https:') ? https : http;
      client.get(u, { headers: { 'User-Agent': 'facex-checkin-setup' } }, (res) => {
        if ([301, 302, 303, 307, 308].includes(res.statusCode) && res.headers.location && redirects < 6) {
          res.resume();
          return get(new URL(res.headers.location, u).href, redirects + 1);
        }
        if (res.statusCode !== 200) { res.resume(); return reject(new Error(`HTTP ${res.statusCode} for ${u}`)); }
        const total = +(res.headers['content-length'] || 0);
        const tmp = dest + '.part';
        const out = fs.createWriteStream(tmp);
        let done = 0;
        res.on('data', (chunk) => { done += chunk.length; process.stdout.write(`\r  ${(done / 1e6).toFixed(1)} MB${total ? ` / ${(total / 1e6).toFixed(1)} MB` : ''}`); });
        res.on('end', () => { out.end(() => { fs.renameSync(tmp, dest); process.stdout.write('\n'); resolve(done); }); });
        res.on('error', reject);
        out.on('error', reject);
        res.pipe(out);
      }).on('error', reject);
    };
    get(url);
  }).then((bytes) => {
    if (expectedBytes && bytes !== expectedBytes) throw new Error(`size mismatch for ${dest}: expected ${expectedBytes}, got ${bytes}`);
    return bytes;
  });
}

console.log('1/3  Copying onnxruntime-web runtime from node_modules → ort/');
if (!fs.existsSync(dist)) {
  console.error('     node_modules/onnxruntime-web not found — run `npm install` first (or use `npm run setup` which does it for you).');
  process.exit(1);
}
fs.mkdirSync(ortDir, { recursive: true });
for (const f of ORT_FILES) {
  fs.copyFileSync(path.join(dist, f), path.join(ortDir, f));
  console.log(`     ✓ ort/${f}`);
}

console.log('2/3  Downloading model weights');
fs.mkdirSync(modelsDir, { recursive: true });
const MODELS = [
  { file: 'edgeface_xs_fp32.bin', url: 'https://github.com/facex-engine/facex/releases/download/v1.0.0/edgeface_xs_fp32.bin', bytes: 7133584 },
  { file: 'facex_detect.enc',     url: 'https://raw.githubusercontent.com/facex-engine/facex/main/docs/demo/facex_detect.enc', bytes: 410722 },
  { file: 'facex_tiny.enc',       url: 'https://raw.githubusercontent.com/facex-engine/facex/main/docs/demo/facex_tiny.enc', bytes: 1809249 },
  { file: 'facex_nano.enc',       url: 'https://raw.githubusercontent.com/facex-engine/facex/main/docs/demo/facex_nano.enc', bytes: 802405 },
  { file: 'minifasnet_v2_27.enc', url: 'https://raw.githubusercontent.com/facex-engine/facex/main/docs/demo/minifasnet_v2_27.enc', bytes: 1744879 },
  { file: 'minifasnet_v1se_40.enc', url: 'https://raw.githubusercontent.com/facex-engine/facex/main/docs/demo/minifasnet_v1se_40.enc', bytes: 1744275 },
];
for (const m of MODELS) {
  const dest = path.join(modelsDir, m.file);
  if (fs.existsSync(dest) && fs.statSync(dest).size === m.bytes) { console.log(`     ✓ ${m.file} (cached)`); continue; }
  process.stdout.write(`     ↓ ${m.file}\n`);
  await fetchToFile(m.url, dest, m.bytes);
  console.log(`     ✓ ${m.file}`);
}

console.log('3/3  Checking vendored files');
const required = [
  ...ORT_FILES.map(f => path.join('ort', f)),
  ...MODELS.map(m => path.join('models', m.file)),
  'models/yunet_fp32.bin',   // committed to the repo (OpenCV Zoo-derived, Apache 2.0)
];
let ok = true;
for (const rel of required) {
  const present = fs.existsSync(path.join(root, rel));
  console.log(`     ${present ? '✓' : '✗ MISSING'} ${rel}`);
  if (!present) ok = false;
}
console.log(ok ? '\nDone. Run:  python -m http.server 8080  →  http://localhost:8080' : '\nSetup finished with MISSING files — see above.');
process.exit(ok ? 0 : 1);
