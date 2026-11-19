/* scripts/serve.mjs — dev server with COOP/COEP headers so SharedArrayBuffer is
   available, enabling multi-threaded WASM (onnxruntime-web + the FaceX SDK pool).

   Usage:  npm run serve   →  http://localhost:8080

   Plain `python -m http.server` works for the booth but ships no COOP/COEP, so
   ort.env.wasm.numThreads must stay at 1 (single-threaded, ~2-4x slower).
*/
import * as http from 'node:http';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const port = +(process.env.PORT || 8080);
const MIME = {
  '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8', '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8', '.wasm': 'application/wasm',
  '.bin': 'application/octet-stream', '.enc': 'application/octet-stream',
  '.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
  '.webp': 'image/webp', '.svg': 'image/svg+xml', '.ico': 'image/x-icon',
  '.txt': 'text/plain; charset=utf-8', '.md': 'text/markdown; charset=utf-8',
};

const server = http.createServer((req, res) => {
  const url = new URL(req.url, 'http://x');
  let rel = decodeURIComponent(url.pathname).replace(/^\/+/, '');
  if (rel === '' || rel.endsWith('/')) rel += 'index.html';
  const abs = path.normalize(path.join(root, rel));
  const prefix = path.normalize(root).toLowerCase() + path.sep;
  if (!abs.toLowerCase().startsWith(prefix) || !fs.existsSync(abs) || !fs.statSync(abs).isFile()) {
    res.writeHead(404, { 'Content-Type': 'text/plain' });
    return res.end('not found');
  }
  // Cross-origin isolation — unlocks SharedArrayBuffer for threaded WASM.
  res.writeHead(200, {
    'Content-Type': MIME[path.extname(abs).toLowerCase()] || 'application/octet-stream',
    'Cross-Origin-Opener-Policy': 'same-origin',
    'Cross-Origin-Embedder-Policy': 'require-corp',
    'Cross-Origin-Resource-Policy': 'same-origin',
    'Cache-Control': 'no-cache',
  });
  fs.createReadStream(abs).pipe(res);
});

server.listen(port, '127.0.0.1', () => {
  console.log(`FaceX Check-In dev server: http://localhost:${port}`);
  console.log('COOP/COEP enabled → multi-threaded WASM available.');
  console.log('Press Ctrl+C to stop.');
});
