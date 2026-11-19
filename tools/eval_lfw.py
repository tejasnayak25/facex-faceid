"""Ground-truth LFW verification eval replicating FaceX's own prepare_lfw.py exactly.

Reads ../facex/lfw_pairs.txt, decrypts models/*.enc with the same XOR-split key
as the demo, center-crop 250->168 -> resize 112 -> CHW (x-127.5)/128,
flip-augmented embedding average, 10-fold CV.

Usage: python tools/eval_lfw.py [tiny|nano]
"""
import io
import sys
import pickle
from pathlib import Path

import numpy as np
from PIL import Image
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent.parent
LFW = ROOT / "data" / "lfw-deepfunneled"
PAIRS = ROOT.parent / "facex" / "lfw_pairs.txt"
MODEL = sys.argv[1] if len(sys.argv) > 1 else "tiny"
DIM = int(sys.argv[2]) if len(sys.argv) > 2 else 512

KP = bytes([0x42,0x8a,0x2f,0x98,0xd7,0x28,0xae,0x22,0x71,0x37,0x44,0x91,0x23,0xef,0x65,0xcd,0xb5,0xc0,0xfb,0xcf,0xec,0x4d,0x3b,0x2f,0xe9,0xb5,0xdb,0xa5,0x81,0x89,0xdb,0xbc])
KO = bytes([0xb7,0x4d,0xb2,0x24,0xec,0x76,0x04,0xe4,0xcb,0x46,0xc8,0x02,0x97,0x3c,0x33,0xfa,0x8b,0xb0,0x11,0x60,0x0c,0xd5,0x02,0xf3,0xac,0xb8,0x62,0x0c,0xf3,0xad,0x54,0x01])
KEY = bytes(a ^ b for a, b in zip(KP, KO))

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

blob = (ROOT / "models" / f"facex_{MODEL}.enc").read_bytes()
onnx_bytes = AESGCM(KEY).decrypt(blob[:12], blob[12:], None)
sess = ort.InferenceSession(onnx_bytes, providers=["CPUExecutionProvider"])
print(f"model={MODEL} OK · inputs={[i.name for i in sess.get_inputs()]} · outputs={[o.name for o in sess.get_outputs()]}")

def crop_resize(img: Image.Image, out: int = 112, box: int = 168) -> Image.Image:
    W, H = img.size
    left = (W - box) // 2
    top = (H - box) // 2
    return img.crop((left, top, left + box, top + box)).resize((out, out), Image.BILINEAR)

def chw(img: Image.Image) -> np.ndarray:
    arr = np.asarray(img, dtype=np.float32)
    arr = (arr - 127.5) / 128.0
    return np.transpose(arr, (2, 0, 1))[None]

def embed(path: Path) -> np.ndarray:
    img = crop_resize(Image.open(path).convert("RGB"))
    a = chw(img)
    e1 = sess.run(None, {"input": a})[0][0]
    e2 = sess.run(None, {"input": a[:, :, :, ::-1].copy()})[0][0]
    e = e1 + e2
    return e / np.linalg.norm(e)

lines = PAIRS.read_text().strip().split("\n")
n_folds, n_per_fold = map(int, lines[0].split())
pairs, folds = [], []
idx = 1
for fold in range(n_folds):
    for _ in range(n_per_fold):
        name, i, j = lines[idx].split(); idx += 1
        pairs.append(((name, int(i)), (name, int(j)), True, fold))
    for _ in range(n_per_fold):
        n1, i, n2, j = lines[idx].split(); idx += 1
        pairs.append(((n1, int(i)), (n2, int(j)), False, fold))

cache = {}
def emb(key):
    if key not in cache:
        name, n = key
        cache[key] = embed(LFW / name / f"{name}_{n:04d}.jpg")
    return cache[key]

sims, issame = [], []
for k, ((a, b, same, fold)) in enumerate(pairs):
    sims.append(float(emb(a) @ emb(b)))
    issame.append(bool(same))
    if (k + 1) % 1000 == 0:
        print(f"pair {k+1}/{len(pairs)}", flush=True)

sims = np.array(sims); issame = np.array(issame)
gen, imp = sims[issame], sims[~issame]
print(f"genuine mean={gen.mean():.4f} (min {gen.min():.3f}) · impostor mean={imp.mean():.4f} (max {imp.max():.3f})")

accs, thrs = [], []
for f in range(n_folds):
    test_idx = np.array([i for i, p in enumerate(pairs) if p[3] == f])
    train_idx = np.array([i for i, p in enumerate(pairs) if p[3] != f])
    best_t, best_acc = 0.0, 0.0
    for t in np.linspace(0, 1, 401):
        acc = ((sims[train_idx] >= t) == issame[train_idx]).mean()
        if acc > best_acc:
            best_acc, best_t = acc, t
    accs.append(((sims[test_idx] >= best_t) == issame[test_idx]).mean())
    thrs.append(best_t)
print(f"LFW 10-fold CV accuracy: {np.mean(accs)*100:.2f}% ± {np.std(accs):.4f}   mean best threshold: {np.mean(thrs):.3f}")
