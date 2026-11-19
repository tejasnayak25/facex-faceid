"""Ground-truth LFW verification for YuNet + SFace (OpenCV pipeline).

Same protocol as eval_lfw.py: LFW deepfunneled, official lfw_pairs.txt,
10-fold CV with best threshold per fold. cv2 does detect → 5-point align →
128-d embedding; cosine similarity between pairs.

Usage: python tools/eval_sface.py
"""
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parent.parent
LFW = ROOT / "data" / "lfw-deepfunneled"
PAIRS = ROOT.parent / "facex" / "lfw_pairs.txt"
DET = ROOT / "models" / "yunet_2023mar_int8.onnx"
REC = ROOT / "models" / "sface_int8.onnx"

detector = cv2.FaceDetectorYN.create(str(DET), "", (320, 320), 0.6, 0.3, 50)
recognizer = cv2.FaceRecognizerSF.create(str(REC), "")

def embed(img_path: Path):
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    h, w = img.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(img)
    if faces is None or len(faces) == 0:
        return None
    f = max(faces, key=lambda f: f[2] * f[3])   # largest face
    aligned = recognizer.alignCrop(img, f)
    emb = recognizer.feature(aligned).flatten().astype(np.float64)
    return emb / (np.linalg.norm(emb) + 1e-9)

lines = PAIRS.read_text().strip().split("\n")
n_folds, n_per_fold = map(int, lines[0].split())
pairs = []
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

sims, issame, folds_out = [], [], []
skipped = 0
for k, (a, b, same, fold) in enumerate(pairs):
    ea, eb = emb(a), emb(b)
    if ea is None or eb is None:
        skipped += 1
        continue
    sims.append(float(ea @ eb)); issame.append(bool(same)); folds_out.append(fold)
    if (k + 1) % 1000 == 0:
        print(f"pair {k+1}/{len(pairs)}", flush=True)

sims = np.array(sims); issame = np.array(issame); folds_out = np.array(folds_out)
print(f"evaluated {len(sims)}/{len(pairs)} pairs ({skipped} skipped: no face)")
gen, imp = sims[issame], sims[~issame]
print(f"genuine mean={gen.mean():.4f} (min {gen.min():.3f}) · impostor mean={imp.mean():.4f} (max {imp.max():.3f})")

allp = np.concatenate([gen, imp])
labels = np.concatenate([np.ones_like(gen), np.zeros_like(imp)])
order = np.argsort(allp); allp = allp[order]; labels = labels[order]
rank = np.arange(1, len(allp) + 1)
auc = (rank[labels == 1].sum() - len(gen) * (len(gen) + 1) / 2) / (len(gen) * len(imp))
print(f"AUC={auc:.4f}")

accs, thrs = [], []
print("fold   acc    thr")
for f in range(n_folds):
    test_idx = np.where(folds_out == f)[0]; train_idx = np.where(folds_out != f)[0]
    best_t, best_acc = 0.0, 0.0
    for t in np.linspace(0, 1, 401):
        acc = ((sims[train_idx] >= t) == issame[train_idx]).mean()
        if acc > best_acc:
            best_acc, best_t = acc, t
    t_acc = ((sims[test_idx] >= best_t) == issame[test_idx]).mean()
    accs.append(t_acc); thrs.append(best_t)
    print(f"{f}      {t_acc:.4f}  {best_t:.3f}")
print(f"LFW 10-fold CV accuracy: {np.mean(accs)*100:.2f}% ± {np.std(accs):.4f}   mean best threshold: {np.mean(thrs):.3f}")
