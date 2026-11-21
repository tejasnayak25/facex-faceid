#!/usr/bin/env python3
"""Rebuild embeddings from dataset using OpenCV YuNet + SFace and export in Event format.

Output schema:
[
  {
    "id": "user_1",
    "name": "Tejas Nayak",
    "ticketId": "TKT-VIP-8841",
    "ticketType": "VIP All-Access",
    "avatar": "https://images.unsplash.com/photo-1535713875002-d1d0cf377fde?w=150",
    "vector": [0.0451, -0.0123, 0.0892, ...]
  }
]
"""
import argparse
import concurrent.futures
import json
import os
import random
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

# Deterministic ticket and avatar assignment
random.seed(42)

TICKET_TYPES = [
    ("VIP All-Access", "VIP"),
    ("General Admission", "GA"),
    ("Speaker Pass", "SPK"),
    ("Early Bird Pass", "EB"),
    ("All-Access Pass", "ALL"),
    ("Press & Media", "MED"),
    ("Student Pass", "STU"),
    ("Executive VIP", "EXEC"),
]

AVATAR_POOL = [
    "https://images.unsplash.com/photo-1535713875002-d1d0cf377fde?w=150",
    "https://images.unsplash.com/photo-1494790108377-be9c29b29330?w=150",
    "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=150",
    "https://images.unsplash.com/photo-1517841905240-472988babdf9?w=150",
    "https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=150",
    "https://images.unsplash.com/photo-1500648767791-00dcc994a43e?w=150",
    "https://images.unsplash.com/photo-1522075469751-3a6694fb2f61?w=150",
    "https://images.unsplash.com/photo-1544005313-94ddf0286df2?w=150",
    "https://images.unsplash.com/photo-1506794778202-cad84cf45f1d?w=150",
    "https://images.unsplash.com/photo-1519085360753-af0119f7cbe7?w=150",
    "https://images.unsplash.com/photo-1524504388940-b1c1722653e1?w=150",
    "https://images.unsplash.com/photo-1528892952291-009c663ce843?w=150",
    "https://images.unsplash.com/photo-1539571696357-5a69c17a67c6?w=150",
    "https://images.unsplash.com/photo-1517462964-21fdcec3f25b?w=150",
    "https://images.unsplash.com/photo-1492562080023-ab3db95bfbce?w=150",
]

_tls = threading.local()


def get_models(yunet_path: str, sface_path: str, score_thresh: float):
    if not hasattr(_tls, "detector"):
        _tls.detector = cv2.FaceDetectorYN.create(yunet_path, "", (250, 250), score_thresh, 0.3, 50)
        _tls.recognizer = cv2.FaceRecognizerSF.create(sface_path, "")
    return _tls.detector, _tls.recognizer


def clean_name(folder_name: str) -> str:
    parts = folder_name.split("_")
    return " ".join(parts).strip()


def process_identity(item):
    idx, dir_path, yunet_path, sface_path, score_thresh, max_per_student, valid_exts = item
    detector, recognizer = get_models(yunet_path, sface_path, score_thresh)

    img_files = sorted([f for f in dir_path.iterdir() if f.suffix.lower() in valid_exts])
    if not img_files:
        return None

    if max_per_student > 0:
        selected_files = img_files[:max_per_student]
    else:
        selected_files = img_files

    embs = []
    num_processed = 0

    for img_file in selected_files:
        img = cv2.imread(str(img_file))
        if img is None:
            continue

        h, w = img.shape[:2]
        detector.setInputSize((w, h))
        ret, faces = detector.detect(img)
        num_processed += 1

        if faces is None or len(faces) == 0:
            continue

        # Pick largest face
        f = max(faces, key=lambda x: x[2] * x[3])
        aligned = recognizer.alignCrop(img, f)
        feat = recognizer.feature(aligned).flatten().astype(np.float32)
        norm = np.linalg.norm(feat)
        if norm > 1e-6:
            feat /= norm
            embs.append(feat)

    if not embs:
        return (idx, dir_path.name, None, num_processed)

    mean_emb = np.mean(embs, axis=0)
    norm = np.linalg.norm(mean_emb)
    if norm > 1e-6:
        mean_emb /= norm

    emb_list = [round(float(v), 6) for v in mean_emb]
    return (idx, dir_path.name, emb_list, num_processed)


def main():
    parser = argparse.ArgumentParser(description="Rebuild embeddings in Event format")
    parser.add_argument("--dataset", type=str, default="checkin/data/lfw-deepfunneled")
    parser.add_argument("--out", type=str, default="embeddings.json")
    parser.add_argument("--yunet", type=str, default="checkin/models/yunet.onnx")
    parser.add_argument("--sface", type=str, default="checkin/models/sface.onnx")
    parser.add_argument("--max-per-student", type=int, default=5)
    parser.add_argument("--score-thresh", type=float, default=0.6)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4))
    args = parser.parse_args()

    dataset_path = Path(args.dataset).resolve()
    yunet_path = Path(args.yunet).resolve()
    sface_path = Path(args.sface).resolve()

    if not dataset_path.exists():
        print(f"Error: dataset path not found: {dataset_path}", file=sys.stderr)
        sys.exit(1)
    if not yunet_path.exists():
        print(f"Error: YuNet model not found: {yunet_path}", file=sys.stderr)
        sys.exit(1)
    if not sface_path.exists():
        print(f"Error: SFace model not found: {sface_path}", file=sys.stderr)
        sys.exit(1)

    print(f"=== FaceX Embeddings Rebuilder (Event Format) ===")
    print(f"  Dataset : {dataset_path}")
    print(f"  Detector: {yunet_path.name} (YuNet)")
    print(f"  Features: {sface_path.name} (SFace 128-d)")
    print(f"  Threads : {args.workers} workers")

    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
    subdirs = sorted([d for d in dataset_path.iterdir() if d.is_dir()])
    print(f"  Total identities: {len(subdirs)}")

    # Pre-generate deterministic ticket types and avatars
    ticket_data = []
    for idx in range(len(subdirs)):
        ttype, tcode = random.choice(TICKET_TYPES)
        ticket_num = random.randint(1000, 9999)
        ticket_id = f"TKT-{tcode}-{ticket_num}"
        avatar = random.choice(AVATAR_POOL)
        ticket_data.append((ticket_id, ttype, avatar))

    items = [
        (i, d, str(yunet_path), str(sface_path), args.score_thresh, args.max_per_student, valid_exts)
        for i, d in enumerate(subdirs)
    ]

    t0 = time.time()
    results = [None] * len(subdirs)
    total_images = 0
    completed_count = 0

    print(f"\nProcessing {len(subdirs)} identities using {args.workers} concurrent workers...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_idx = {executor.submit(process_identity, item): item[0] for item in items}
        for future in concurrent.futures.as_completed(future_to_idx):
            res = future.result()
            if res is not None:
                idx, folder_name, emb, n_img = res
                results[idx] = (folder_name, emb)
                total_images += n_img
            completed_count += 1

            if completed_count % 1000 == 0 or completed_count == len(subdirs):
                dt = time.time() - t0
                ips = total_images / max(0.001, dt)
                print(f"  [{completed_count}/{len(subdirs)}] {completed_count*100//len(subdirs)}% complete ({total_images} images, {ips:.1f} img/s, {dt:.1f}s elapsed)")

    dt_total = time.time() - t0
    print(f"\nProcessing completed in {dt_total:.2f}s ({total_images / max(0.001, dt_total):.1f} img/s)!")

    # Format attendees in event format
    attendees = []
    failed = []
    for idx, res in enumerate(results):
        if res is None or res[1] is None:
            failed.append(subdirs[idx].name)
            continue
        folder_name, emb = res
        name = clean_name(folder_name)
        ticket_id, ttype, avatar = ticket_data[idx]

        attendees.append({
            "id": f"user_{len(attendees) + 1}",
            "name": name,
            "ticketId": ticket_id,
            "ticketType": ttype,
            "avatar": avatar,
            "vector": emb,
        })

    print(f"  Enrolled: {len(attendees)} / {len(subdirs)} attendees")
    if failed:
        print(f"  Failed: {len(failed)}")

    out_file = Path(args.out).resolve()
    print(f"\nSaving event format to {out_file}...")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(attendees, f, indent=2)

    checkin_file = Path("checkin/embeddings.json").resolve()
    if out_file != checkin_file:
        print(f"Also copying to {checkin_file}...")
        with open(checkin_file, "w", encoding="utf-8") as f:
            json.dump(attendees, f, indent=2)

    size_mb = out_file.stat().st_size / (1024 * 1024)
    print(f"Done! Final output size: {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
