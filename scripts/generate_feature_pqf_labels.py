#!/usr/bin/env python3
"""Generate MFQEv2 PQF labels from LPIPS feature distance.

A frame is a PQF when its LQ-to-GT feature distance is strictly lower than
the distances of both immediate temporal neighbors.  The script also stores
LQ-only AlexNet descriptors used to train the BiLSTM PQF detector.
"""

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import lmdb
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.feature_loss import LPIPSAlexFeatureLoss


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="datasets/108data/train_108")
    parser.add_argument("--output-dir", default="datasets/108data/feature_pqf")
    parser.add_argument("--qp", nargs="+", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--resize", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--decode-reduction",
        type=int,
        choices=(1, 2, 4, 8),
        default=2,
        help="Decode PNGs at reduced resolution before the 128x128 feature resize",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--max-frames", type=int, help="Limit frames for a preprocessing smoke test"
    )
    return parser.parse_args()


def key_order(key):
    clip, sequence, frame_name = key.split("/")
    frame = int(frame_name[2:-4])
    return int(clip), int(sequence), frame


def decode(txn, key, size, reduction):
    value = txn.get(key.encode("utf-8"))
    if value is None:
        raise KeyError(f"Missing LMDB key: {key}")
    decode_flags = {
        1: cv2.IMREAD_GRAYSCALE,
        2: cv2.IMREAD_REDUCED_GRAYSCALE_2,
        4: cv2.IMREAD_REDUCED_GRAYSCALE_4,
        8: cv2.IMREAD_REDUCED_GRAYSCALE_8,
    }
    image = cv2.imdecode(np.frombuffer(value, np.uint8), decode_flags[reduction])
    if image is None:
        raise ValueError(f"Unable to decode LMDB key: {key}")
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().div_(255.0)


def resized_batch(txn, keys, size, reduction):
    return torch.cat([decode(txn, key, size, reduction) for key in keys], dim=0)


def references_for_clip(keys, scores):
    labels = np.zeros(len(keys), dtype=np.bool_)
    if len(keys) >= 3:
        labels[1:-1] = (scores[1:-1] < scores[:-2]) & (
            scores[1:-1] < scores[2:]
        )
    pqf_indices = np.flatnonzero(labels).tolist()
    entries = []
    for index, key in enumerate(keys):
        previous = [candidate for candidate in pqf_indices if candidate < index]
        subsequent = [candidate for candidate in pqf_indices if candidate > index]
        previous_key = keys[previous[-1]] if previous else key
        next_key = keys[subsequent[0]] if subsequent else key
        entries.append(
            {
                "key": key,
                "score": float(scores[index]),
                "is_pqf": bool(labels[index]),
                "prev_pqf": previous_key,
                "next_pqf": next_key,
            }
        )
    return entries


def generate_for_qp(args, qp, metric):
    root = Path(args.root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    label_path = output_dir / f"QP{qp}_lpips_alex.json"
    feature_path = output_dir / f"QP{qp}_detector_features.npz"
    if label_path.exists() and feature_path.exists() and not args.force:
        print(f"QP{qp}: labels already exist; use --force to regenerate")
        return

    gt_path = root / "gt.lmdb"
    lq_path = root / f"QP{qp}" / "lq.lmdb"
    meta_path = gt_path / "meta_info.txt"
    keys = sorted(
        [line.split(" ", 1)[0] for line in meta_path.read_text().splitlines()],
        key=key_order,
    )
    if args.max_frames:
        keys = keys[: args.max_frames]
    gt_env = lmdb.open(
        str(gt_path), readonly=True, lock=False, readahead=True, meminit=False
    )
    lq_env = lmdb.open(
        str(lq_path), readonly=True, lock=False, readahead=True, meminit=False
    )

    all_scores = []
    all_features = []
    with gt_env.begin(write=False) as gt_txn, lq_env.begin(write=False) as lq_txn:
        for start in range(0, len(keys), args.batch_size):
            batch_keys = keys[start : start + args.batch_size]
            lq = resized_batch(
                lq_txn, batch_keys, args.resize, args.decode_reduction
            ).to(args.device)
            gt = resized_batch(
                gt_txn, batch_keys, args.resize, args.decode_reduction
            ).to(args.device)
            distances, embeddings = metric.distance_and_embedding(lq, gt)
            all_scores.append(distances.cpu().numpy())
            all_features.append(embeddings.cpu().numpy())
            if start == 0 or (start // args.batch_size + 1) % 10 == 0:
                done = min(start + args.batch_size, len(keys))
                print(f"QP{qp}: feature pass {done}/{len(keys)}", flush=True)
    gt_env.close()
    lq_env.close()

    scores = np.concatenate(all_scores).astype(np.float32)
    features = np.concatenate(all_features).astype(np.float16)
    clip_to_indices = defaultdict(list)
    for index, key in enumerate(keys):
        clip_to_indices[key.split("/", 1)[0]].append(index)

    entries = []
    labels = np.zeros(len(keys), dtype=np.uint8)
    for clip in sorted(clip_to_indices, key=int):
        indices = clip_to_indices[clip]
        clip_entries = references_for_clip(
            [keys[index] for index in indices], scores[indices]
        )
        for index, entry in zip(indices, clip_entries):
            labels[index] = int(entry["is_pqf"])
        entries.extend(clip_entries)

    metadata = {
        "version": 1,
        "qp": qp,
        "rule": "score[t] < score[t-1] and score[t] < score[t+1]",
        "metric": "LPIPS-AlexNet-v0.1",
        "resize": args.resize,
        "decode_reduction": args.decode_reduction,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "num_frames": len(keys),
        "num_pqf": int(labels.sum()),
        "entries": entries,
    }
    label_tmp = label_path.with_suffix(".tmp")
    label_tmp.write_text(json.dumps(metadata, separators=(",", ":")))
    label_tmp.replace(label_path)

    feature_tmp = feature_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        feature_tmp,
        keys=np.asarray(keys),
        features=features,
        labels=labels,
        scores=scores,
    )
    feature_tmp.replace(feature_path)
    print(
        f"QP{qp}: wrote {len(keys)} frames, {int(labels.sum())} PQFs "
        f"({labels.mean():.2%})"
    )


def main():
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    metric = LPIPSAlexFeatureLoss(resize=None).to(args.device).eval()
    for qp in args.qp:
        generate_for_qp(args, qp, metric)


if __name__ == "__main__":
    main()
