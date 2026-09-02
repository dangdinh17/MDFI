#!/usr/bin/env python3
"""Integrity checks for the local 108-video MFQEv2 LMDB dataset."""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import lmdb
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="datasets/108data/train_108")
    parser.add_argument("--label-dir", default="datasets/108data/feature_pqf")
    parser.add_argument("--qp", nargs="+", type=int, default=[22, 27, 32, 37])
    parser.add_argument("--require-labels", action="store_true")
    return parser.parse_args()


def meta(path):
    payload = path.read_bytes()
    keys = [line.split(" ", 1)[0] for line in payload.decode().splitlines()]
    return keys, hashlib.sha256(payload).hexdigest()


def decode_samples(lmdb_path, keys):
    env = lmdb.open(
        str(lmdb_path), readonly=True, lock=False, readahead=False, meminit=False
    )
    shapes = []
    with env.begin(write=False) as txn:
        for key in (keys[0], keys[len(keys) // 2], keys[-1]):
            payload = txn.get(key.encode())
            if payload is None:
                raise KeyError(f"{lmdb_path}: missing {key}")
            image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"{lmdb_path}: cannot decode {key}")
            shapes.append([key, list(image.shape)])
    env.close()
    return shapes


def main():
    args = parse_args()
    root = Path(args.root)
    gt_keys, gt_hash = meta(root / "gt.lmdb" / "meta_info.txt")
    if len(gt_keys) != len(set(gt_keys)):
        raise ValueError("GT metadata contains duplicate keys")
    report = {
        "root": str(root.resolve()),
        "num_frames": len(gt_keys),
        "num_clips": len({key.split("/", 1)[0] for key in gt_keys}),
        "gt_meta_sha256": gt_hash,
        "gt_samples": decode_samples(root / "gt.lmdb", gt_keys),
        "qps": {},
    }
    key_set = set(gt_keys)
    for qp in args.qp:
        lq_path = root / f"QP{qp}" / "lq.lmdb"
        lq_keys, lq_hash = meta(lq_path / "meta_info.txt")
        if lq_keys != gt_keys:
            raise ValueError(f"QP{qp}: LQ metadata keys differ from GT")
        qp_report = {
            "meta_sha256": lq_hash,
            "samples": decode_samples(lq_path, lq_keys),
        }
        label_path = Path(args.label_dir) / f"QP{qp}_lpips_alex.json"
        if label_path.exists():
            labels = json.loads(label_path.read_text())
            entries = labels["entries"]
            if len(entries) != len(gt_keys):
                raise ValueError(f"QP{qp}: label count differs from GT")
            for entry in entries:
                if (
                    entry["key"] not in key_set
                    or entry["prev_pqf"] not in key_set
                    or entry["next_pqf"] not in key_set
                ):
                    raise KeyError(f"QP{qp}: invalid label reference in {entry}")
            qp_report.update(
                num_pqf=labels["num_pqf"], pqf_ratio=labels["num_pqf"] / len(entries)
            )
        elif args.require_labels:
            raise FileNotFoundError(label_path)
        report["qps"][str(qp)] = qp_report
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
