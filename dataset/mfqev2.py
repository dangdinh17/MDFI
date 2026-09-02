"""Datasets for MFQEv2-style compressed-video enhancement.

``MFQEv2Dataset`` keeps the seven-frame STDF/TGAF loader used by the
existing experiments. ``FeaturePQFMFQEv2Dataset`` implements the original
MFQE idea: the current compressed frame is enhanced using its nearest
previous and next peak-quality frames (PQFs). PQF references are read from a
metadata file generated with ``scripts/generate_feature_pqf_labels.py``.
"""

import json
import os.path as op
import random
from pathlib import Path

import cv2
import lmdb
import numpy as np
import torch
from torch.utils import data as data

from utils import FileClient, augment, paired_random_crop, totensor


def _bytes2img(img_bytes):
    img_np = np.frombuffer(img_bytes, np.uint8)
    decoded = cv2.imdecode(img_np, cv2.IMREAD_GRAYSCALE)
    if decoded is None:
        raise ValueError("Unable to decode an LMDB image")
    return np.expand_dims(decoded, 2).astype(np.float32) / 255.0


def _resolve_training_root(opts_dict):
    """Resolve a configured root while retaining legacy config support."""
    configured = opts_dict.get("root")
    gt_path = opts_dict["gt_path"]
    if configured and op.exists(op.join(configured, gt_path)):
        return configured
    return opts_dict.get("fallback_root", "datasets/108data/train_108")


class MFQEv2Dataset(data.Dataset):
    """Legacy consecutive-frame MFQEv2 dataset used by STDF/TGAF.

    The previous implementation used every LMDB metadata line as a sample but
    ignored the frame part of the key. With the local 15-frame LMDB layout,
    that silently repeated every sequence 15 times. This loader now retains
    only one center key per sequence.
    """

    def __init__(self, opts_dict, radius):
        super().__init__()
        self.opts_dict = opts_dict
        self.radius = radius
        self.root = _resolve_training_root(opts_dict)
        self.gt_root = op.join(self.root, opts_dict["gt_path"])
        self.lq_root = op.join(self.root, opts_dict["lq_path"])

        meta_info_path = op.join(self.gt_root, opts_dict["meta_info_fp"])
        with open(meta_info_path, "r") as fin:
            all_keys = [line.split(" ")[0] for line in fin]

        center_name = f"im{radius + 1}.png"
        self.keys = [key for key in all_keys if key.endswith("/" + center_name)]
        if not self.keys:
            raise RuntimeError(
                f"No center-frame keys ending in {center_name} found in "
                f"{meta_info_path}"
            )

        self.file_client = None
        self.io_opts_dict = {
            "type": "lmdb",
            "db_paths": [self.lq_root, self.gt_root],
            "client_keys": ["lq", "gt"],
        }
        self.neighbor_list = list(range(1, 2 * radius + 2))

    def __getitem__(self, index):
        if self.file_client is None:
            io_type = self.io_opts_dict.pop("type")
            self.file_client = FileClient(io_type, **self.io_opts_dict)

        neighbors = self.neighbor_list.copy()
        if self.opts_dict["random_reverse"] and random.random() < 0.5:
            neighbors.reverse()

        key = self.keys[index]
        clip, seq, _ = key.split("/")
        center = self.radius + 1
        img_gt_path = f"{clip}/{seq}/im{center}.png"
        img_gt = _bytes2img(self.file_client.get(img_gt_path, "gt"))
        img_lqs = [
            _bytes2img(
                self.file_client.get(f"{clip}/{seq}/im{neighbor}.png", "lq")
            )
            for neighbor in neighbors
        ]

        img_gt, img_lqs = paired_random_crop(
            img_gt, img_lqs, self.opts_dict["gt_size"], img_gt_path
        )
        img_results = augment(
            img_lqs + [img_gt],
            self.opts_dict["use_flip"],
            self.opts_dict["use_rot"],
        )
        img_results = totensor(img_results)
        return {
            "lq": torch.stack(img_results[:-1], dim=0),
            "gt": img_results[-1],
            "key": key,
        }

    def __len__(self):
        return len(self.keys)


class FeaturePQFMFQEv2Dataset(data.Dataset):
    """Three-frame MFQEv2 training set with feature-derived PQF references.

    Each sample contains ``[previous PQF, current LQ, next PQF]`` and the
    current GT frame. All four images receive the same crop and augmentation.
    """

    def __init__(self, opts_dict, radius=None):
        super().__init__()
        del radius  # accepted for compatibility with the repository factory
        self.opts_dict = opts_dict
        self.root = Path(opts_dict["root"])
        self.gt_root = self.root / opts_dict.get("gt_path", "gt.lmdb")
        self.lq_root = self.root / opts_dict["lq_path"]
        self.label_path = Path(opts_dict["label_path"])
        self.crop_size = int(opts_dict.get("gt_size", 128))
        self.use_flip = bool(opts_dict.get("use_flip", True))
        self.use_rot = bool(opts_dict.get("use_rot", True))
        self.random_reverse = bool(opts_dict.get("random_reverse", False))
        self.crop_mode = opts_dict.get("crop_mode", "random")
        self.cache_mode = opts_dict.get("cache_mode", "none")

        with self.label_path.open("r") as fin:
            metadata = json.load(fin)
        if int(metadata["qp"]) != int(opts_dict["qp"]):
            raise ValueError(
                f"Label QP {metadata['qp']} does not match dataset QP "
                f"{opts_dict['qp']}"
            )

        clip_ids = opts_dict.get("clip_ids")
        clip_set = {str(int(v)).zfill(3) for v in clip_ids} if clip_ids else None
        self.samples = [
            item
            for item in metadata["entries"]
            if clip_set is None or item["key"].split("/", 1)[0] in clip_set
        ]
        max_samples = opts_dict.get("max_samples")
        if max_samples:
            self.samples = self.samples[: int(max_samples)]
        if not self.samples:
            raise RuntimeError(
                f"No samples selected from feature-PQF labels {self.label_path}"
            )

        self.file_client = None
        self.gt_cache = None
        self.lq_cache = None
        if self.cache_mode == "ram":
            gt_keys = sorted({sample["key"] for sample in self.samples})
            lq_keys = sorted(
                {
                    key
                    for sample in self.samples
                    for key in (
                        sample["prev_pqf"],
                        sample["key"],
                        sample["next_pqf"],
                    )
                }
            )
            self.gt_cache = self._preload_lmdb(self.gt_root, gt_keys, "GT")
            self.lq_cache = self._preload_lmdb(self.lq_root, lq_keys, "LQ")
        elif self.cache_mode == "none":
            self.io_opts_dict = {
                "type": "lmdb",
                "db_paths": [str(self.lq_root), str(self.gt_root)],
                "client_keys": ["lq", "gt"],
            }
        else:
            raise ValueError(f"Unsupported cache mode: {self.cache_mode}")

    @staticmethod
    def _preload_lmdb(path, keys, name):
        cache = {}
        env = lmdb.open(
            str(path), readonly=True, lock=False, readahead=True, meminit=False
        )
        with env.begin(write=False) as txn:
            for index, key in enumerate(keys, 1):
                payload = txn.get(key.encode("ascii"))
                if payload is None:
                    raise KeyError(f"{path}: missing {key}")
                image = cv2.imdecode(
                    np.frombuffer(payload, np.uint8), cv2.IMREAD_GRAYSCALE
                )
                if image is None:
                    raise ValueError(f"{path}: cannot decode {key}")
                cache[key] = np.expand_dims(image, 2)
                if index % 2000 == 0 or index == len(keys):
                    gib = sum(value.nbytes for value in cache.values()) / 2**30
                    print(
                        f"RAM cache {name}: {index}/{len(keys)} frames, "
                        f"{gib:.2f} GiB",
                        flush=True,
                    )
        env.close()
        return cache

    def _read(self, key, client_key):
        if self.cache_mode == "ram":
            cache = self.lq_cache if client_key == "lq" else self.gt_cache
            return cache[key]
        return _bytes2img(self.file_client.get(key, client_key))

    def __getitem__(self, index):
        if self.cache_mode == "none" and self.file_client is None:
            io_type = self.io_opts_dict.pop("type")
            self.file_client = FileClient(io_type, **self.io_opts_dict)

        sample = self.samples[index]
        ref_keys = [sample["prev_pqf"], sample["key"], sample["next_pqf"]]
        img_lqs = [self._read(key, "lq") for key in ref_keys]
        img_gt = self._read(sample["key"], "gt")

        if self.crop_mode == "center":
            height, width = img_gt.shape[:2]
            if height < self.crop_size or width < self.crop_size:
                raise ValueError(
                    f"Frame {sample['key']} ({height}x{width}) is smaller than "
                    f"crop size {self.crop_size}"
                )
            top = (height - self.crop_size) // 2
            left = (width - self.crop_size) // 2
            img_gt = img_gt[top : top + self.crop_size, left : left + self.crop_size]
            img_lqs = [
                image[top : top + self.crop_size, left : left + self.crop_size]
                for image in img_lqs
            ]
        elif self.crop_mode == "random":
            img_gt, img_lqs = paired_random_crop(
                img_gt, img_lqs, self.crop_size, sample["key"]
            )
        else:
            raise ValueError(f"Unsupported crop mode: {self.crop_mode}")
        img_lqs = [
            np.ascontiguousarray(image, dtype=np.float32) / 255.0
            if image.dtype == np.uint8
            else np.ascontiguousarray(image, dtype=np.float32)
            for image in img_lqs
        ]
        img_gt = (
            np.ascontiguousarray(img_gt, dtype=np.float32) / 255.0
            if img_gt.dtype == np.uint8
            else np.ascontiguousarray(img_gt, dtype=np.float32)
        )
        results = augment(img_lqs + [img_gt], self.use_flip, self.use_rot)
        results = totensor(results)
        img_lqs = torch.stack(results[:3], dim=0)
        img_gt = results[3]

        if self.random_reverse and random.random() < 0.5:
            img_lqs = img_lqs.flip(0)
            ref_keys = list(reversed(ref_keys))

        return {
            "lq": img_lqs,
            "gt": img_gt,
            "is_pqf": torch.tensor(sample["is_pqf"], dtype=torch.bool),
            "feature_score": torch.tensor(sample["score"], dtype=torch.float32),
            "key": sample["key"],
            "ref_keys": ref_keys,
        }

    def __len__(self):
        return len(self.samples)
