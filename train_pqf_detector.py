#!/usr/bin/env python3
"""Train the BiLSTM PQF detector from LQ-only perceptual descriptors."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from models.pqf_detector import FeaturePQFDetector


class SequenceChunks(Dataset):
    def __init__(self, features, labels, keys, clips, chunk_length):
        self.items = []
        clip_set = {str(int(value)).zfill(3) for value in clips}
        for clip in sorted(clip_set, key=int):
            indices = [i for i, key in enumerate(keys) if key.split("/", 1)[0] == clip]
            for start in range(0, len(indices), chunk_length):
                selected = indices[start : start + chunk_length]
                if selected:
                    self.items.append((features[selected], labels[selected]))

    def __getitem__(self, index):
        features, labels = self.items[index]
        return torch.from_numpy(features), torch.from_numpy(labels)

    def __len__(self):
        return len(self.items)


def collate(batch):
    lengths = torch.tensor([len(item[1]) for item in batch], dtype=torch.long)
    max_length = int(lengths.max())
    feature_size = batch[0][0].shape[1]
    features = torch.zeros(len(batch), max_length, feature_size)
    labels = torch.zeros(len(batch), max_length)
    mask = torch.zeros(len(batch), max_length, dtype=torch.bool)
    for index, (item_features, item_labels) in enumerate(batch):
        length = len(item_labels)
        features[index, :length] = item_features
        labels[index, :length] = item_labels
        mask[index, :length] = True
    return features, labels, mask


def metrics(predictions, labels):
    targets = labels.bool()
    tp = int((predictions & targets).sum())
    fp = int((predictions & ~targets).sum())
    fn = int((~predictions & targets).sum())
    tn = int((~predictions & ~targets).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    accuracy = (tp + tn) / max(tp + fp + fn + tn, 1)
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


@torch.no_grad()
def validate(model, loader, device, criterion):
    model.eval()
    losses, sequence_probabilities, sequence_labels = [], [], []
    for features, labels, mask in loader:
        features, labels, mask = features.to(device), labels.to(device), mask.to(device)
        logits = model(features)
        losses.append(float(criterion(logits[mask], labels[mask])))
        for row, length in enumerate(mask.sum(dim=1).tolist()):
            sequence_probabilities.append(logits[row, :length].sigmoid().cpu())
            sequence_labels.append(labels[row, :length].cpu())
    best = None
    for threshold in np.linspace(0.05, 0.95, 181):
        predictions = [
            model.local_peak_labels(probabilities, float(threshold))
            for probabilities in sequence_probabilities
        ]
        candidate = metrics(torch.cat(predictions), torch.cat(sequence_labels))
        candidate["threshold"] = float(threshold)
        if best is None or candidate["f1"] > best["f1"]:
            best = candidate
    result = best
    result["loss"] = float(np.mean(losses))
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qp", type=int, required=True)
    parser.add_argument("--feature-dir", default="datasets/108data/feature_pqf")
    parser.add_argument("--output-dir", default="exp")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--chunk-length", type=int, default=120)
    parser.add_argument("--hidden-size", type=int, default=96)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    source = Path(args.feature_dir) / f"QP{args.qp}_detector_features.npz"
    arrays = np.load(source)
    keys = arrays["keys"].tolist()
    raw_features = arrays["features"].astype(np.float32)
    labels = arrays["labels"].astype(np.float32)

    train_indices = np.asarray(
        [i for i, key in enumerate(keys) if int(key.split("/", 1)[0]) <= 100]
    )
    mean = raw_features[train_indices].mean(axis=0, keepdims=True)
    std = raw_features[train_indices].std(axis=0, keepdims=True).clip(min=1e-6)
    features = ((raw_features - mean) / std).astype(np.float32)
    train_set = SequenceChunks(features, labels, keys, range(1, 101), args.chunk_length)
    val_set = SequenceChunks(features, labels, keys, range(101, 109), args.chunk_length)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate
    )

    device = torch.device(args.device)
    model = FeaturePQFDetector(
        input_size=features.shape[1], hidden_size=args.hidden_size
    ).to(device)
    positives = labels[train_indices].sum()
    negatives = len(train_indices) - positives
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(negatives / max(positives, 1), device=device)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    output_dir = Path(args.output_dir) / f"MFQEv2_feature_QP{args.qp}" / "detector"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "metrics.jsonl"
    log_path.write_text("")
    best_f1 = -1.0
    stale_epochs = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for batch_features, batch_labels, mask in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            mask = mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_features)
            loss = criterion(logits[mask], batch_labels[mask])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_losses.append(float(loss.detach()))
        result = validate(model, val_loader, device, criterion)
        result.update(epoch=epoch, train_loss=float(np.mean(train_losses)))
        with log_path.open("a") as log_file:
            log_file.write(json.dumps(result) + "\n")
        print(f"QP{args.qp} detector epoch {epoch}: {result}", flush=True)
        if result["f1"] > best_f1:
            best_f1 = result["f1"]
            stale_epochs = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "qp": args.qp,
                    "input_size": features.shape[1],
                    "hidden_size": args.hidden_size,
                    "feature_mean": mean.squeeze(0),
                    "feature_std": std.squeeze(0),
                    "metrics": result,
                },
                output_dir / "best.pth",
            )
        else:
            stale_epochs += 1
            if stale_epochs >= 5:
                break


if __name__ == "__main__":
    main()
