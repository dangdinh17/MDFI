#!/usr/bin/env python3
"""Train feature-PQF MFQE 2.0 with Charbonnier + LPIPS loss."""

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from dataset.mfqev2 import FeaturePQFMFQEv2Dataset
from models.mfqev2_feature import FeaturePQFMFQEv2
from utils.deep_learning import CharbonnierLoss
from utils.feature_loss import LPIPSAlexFeatureLoss


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_MFQEv2_feature.yml")
    parser.add_argument("--qp", type=int, required=True)
    parser.add_argument("--resume")
    parser.add_argument("--init")
    parser.add_argument("--num-iter", type=int)
    parser.add_argument("--output-dir", help="Override the configured experiment directory")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def render(value, qp):
    if isinstance(value, str):
        return value.format(qp=qp)
    if isinstance(value, dict):
        return {key: render(item, qp) for key, item in value.items()}
    if isinstance(value, list):
        return [render(item, qp) for item in value]
    return value


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def clip_range(bounds):
    return list(range(int(bounds[0]), int(bounds[1]) + 1))


def make_dataset(config, qp, train):
    dataset_config = {
        "root": config["root"],
        "gt_path": config["gt_path"],
        "lq_path": config["lq_path"],
        "label_path": config["label_path"],
        "qp": qp,
        "gt_size": config["crop_size"],
        "clip_ids": clip_range(config["train_clips"] if train else config["val_clips"]),
        "crop_mode": "random" if train else "center",
        "use_flip": config["use_flip"] if train else False,
        "use_rot": config["use_rot"] if train else False,
        "random_reverse": config["random_reverse"] if train else False,
        "cache_mode": config.get("cache_mode", "none"),
    }
    if not train:
        dataset_config["max_samples"] = config.get("val_max_samples")
    return FeaturePQFMFQEv2Dataset(dataset_config)


def load_model_state(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    if state_dict and next(iter(state_dict)).startswith("module."):
        state_dict = {key[7:]: value for key, value in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    return checkpoint


@torch.no_grad()
def validate(model, loader, feature_loss, device):
    model.eval()
    base_psnr, enhanced_psnr, base_feature_values, feature_values = [], [], [], []
    for batch in loader:
        frames = batch["lq"].to(device, non_blocking=True)
        gt = batch["gt"].to(device, non_blocking=True)
        is_pqf = batch["is_pqf"].to(device, non_blocking=True)
        prediction = model(frames, is_pqf).clamp(0.0, 1.0)
        base_mse = (frames[:, 1] - gt).square().flatten(1).mean(1)
        enhanced_mse = (prediction - gt).square().flatten(1).mean(1)
        base_psnr.extend((-10.0 * torch.log10(base_mse.clamp_min(1e-12))).cpu().tolist())
        enhanced_psnr.extend(
            (-10.0 * torch.log10(enhanced_mse.clamp_min(1e-12))).cpu().tolist()
        )
        base_feature_values.append(float(feature_loss(frames[:, 1], gt)))
        feature_values.append(float(feature_loss(prediction, gt)))
    base = float(np.mean(base_psnr))
    enhanced = float(np.mean(enhanced_psnr))
    model.train()
    return {
        "base_psnr": base,
        "enhanced_psnr": enhanced,
        "delta_psnr": enhanced - base,
        "base_feature_loss": float(np.mean(base_feature_values)),
        "feature_loss": float(np.mean(feature_values)),
    }


def save_checkpoint(path, model, optimizer, scheduler, scaler, iteration, best, config, metrics=None):
    torch.save(
        {
            "iteration": iteration,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_delta_psnr": best,
            "config": config,
            "metrics": metrics,
        },
        path,
    )


def main():
    args = parse_args()
    if args.qp not in (22, 27, 32, 37):
        raise ValueError("This prepared dataset contains QP22, QP27, QP32, QP37")
    with open(args.config, "r") as config_file:
        config = render(yaml.safe_load(config_file), args.qp)
    if args.num_iter:
        config["train"]["num_iter"] = args.num_iter
    if args.output_dir:
        config["train"]["output_dir"] = args.output_dir
    device = torch.device(args.device)
    seed_everything(int(config["train"]["seed"]))
    torch.backends.cudnn.benchmark = True

    train_set = make_dataset(config["dataset"], args.qp, train=True)
    val_set = make_dataset(config["dataset"], args.qp, train=False)
    train_loader = DataLoader(
        train_set,
        batch_size=int(config["dataset"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["dataset"]["num_workers"]),
        pin_memory=True,
        drop_last=True,
        persistent_workers=int(config["dataset"]["num_workers"]) > 0,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=int(config["dataset"]["batch_size"]),
        shuffle=False,
        num_workers=min(4, int(config["dataset"]["num_workers"])),
        pin_memory=True,
    )
    model = FeaturePQFMFQEv2(**config["model"]).to(device)
    charbonnier = CharbonnierLoss(eps=float(config["loss"]["charbonnier_eps"]))
    feature_loss = LPIPSAlexFeatureLoss(resize=None).to(device).eval()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["train"]["learning_rate"]),
        weight_decay=float(config["train"]["weight_decay"]),
    )
    num_iter = int(config["train"]["num_iter"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_iter,
        eta_min=float(config["train"]["min_learning_rate"]),
    )
    amp_enabled = bool(config["train"]["amp"] and device.type == "cuda")
    scaler = GradScaler(enabled=amp_enabled)
    start_iter, best_delta = 0, -math.inf
    if args.resume:
        checkpoint = load_model_state(model, args.resume, device)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_iter = int(checkpoint["iteration"])
        best_delta = float(checkpoint.get("best_delta_psnr", -math.inf))
    elif args.init:
        load_model_state(model, args.init, device)
        print(f"Initialized QP{args.qp} from {args.init}")

    output_dir = Path(config["train"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.yml").write_text(yaml.safe_dump(config, sort_keys=False))
    log_path = output_dir / "metrics.jsonl"
    if not args.resume:
        log_path.write_text("")
    print(
        f"QP{args.qp}: {len(train_set)} train samples, {len(val_set)} val samples, "
        f"crop={config['dataset']['crop_size']}, iterations={num_iter}",
        flush=True,
    )

    model.train()
    loader_iterator = iter(train_loader)
    interval_start = time.time()
    running = {"total": 0.0, "charbonnier": 0.0, "feature": 0.0, "count": 0}
    for iteration in range(start_iter + 1, num_iter + 1):
        try:
            batch = next(loader_iterator)
        except StopIteration:
            loader_iterator = iter(train_loader)
            batch = next(loader_iterator)
        frames = batch["lq"].to(device, non_blocking=True)
        gt = batch["gt"].to(device, non_blocking=True)
        is_pqf = batch["is_pqf"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=amp_enabled):
            prediction = model(frames, is_pqf)
            char_loss = charbonnier(prediction, gt)
        with autocast(enabled=False):
            feat_loss = feature_loss(prediction.float(), gt.float())
            loss = (
                float(config["loss"]["charbonnier_weight"]) * char_loss.float()
                + float(config["loss"]["feature_weight"]) * feat_loss
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(config["train"]["gradient_clip"])
        )
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        running["total"] += float(loss.detach())
        running["charbonnier"] += float(char_loss.detach())
        running["feature"] += float(feat_loss.detach())
        running["count"] += 1
        if iteration % int(config["train"]["print_interval"]) == 0:
            count = running.pop("count")
            record = {
                "iteration": iteration,
                "lr": optimizer.param_groups[0]["lr"],
                "loss": running["total"] / count,
                "charbonnier": running["charbonnier"] / count,
                "feature_loss": running["feature"] / count,
                "seconds": time.time() - interval_start,
            }
            with log_path.open("a") as log_file:
                log_file.write(json.dumps(record) + "\n")
            print(f"QP{args.qp} train: {record}", flush=True)
            running = {"total": 0.0, "charbonnier": 0.0, "feature": 0.0, "count": 0}
            interval_start = time.time()

        validation_metrics = None
        if iteration % int(config["train"]["validation_interval"]) == 0 or iteration == num_iter:
            validation_metrics = validate(model, val_loader, feature_loss, device)
            record = {"iteration": iteration, "validation": validation_metrics}
            with log_path.open("a") as log_file:
                log_file.write(json.dumps(record) + "\n")
            print(f"QP{args.qp} validation: {record}", flush=True)
            if validation_metrics["delta_psnr"] > best_delta:
                best_delta = validation_metrics["delta_psnr"]
                save_checkpoint(
                    output_dir / "best.pth",
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    iteration,
                    best_delta,
                    config,
                    validation_metrics,
                )

        if iteration % int(config["train"]["checkpoint_interval"]) == 0 or iteration == num_iter:
            save_checkpoint(
                output_dir / f"checkpoint_{iteration:06d}.pth",
                model,
                optimizer,
                scheduler,
                scaler,
                iteration,
                best_delta,
                config,
                validation_metrics,
            )
            save_checkpoint(
                output_dir / "last.pth",
                model,
                optimizer,
                scheduler,
                scaler,
                iteration,
                best_delta,
                config,
                validation_metrics,
            )


if __name__ == "__main__":
    main()
