#!/usr/bin/env python3
"""Evaluate VMAF and LPIPS for output/QP37 models against groundtruth sequences.

Usage: run from repository root (where output/ and data/ live):
    python metrics/evaluate_qp37_metrics.py

Requirements:
    - ffmpeg with libvmaf available on PATH
    - python packages: numpy, pandas, openpyxl, torch, lpips, tqdm

This script expects:
  - output/QP37/<model>/*.yuv  (enhanced Y-only YUV sequences in yuv420p format)
  - data/test_18/gt/*.yuv      (groundtruth sequences with same filenames)

Output:
  - metrics/results_QP37.xlsx with sheets: VMAF and LPIPS
"""
import os
import re
import json
import subprocess
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
from tqdm import tqdm

# === Configuration (edit these defaults directly) ===
# Metrics to compute. Set to None or [] to use all supported metrics.
DEFAULT_METRICS = None
# Models to evaluate (list of folder names in output/QP37). Set to None to evaluate all found models.
DEFAULT_MODELS = None
# Baseline model name for delta calculations
DEFAULT_BASELINE = 'VVC'
# Output Excel path (None -> metrics/results_QP37.xlsx)
DEFAULT_OUT = 'metrics/QP37.xlsx'
# ==================================================

try:
    from skimage.metrics import peak_signal_noise_ratio as sk_psnr
    from skimage.metrics import structural_similarity as sk_ssim
except Exception:
    sk_psnr = None
    sk_ssim = None


def parse_seq_filename(filename: str) -> Tuple[str, Tuple[int, int], int]:
    """Parse sequence name like BasketballDrill_832x480_500.yuv -> (name, (w,h), tot_frm)
    Note: resolution string is WxH in filename.
    """
    stem = Path(filename).stem
    m = re.match(r"(?P<name>.+)_(?P<w>\d+)x(?P<h>\d+)_(?P<frm>\d+)$", stem)
    if not m:
        raise ValueError(f"Filename {filename} does not match expected pattern")
    name = m.group('name')
    w = int(m.group('w'))
    h = int(m.group('h'))
    frm = int(m.group('frm'))
    return name, (w, h), frm


def import_yuv(seq_path: str, h: int, w: int, tot_frm: int, yuv_type='420p', start_frm=0, only_y=True, verbose=False):
    """Load Y, U, and V channels separately from an 8bit yuv420p video.
    Returns y_seq (tot_frm, h, w) if only_y True else (y_seq, u_seq, v_seq).
    """
    if yuv_type == '420p':
        hh, ww = h // 2, w // 2
    elif yuv_type == '444p':
        hh, ww = h, w
    else:
        raise Exception('yuv_type not supported.')

    y_size, u_size, v_size = h * w, hh * ww, hh * ww
    blk_size = y_size + u_size + v_size

    y_seq = np.zeros((tot_frm, h, w), dtype=np.uint8)
    if not only_y:
        u_seq = np.zeros((tot_frm, hh, ww), dtype=np.uint8)
        v_seq = np.zeros((tot_frm, hh, ww), dtype=np.uint8)

    with open(seq_path, 'rb') as fp:
        if verbose:
            print(seq_path)
        for i in range(tot_frm):
            fp.seek(int(blk_size * (start_frm + i)), 0)
            y_frm = np.fromfile(fp, dtype=np.uint8, count=y_size).reshape(h, w)
            if only_y:
                y_seq[i, ...] = y_frm
            else:
                u_frm = np.fromfile(fp, dtype=np.uint8, count=u_size).reshape(hh, ww)
                v_frm = np.fromfile(fp, dtype=np.uint8, count=v_size).reshape(hh, ww)
                y_seq[i, ...], u_seq[i, ...], v_seq[i, ...] = y_frm, u_frm, v_frm

    if only_y:
        return y_seq
    else:
        return y_seq, u_seq, v_seq


def compute_vmaf(ref_path: str, dist_path: str, w: int, h: int, model_path: str = None) -> float:
    """Compute aggregate VMAF score using ffmpeg + libvmaf. Inputs are raw yuv420p files.
    Returns the aggregate VMAF score (float) or np.nan on error.
    """
    # prefer provided model path, else let libvmaf use default
    # create a unique log path per dist to avoid overwriting and to help debugging
    dist_stem = Path(dist_path).stem.replace('.', '_')
    log_path = f'metrics/logs/vmaf_{dist_stem}.json'
    model_arg = '/home/u9564043/vmaf_model/vmaf_v0.6.1.json'
    # if model_path:
    #     model_arg = f'model=path={model_path}:'

    cmd = [
        './ffmpeg',
        '-f', 'rawvideo', '-pixel_format', 'yuv420p', '-video_size', f'{w}x{h}', '-i', ref_path,
        '-f', 'rawvideo', '-pixel_format', 'yuv420p', '-video_size', f'{w}x{h}', '-i', dist_path,
        '-lavfi', f"[0:v]setpts=PTS-STARTPTS[ref];[1:v]setpts=PTS-STARTPTS[dist];[ref][dist]libvmaf=model=path={model_arg}:log_path={log_path}:log_fmt=json",
        '-f', 'null', '-'
    ]

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        print(f"ffmpeg/libvmaf failed (returncode={proc.returncode}). stderr:\n{proc.stderr}")
        # keep stderr for debugging
        errlog = f'vmaf_{dist_stem}.stderr.txt'
        try:
            with open(errlog, 'w') as ef:
                ef.write(proc.stderr)
            print(f'Wrote ffmpeg stderr to {errlog}')
        except Exception:
            pass
        return float('nan')

    # try to load the JSON log and extract aggregate VMAF
    if not os.path.exists(log_path):
        print(f'libvmaf did not produce log {log_path}. ffmpeg stderr:\n{proc.stderr}')
        return float('nan')

    try:
        with open(log_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f'Failed to read/parse {log_path}: {e}')
        print('ffmpeg stderr:\n', proc.stderr)
        return float('nan')

    # agg = data.get('aggregate')
    # if not agg:
    #     print(f'No aggregate field in {log_path}; full JSON:')
    #     print(json.dumps(data, indent=2)[:10000])
    #     return float('nan')

    pooled = data.get('pooled_metrics') or data.get('pooled_metrics_v2')
    # Try several common locations for an aggregate VMAF value.
    # 1) data['aggregate'] usually contains per-asset aggregate when running on one asset
    agg = data.get('aggregate') or {}
    for key in ('VMAF', 'vmaf', 'aggregate_value', 'VMAF_score', 'vmaf_score'):
        if key in agg and agg[key] is not None:
            try:
                return float(agg[key])
            except Exception:
                pass

    # 2) pooled_metrics (when libvmaf returns pooled metrics across assets)
    pooled = data.get('pooled_metrics') or data.get('pooled') or {}
    if isinstance(pooled, dict):
        # pooled may have structure like {'vmaf': {'mean': 92.3, ...}, ...}
        v = pooled.get('vmaf') or pooled.get('VMAF')
        if isinstance(v, dict):
            for subk in ('mean', 'avg', 'value'):
                if subk in v and v[subk] is not None:
                    try:
                        return float(v[subk])
                    except Exception:
                        pass
        else:
            # pooled.vmaf might itself be a number
            try:
                return float(v)
            except Exception:
                pass

    # 3) fallback: average per-frame metric in data['frames']
    frames = data.get('frames') or []
    v_list = []
    for f in frames:
        # sometimes metrics stored under f['metrics']
        metrics = f.get('metrics') if isinstance(f.get('metrics'), dict) else f
        for key in ('vmaf', 'VMAF', 'VMAF_score', 'vmaf_score'):
            if isinstance(metrics, dict) and key in metrics and metrics[key] is not None:
                try:
                    v_list.append(float(metrics[key]))
                except Exception:
                    pass
    if v_list:
        return float(sum(v_list) / len(v_list))

    # nothing found
    print(f'No VMAF value found in {log_path}; JSON keys: {list(data.keys())}')
    return float('nan')


def compute_lpips_for_seq(ref_y: np.ndarray, dist_y: np.ndarray, loss_fn=None, device=None) -> float:
    """Compute mean LPIPS for two Y sequences. Both arrays shape (n,h,w), dtype uint8.
    Uses lpips package on replicated Y->RGB.
    Returns mean LPIPS (float).
    """
    # loss_fn (LPIPS model) and device can be provided to avoid re-creating model per call
    try:
        import torch
    except Exception:
        raise RuntimeError('torch is required for LPIPS')
    if loss_fn is None:
        try:
            import lpips
        except Exception:
            raise RuntimeError('LPIPS dependencies not installed: lpips')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        loss_fn = lpips.LPIPS(net='alex').to(device)
    else:
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    n = min(ref_y.shape[0], dist_y.shape[0])
    if n == 0:
        return float('nan')

    # batch on GPU for better throughput; on CPU use batch_size=1 to reduce memory
    batch_size = 16 if device.type == 'cuda' else 1
    tot = 0.0

    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        a_batch = ref_y[start:end].astype(np.float32) / 255.0  # (b,h,w)
        b_batch = dist_y[start:end].astype(np.float32) / 255.0

        # convert to tensors with 3 channels replicated from Y, shape (b,3,h,w)
        ta = torch.from_numpy(np.stack([a_batch, a_batch, a_batch], axis=1)).to(device)
        tb = torch.from_numpy(np.stack([b_batch, b_batch, b_batch], axis=1)).to(device)

        # scale to [-1,1]
        ta = ta * 2.0 - 1.0
        tb = tb * 2.0 - 1.0

        with torch.no_grad():
            d = loss_fn(ta, tb)
            # lpips returns shape (b,1,1,1) or (b,1)
            d = d.view(d.size(0), -1).mean(dim=1)
            tot += float(d.sum().cpu().item())

    return tot / n


def compute_psnr_ssim_for_seq(ref_y: np.ndarray, dist_y: np.ndarray) -> Tuple[float, float]:
    """Compute mean PSNR and SSIM for two Y sequences. Both arrays shape (n,h,w), dtype uint8.
    Returns (psnr_mean, ssim_mean).
    """
    if ref_y.shape[0] == 0:
        return float('nan'), float('nan')

    # PSNR: if skimage available use it; otherwise compute MSE-based PSNR
    psnrs = []
    ssims = []
    for i in range(min(ref_y.shape[0], dist_y.shape[0])):
        a = ref_y[i].astype(np.float32)
        b = dist_y[i].astype(np.float32)
        # PSNR
        if sk_psnr is not None:
            try:
                p = sk_psnr(a, b, data_range=255)
            except Exception:
                # fallback
                mse = np.mean((a - b) ** 2)
                p = 10 * np.log10((255.0 ** 2) / mse) if mse > 0 else float('inf')
        else:
            mse = np.mean((a - b) ** 2)
            p = 10 * np.log10((255.0 ** 2) / mse) if mse > 0 else float('inf')
        psnrs.append(p)

        # SSIM
        if sk_ssim is not None:
            try:
                s = sk_ssim(a, b, data_range=255)
            except Exception:
                s = float('nan')
        else:
            s = float('nan')
        ssims.append(s)

    psnr_mean = float(np.mean([v for v in psnrs if np.isfinite(v)])) if any(np.isfinite(psnrs)) else float('nan')
    ssim_mean = float(np.mean([v for v in ssims if not np.isnan(v)])) if any([not np.isnan(v) for v in ssims]) else float('nan')
    return psnr_mean, ssim_mean


def main():
    repo_root = Path(__file__).resolve().parents[1]
    # Use predefined defaults instead of CLI arguments. Edit the DEFAULT_* constants at
    # the top of this file to change which metrics/models are evaluated.
    metrics_arg = DEFAULT_METRICS
    models_arg = DEFAULT_MODELS
    baseline = DEFAULT_BASELINE
    out_arg = DEFAULT_OUT

    out_base = repo_root / 'output' / 'QP37'
    gt_dir = repo_root / 'data' / 'test_18' / 'gt'
    if not out_base.exists():
        print('output/QP37 not found')
        return
    if not gt_dir.exists():
        print('data/test_18/gt not found')
        return

    found_models = [p.name for p in out_base.iterdir() if p.is_dir()]
    if models_arg:
        # filter to only requested models that exist
        models = [m for m in models_arg if m in found_models]
        missing = [m for m in models_arg if m not in found_models]
        if missing:
            print(f'Warning: requested models not found and will be skipped: {missing}')
    else:
        models = found_models

    # collect sequence filenames across selected models (union)
    seq_files = set()
    model_files = {}
    for m in models:
        mdir = out_base / m
        files = sorted([p.name for p in mdir.glob('*.yuv')])
        model_files[m] = files
        seq_files.update(files)
    seq_files = sorted(seq_files)

    # determine which metrics to compute
    default_metrics = ['VMAF', 'LPIPS', 'PSNR', 'SSIM']
    if metrics_arg:
        metrics = [m.upper() for m in metrics_arg]
        # validate
        for mm in metrics:
            if mm not in default_metrics:
                raise ValueError(f'Unsupported metric: {mm}')
    else:
        metrics = default_metrics

    # prepare result tables for requested metrics
    metric_tables = {m: pd.DataFrame(index=seq_files, columns=models, dtype=float) for m in metrics}

    # instantiate LPIPS model once (if requested and available) to speed up
    lpips_model = None
    lpips_device = None
    if 'LPIPS' in metrics:
        try:
            import torch
            import lpips
            lpips_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            lpips_model = lpips.LPIPS(net='alex').to(lpips_device)
        except Exception:
            print('LPIPS model not available; LPIPS values will be NaN')
            lpips_model = None
            lpips_device = None

    # main loop
    for seq in tqdm(seq_files, desc='sequences'):
        gt_path = gt_dir / seq
        if not gt_path.exists():
            print(f'GT not found for {seq}, skipping')
            continue
        try:
            name, (w, h), frm = parse_seq_filename(seq)
        except Exception as e:
            print('Failed parse', seq, e)
            continue

        # load reference Y once per sequence if needed by any metric
        need_ref = any(m in metrics for m in ('LPIPS', 'PSNR', 'SSIM'))
        ref_y = None
        if need_ref:
            ref_y = import_yuv(str(gt_path), h, w, frm, yuv_type='420p', only_y=True)

        for m in models:
            if seq not in model_files.get(m, []):
                for met in metrics:
                    metric_tables[met].at[seq, m] = float('nan')
                continue
            enh_path = out_base / m / seq

            # compute VMAF
            if 'VMAF' in metrics:
                v = compute_vmaf(str(gt_path), str(enh_path), w, h)
                metric_tables['VMAF'].at[seq, m] = v

            # load dist Y only if needed
            need_dist = any(met in metrics for met in ('LPIPS', 'PSNR', 'SSIM'))
            dist_y = None
            if need_dist:
                dist_y = import_yuv(str(enh_path), h, w, frm, yuv_type='420p', only_y=True)

            if 'PSNR' in metrics or 'SSIM' in metrics:
                try:
                    p, s = compute_psnr_ssim_for_seq(ref_y, dist_y)
                except Exception as e:
                    print('PSNR/SSIM computation failed:', e)
                    p, s = float('nan'), float('nan')
                if 'PSNR' in metrics:
                    metric_tables['PSNR'].at[seq, m] = p
                if 'SSIM' in metrics:
                    metric_tables['SSIM'].at[seq, m] = s

            if 'LPIPS' in metrics:
                if lpips_model is not None:
                    try:
                        lp = compute_lpips_for_seq(ref_y, dist_y, loss_fn=lpips_model, device=lpips_device)
                    except Exception as e:
                        print('LPIPS computation failed:', e)
                        lp = float('nan')
                else:
                    lp = float('nan')
                metric_tables['LPIPS'].at[seq, m] = lp

    # compute deltas vs baseline
    delta_tables = {}
    if baseline in models:
        for met in metrics:
            base_series = metric_tables[met][baseline]
            delta_tables[met] = metric_tables[met].subtract(base_series, axis=0)
    else:
        print(f'Baseline model "{baseline}" not found among models: {models}. Deltas will be NaN.')
        for met in metrics:
            delta_tables[met] = pd.DataFrame(index=seq_files, columns=models, dtype=float)

    # write excel with requested metric sheets and delta sheets
    out_xlsx = Path(out_arg) if out_arg else repo_root / 'metrics' / 'results_QP37.xlsx'
    mode = 'a' if out_xlsx.exists() else 'w'

    if mode == 'a':
        writer = pd.ExcelWriter(out_xlsx, engine='openpyxl', mode='a', if_sheet_exists='replace')
    else:
        writer = pd.ExcelWriter(out_xlsx, engine='openpyxl', mode='w')

    with writer:
        for met in metrics:
            metric_tables[met].to_excel(writer, sheet_name=met)
        for met in metrics:
            delta_tables[met].to_excel(writer, sheet_name=f'{met}_delta')

    print('Wrote', out_xlsx)

if __name__ == '__main__':
    main()
