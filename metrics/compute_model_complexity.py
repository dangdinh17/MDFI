
import torch
from pathlib import Path
import time
from thop import profile
from thop import clever_format
import yaml
try:
    import psutil
except Exception:
    psutil = None
# ===== load model =====
# ví dụ: model = TGAF(...) hoặc torch.load("model.pth")
from models.tgaf import TGAF
from models.tvqe import TVQE
from models.stff import STFF
from models.ovqe import OVQE
from models import *
# bạn cần điều chỉnh input shape cho đúng (ví dụ video super-resolution: [B,C,T,H,W] hoặc [B,C,H,W])
with open('configs/train_TGAF.yml', 'r') as fp:
    tgaf = yaml.load(fp, Loader=yaml.FullLoader)
with open('configs/train_TVQE.yml', 'r') as fp:
    tvqe = yaml.load(fp, Loader=yaml.FullLoader)
models = {
    "TGAF": TGAF(tgaf['network']),
    "TVQE": TVQE(tvqe['network']),
    "STFF": STFF(),
    "OVQE": OVQE(),
    "MDFI": MDFI(),
}

WARMUP = 5
REPEATS = 20

results = []
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

for name, model in models.items():
    print('\n---', name)
    model = model.to(device)
    model.eval()

    # build dummy inputs
    if name in ("TGAF", "TVQE"):
        dummy = torch.randn(1, 7, 1280, 720, device=device)
    else:
        dummy = torch.randn(1, 1, 1280, 720, device=device)

    # count params
    params = sum(p.numel() for p in model.parameters())

    # compute FLOPs (may fail for some ops)
    try:
        if name in ('MDFI', 'CVQE'):
            flops, p = profile(model, inputs=(dummy, dummy), verbose=False)
        else:
            flops, p = profile(model, inputs=(dummy,), verbose=False)
    except Exception as e:
        print('FLOP counting failed:', e)
        flops = float('nan')

    # measure latency and peak GPU memory (with warmup)
    avg_time = float('nan')
    peak_mem = None
    try:
        # reset stats
        if device.type == 'cuda':
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        # warmup
        with torch.no_grad():
            for _ in range(WARMUP):
                if name in ('MDFI', 'CVQE'):
                    _ = model(dummy, dummy)
                else:
                    _ = model(dummy)
            if device.type == 'cuda':
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()

            times = []
            for _ in range(REPEATS):
                t0 = time.time()
                if name in ('MDFI', 'CVQE'):
                    _ = model(dummy, dummy)
                else:
                    _ = model(dummy)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                t1 = time.time()
                times.append(t1 - t0)

            avg_time = sum(times) / len(times)
            if device.type == 'cuda':
                peak_mem = torch.cuda.max_memory_allocated()
            elif psutil is not None:
                # best-effort CPU memory (RSS)
                proc = psutil.Process()
                peak_mem = proc.memory_info().rss
    except RuntimeError as e:
        print('Runtime measurement failed:', e)
        # on CUDA OOM or other failure, try a smaller spatial size as fallback
        try:
            print('Retrying measurement with smaller input (640x360)')
            small = torch.randn(1, dummy.size(1), 640, 360, device=device)
            with torch.no_grad():
                for _ in range(WARMUP):
                    _ = model(small) if name not in ('MDFI','CVQE') else model(small, small)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                times = []
                for _ in range(REPEATS):
                    t0 = time.time()
                    _ = model(small) if name not in ('MDFI','CVQE') else model(small, small)
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                    t1 = time.time()
                    times.append(t1 - t0)
                avg_time = sum(times) / len(times)
                if device.type == 'cuda':
                    peak_mem = torch.cuda.max_memory_allocated()
                elif psutil is not None:
                    proc = psutil.Process()
                    peak_mem = proc.memory_info().rss
        except Exception as e2:
            print('Fallback measurement also failed:', e2)

    flops_fmt, params_fmt = clever_format([flops, params], "%.3f") if not (flops!=flops) else (float('nan'), f"{params}")
    print(f"{name}: Params={params_fmt}, FLOPs={flops_fmt}, avg_time_s={avg_time:.4f}, peak_mem_bytes={peak_mem}")
    results.append({'model': name, 'params': params, 'flops': flops, 'avg_time_s': avg_time, 'peak_mem_bytes': peak_mem})

# write CSV
import csv
out_csv = Path(__file__).resolve().parents[1] / 'metrics' / 'model_complexity_720p.csv'
with open(out_csv, 'w', newline='') as cf:
    writer = csv.DictWriter(cf, fieldnames=['model','params','flops','avg_time_s','peak_mem_bytes'])
    writer.writeheader()
    for r in results:
        writer.writerow(r)

print('\nWrote results to', out_csv)
