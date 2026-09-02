import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import torch
import time
import yaml
import sys
import gc
from pathlib import Path

from fvcore.nn import FlopCountAnalysis
from thop import profile
from thop import clever_format
import pandas as pd

from models.tgaf import TGAF
from models.tvqe import TVQE
from models.stff import STFF
from models.ovqe import OVQE
from models import *
torch.set_grad_enabled(False)
xlsx_name = 'QP37_complexity.xlsx'
# ===== nhận tên model =====
name = sys.argv[1]

# ===== load config =====
with open('configs/train_TGAF.yml', 'r') as fp:
    tgaf = yaml.load(fp, Loader=yaml.FullLoader)

with open('configs/train_TVQE.yml', 'r') as fp:
    tvqe = yaml.load(fp, Loader=yaml.FullLoader)

# ===== init model =====
if name == "TGAF":
    model = TGAF(tgaf['network'])
elif name == "TVQE":
    model = TVQE(tvqe['network'])
elif name == "STFF":
    model = STFF()
elif name == "OVQE":
    model = OVQE()
elif name == "MDFI":
    model = MDFI()
elif name == "MDFI_64_1":
    model = MDFI(out_nc=64, num_sft=1)
elif name == "MDFI_32_16":
    model = MDFI(out_nc=32, num_sft=16)
elif name == "MDFI_32_1":
    model = MDFI(out_nc=32, num_sft=1)
    
else:
    raise ValueError("Unknown model")

device = torch.device("cuda")
model = model.to(device).eval()

print(f"\n{name}:")
nfs = 7
# h, w = 1080, 1920
h, w = 720, 1280
# ===== INPUT =====
if name in ["TGAF", "TVQE"]:
    inp = torch.randn(1, 7, w, h).to(device)
    inputs = inp
    input_thop = (inp, )
elif name in ['MDFI', 'CVQE', 'MDFI_64_1', 'MDFI_32_16', 'MDFI_32_1']:
    inp1 = torch.randn(1, nfs, w, h).to(device)
    inp2 = torch.randn(1, nfs, w, h).to(device)
    inputs = (inp1, inp2)
    input_thop = (inp1, inp2, )
else:
    inp = torch.randn(1, nfs, w, h).to(device)
    inputs = inp
    input_thop = (inp, )

# ===== FLOPs =====
# fvcore
# flops = FlopCountAnalysis(model, inputs)
# flops = flops.unsupported_ops_warnings(False)
# flops = flops.uncalled_modules_warnings(False)


# thop
import copy
model_flops = copy.deepcopy(model)
with torch.no_grad():
    flops, params = profile(model_flops, input_thop, verbose=False, custom_ops={})
# flops, params = clever_format([flops, params], "%.3f")
del model_flops

try:
    total_flops = flops / 1e12
except:
    total_flops = flops.total() / 1e12
if name not in ["TGAF", "TVQE"]:
    total_flops /= nfs 
print(f"  FLOPs:  {total_flops:.3f} T")

# # ===== PARAMS =====
total_params = params / 1e6
print(f"  Params: {total_params:.2f} M")

# ===== MEMORY =====
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()

with torch.no_grad():
    # for _ in range(loops):
    _ = model(*inputs) if isinstance(inputs, tuple) else model(inputs)

torch.cuda.synchronize()
mem = torch.cuda.max_memory_allocated() / 1024**3
if name not in ["TGAF", "TVQE"]:
    mem /= nfs
print(f"  Memory: {mem:.2f} GB")

# ===== LATENCY =====
warmup, runs = 10, 5

with torch.no_grad():
    for _ in range(warmup):
        _ = model(*inputs) if isinstance(inputs, tuple) else model(inputs)

torch.cuda.synchronize()

start = time.time()
with torch.no_grad():
    for _ in range(runs):
        _ = model(*inputs) if isinstance(inputs, tuple) else model(inputs)

torch.cuda.synchronize()
latency = (time.time() - start) / runs * 1000
if name not in ["TGAF", "TVQE"]:
    latency /= nfs
fps = 1000 / latency

print(f"  Time:   {latency:.2f} ms")
print(f"  FPS:    {fps:.2f}")

# === write results to Excel ===
try:
    out_xlsx = Path(__file__).resolve().parents[0] / 'metrics' / xlsx_name
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)

    # prepare row values (store numeric values with clear units)
    row = {
        'FLOPs (T)': float(total_flops),
        'Params (M)': float(total_params),
        'Memory (GB)': float(mem),
        'Time (ms)': float(latency),
        'FPS': float(fps)
    }

    # load existing table if present (robustly). Ensure columns exist.
    cols = ['FLOPs (T)', 'Params (M)', 'Memory (GB)', 'Time (ms)', 'FPS']
    if out_xlsx.exists():
        try:
            existing = pd.read_excel(out_xlsx, sheet_name='benchmarks', index_col=0)
            # if the sheet was empty or had no columns, create empty with desired cols
            if existing is None or existing.shape[1] == 0:
                existing = pd.DataFrame(columns=cols)
            else:
                # ensure it has the expected columns (add missing ones)
                for c in cols:
                    if c not in existing.columns:
                        existing[c] = float('nan')
        except Exception:
            existing = pd.DataFrame(columns=cols)
    else:
        existing = pd.DataFrame(columns=cols)

    # create a one-row DataFrame for this model and upsert into existing
    new_row = pd.DataFrame([row], index=[name])
    # Ensure columns order
    new_row = new_row.reindex(columns=cols)

    # remove old row if exists and append new
    if name in existing.index:
        existing = existing.drop(index=name)
    existing = pd.concat([existing, new_row], axis=0)

    # write back (replace sheet)
    with pd.ExcelWriter(out_xlsx, engine='openpyxl', mode='w') as writer:
        existing.to_excel(writer, sheet_name='benchmarks')
    print('  Wrote benchmark results to', out_xlsx)
except Exception as e:
    print('  Failed to write Excel results:', e)

# ===== CLEAN (optional, nhưng process sẽ tự kill) =====
del model
gc.collect()
torch.cuda.empty_cache()