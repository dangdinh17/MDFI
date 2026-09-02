#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"

python_bin=${PYTHON_BIN:-/home/u9564043/anaconda3/envs/ovqe/bin/python}
config=${MFQEV2_CONFIG:-configs/train_MFQEv2_feature.yml}
qp37_init=${QP37_INIT:-exp/MFQEv2_feature_QP37/enhancer/checkpoint_002000.pth}

if [[ ! -x "$python_bin" ]]; then
    echo "Python executable not found: $python_bin" >&2
    exit 1
fi

if [[ ! -f "$config" ]]; then
    echo "Training config not found: $config" >&2
    exit 1
fi

for qp in 22 27 32 37; do
    label="datasets/108data/feature_pqf/QP${qp}_lpips_alex.json"
    if [[ ! -f "$label" ]]; then
        echo "Feature-PQF label not found: $label" >&2
        echo "Run: $python_bin scripts/generate_feature_pqf_labels.py --qp 22 27 32 37" >&2
        exit 1
    fi
done

qp37_args=()
if [[ "$qp37_init" != "none" ]]; then
    if [[ ! -f "$qp37_init" ]]; then
        echo "QP37 initialization checkpoint not found: $qp37_init" >&2
        echo "Set QP37_INIT=none to train QP37 from scratch." >&2
        exit 1
    fi
    qp37_args=(--init "$qp37_init")
fi

echo "Training MFQEv2 in order: QP37 -> QP32 -> QP27 -> QP22"
echo "Config: $config"
echo "QP37 initialization: $qp37_init"

"$python_bin" train_mfqev2_feature.py --config "$config" --qp 37 "${qp37_args[@]}"
"$python_bin" train_mfqev2_feature.py --config "$config" --qp 32 \
    --init exp/MFQEv2_feature_QP37/enhancer/best.pth
"$python_bin" train_mfqev2_feature.py --config "$config" --qp 27 \
    --init exp/MFQEv2_feature_QP32/enhancer/best.pth
"$python_bin" train_mfqev2_feature.py --config "$config" --qp 22 \
    --init exp/MFQEv2_feature_QP27/enhancer/best.pth

echo "All MFQEv2 training jobs completed. Check exp/MFQEv2_feature_QP*/enhancer/."
