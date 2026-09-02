#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"

python_bin=${PYTHON_BIN:-/home/u9564043/anaconda3/envs/ovqe/bin/python}

"$python_bin" scripts/generate_feature_pqf_labels.py --qp 22 27 32 37

for qp in 22 27 32 37; do
    detector="exp/MFQEv2_feature_QP${qp}/detector/best.pth"
    if [[ ! -f "$detector" ]]; then
        "$python_bin" train_pqf_detector.py --qp "$qp"
    else
        echo "Detector QP${qp} already exists; skipping: $detector"
    fi
done

exec bash scripts/start_mfqev2_training.sh
