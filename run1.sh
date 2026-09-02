#!/bin/bash

for t in 22 27
do
    echo "🔹 Running: python trainandvalidate_tgaf.py --opt_path configs/train_TGAF_$t.yml ..."
    python trainandvalidate_tgaf.py --opt_path configs/train_TGAF_$t.yml
    echo "✅ Done --qp $t"
    echo "---------------------------"
done
for t in 22 27
do
    echo "🔹 Running: python test_tgaf.py --qp $t ..."
    python test_tgaf.py --qp $t
    echo "✅ Done --qp $t"
    echo "---------------------------"
done
for t in 22 27
do
    echo "🔹 Running: python test_tvqe.py --qp $t ..."
    python test_tvqe.py --qp $t
    echo "✅ Done --qp $t"
    echo "---------------------------"
done
echo "🎉 All runs finished!"