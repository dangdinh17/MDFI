#!/bin/bash

for t in 32 37
do
    echo "🔹 Running: python trainandvalidate_tgaf.py --opt_path configs/train_TGAF_$t.yml ..."
    python trainandvalidate_tgaf.py --opt_path configs/train_TGAF_$t.yml
    echo "✅ Done --qp $t"
    echo "---------------------------"
done
for t in 32 37
do
    echo "🔹 Running: python test_tgaf.py --qp $t ..."
    python test_tgaf.py --qp $t
    echo "✅ Done --qp $t"
    echo "---------------------------"
done
# for t in 32 37
# do
#     echo "🔹 Running: python trainandvalidate_tvqe.py --opt_path configs/train_TVQE_$t.yml ..."
#     python trainandvalidate_tvqe.py --opt_path configs/train_TVQE_$t.yml
#     echo "✅ Done --qp $t"
#     echo "---------------------------"
# done
for t in 32 37
do
    echo "🔹 Running: python test_tvqe.py --qp $t ..."
    python test_tvqe.py --qp $t
    echo "✅ Done --qp $t"
    echo "---------------------------"
done
echo "🎉 All runs finished!"