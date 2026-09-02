#!/bin/bash

ROOT="data/MOT20_processed"   # đổi đường dẫn nếu cần

# Danh sách thư mục cần convert
folders=("original" "QP22" "QP27" "QP32" "QP37" "QP42")

for folder in "${folders[@]}"; do

    input_dir="$ROOT/$folder"
    output_dir="$ROOT/${folder}_yuv"

    mkdir -p "$output_dir"

    echo "=== Converting $folder ==="

    for f in "$input_dir"/*.mp4; do
        
        [ -e "$f" ] || continue  # skip nếu không có file

        name=$(basename "$f" .mp4)

        echo "-> $name.mp4 => $name.yuv"

        ffmpeg -y -i "$f" \
            -pix_fmt yuv420p \
            "$output_dir/${name}.yuv"

    done

    echo "Done $folder"
    echo
done

echo ">>> All sequences converted to YUV successfully."
