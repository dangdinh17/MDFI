#!/usr/bin/env python3
"""Extract the N-th frame (default 2) from YUV420p (.yuv) sequences found under a source folder.

Saves PNGs to src/frames2/<subfolder>/<basename>.png

Assumptions:
- filenames contain resolution in the form _{width}x{height}_ (e.g. Foo_1280x720_15.yuv)
- input format is YUV420p (I420): Y plane, then U, then V (subsampled by 2)
"""
import argparse
import os
import re
import sys
from pathlib import Path

try:
    import numpy as np
    from PIL import Image
except Exception as e:
    print("Missing dependency:", e)
    print("Please install dependencies: pip install numpy pillow")
    sys.exit(2)


RES_RE = re.compile(r"_(\d+)x(\d+)_")


def parse_resolution_from_name(fname: str):
    m = RES_RE.search(fname)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def read_yuv_frame(path, width, height, frame_index=0):
    # YUV420p (I420): Y plane size = W*H, U = (W/2)*(H/2), V same
    y_size = width * height
    uv_size = (width // 2) * (height // 2)
    frame_size = y_size + 2 * uv_size

    with open(path, 'rb') as f:
        f.seek(frame_index * frame_size)
        data = f.read(frame_size)
        if len(data) < frame_size:
            raise ValueError(f"File too small for frame {frame_index+1}: expected {frame_size} bytes, got {len(data)}")

    y = np.frombuffer(data[0:y_size], dtype=np.uint8).reshape((height, width))
    u = np.frombuffer(data[y_size:y_size + uv_size], dtype=np.uint8).reshape((height // 2, width // 2))
    v = np.frombuffer(data[y_size + uv_size:y_size + 2 * uv_size], dtype=np.uint8).reshape((height // 2, width // 2))

    # Upsample U and V to full resolution
    u_up = u.repeat(2, axis=0).repeat(2, axis=1)
    v_up = v.repeat(2, axis=0).repeat(2, axis=1)

    # Convert YUV to RGB (BT.601)
    y_f = y.astype(np.float32)
    u_f = u_up.astype(np.float32)
    v_f = v_up.astype(np.float32)

    r = y_f + 1.402 * (v_f - 128.0)
    g = y_f - 0.344136 * (u_f - 128.0) - 0.714136 * (v_f - 128.0)
    b = y_f + 1.772 * (u_f - 128.0)

    rgb = np.stack([r, g, b], axis=2)
    np.clip(rgb, 0, 255, out=rgb)
    rgb = rgb.astype(np.uint8)
    return rgb


def process_folder(src_root: Path, out_root: Path, frame_number: int):
    # frame_number is 1-based (user friendly). Convert to 0-based index
    fi = frame_number - 1
    if not src_root.exists():
        print(f"Source folder {src_root} does not exist")
        return 1

    yuv_paths = list(src_root.rglob('*.yuv'))
    if not yuv_paths:
        print(f"No .yuv files found under {src_root}")
        return 1

    print(f"Found {len(yuv_paths)} .yuv files. Extracting frame {frame_number} from each...")
    for p in yuv_paths:
        rel = p.relative_to(src_root)
        subfolder = rel.parts[0] if len(rel.parts) > 1 else ''
        out_dir = out_root / subfolder
        out_dir.mkdir(parents=True, exist_ok=True)

        basename = p.stem
        res = parse_resolution_from_name(p.name)
        if res is None:
            print(f"Skipping {p}: cannot parse resolution from filename. Expected pattern _<W>x<H>_ in name.")
            continue
        w, h = res
        try:
            rgb = read_yuv_frame(p, w, h, frame_index=fi)
        except Exception as e:
            print(f"Failed to read {p}: {e}")
            continue

        img = Image.fromarray(rgb)
        out_path = out_dir / (basename + f"_f{frame_number}.png")
        img.save(out_path)
        print(f"Saved {out_path}")

    print("Done.")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Extract N-th frame from YUV420p sequences")
    parser.add_argument('--src', type=str, default='data/val_18_15/QP32', help='Source folder containing subfolders with .yuv files')
    parser.add_argument('--frame', type=int, default=2, help='Which frame to extract (1-based). Default 2')
    parser.add_argument('--out', type=str, default=None, help='Output root folder. Default: <src>/frames2')

    args = parser.parse_args()
    src_root = Path(args.src)
    out_root = Path(args.out) if args.out else src_root / 'frames2'

    rc = process_folder(src_root, out_root, args.frame)
    sys.exit(rc)


if __name__ == '__main__':
    main()
