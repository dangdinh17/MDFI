import subprocess

model_names = ["TGAF", "TVQE", "STFF", "OVQE", "MDFI"]
model_names = ["MDFI", "MDFI_64_1", "MDFI_32_16", "MDFI_32_1", "OVQE"]  # Chỉ chạy MDFI để kiểm tra thời gian, bỏ qua các model khác

for name in model_names:
    print(f"\n===== Running {name} =====")
    subprocess.run(["python", "benchmark_single.py", name])