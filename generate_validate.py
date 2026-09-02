import os

sub_frame = 50
# đường dẫn folder chứa các file .yuv
# folder = "/work/u9564043/OVQE_VVC/OVQE_Prior/OVQE/data/val_18_15/gt"

# danh sách file muốn giữ lại
keep_files = {
    "BasketballPass_416x240_500.yuv",
    "BQSquare_416x240_600.yuv",
    "RaceHorses_416x240_300.yuv",

}

def yuv_read_write(input_path, output_path, width, height, num_frames):
    # Tạo thư mục cha nếu chưa có
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    frame_size = width * height * 3 // 2  # YUV420
    
    with open(input_path, "rb") as f:
        with open(output_path, "wb") as fout:
            for i in range(num_frames):
                frame = f.read(frame_size)
                if len(frame) < frame_size:
                    print(f"File kết thúc ở frame {i}")
                    break
                fout.write(frame)
        print(f"Đã lưu frame")

# for qp in [27, 32, 37, 22]:
# # for qp in [42]:
#     fold = f'data/test_18/QP{qp}'
#     out = f'data/val_18_{sub_frame}/QP{qp}'
    
    # fold = f'data/test_18/QP{qp}/QP{qp}_predicted'
    # out = f'data/val_18_15/QP{qp}/pd'
# folder = "/work/u9564043/OVQE_VVC/OVQE_Prior/OVQE/data/test_18/gt"
# out_folder = f"/work/u9564043/OVQE_VVC/OVQE_Prior/OVQE/data/val_18_{sub_frame}/gt"
folder = "/work/u9564043/OVQE_VVC/OVQE_Prior/OVQE/data/test_18/QP32/pd"
out_folder = f"/work/u9564043/OVQE_VVC/OVQE_Prior/OVQE/data/val_18_{sub_frame}/QP32/pd"
    # for sub in os.listdir(fold):
    #     folder = os.path.join(fold, sub)
    #     out_folder = os.path.join(out, sub)
os.makedirs(out_folder, exist_ok=True)
for filename in os.listdir(folder):
    if filename.endswith(".yuv") and filename in keep_files:
        file_path = os.path.join(folder, filename)
        vname, wxh, nfs = filename.split('.')[0].split('_')
        # print(vname, wxh, nfs)
        w, h = wxh.split('x')
        newfile = f'{vname}_{wxh}_{sub_frame}.yuv' 
        out_path = os.path.join(out_folder, newfile)
        print(file_path, out_path)
        yuv_read_write(file_path, out_path, int(w), int(h), sub_frame)