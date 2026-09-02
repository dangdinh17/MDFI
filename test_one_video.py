import torch
import numpy as np
from collections import OrderedDict
from models.mdfi import MDFI
import utils
from tqdm import tqdm
import models
import cv2
import os
from PIL import Image
import yaml
# modeltype = 'STFF'
# modeltype = 'OVQE'
# modeltype = 'TGAF'
# modeltype = 'TVQE'
# data/test_18/QP37/pd/.yuv data/test_18/QP37/pd/.yuv
seqs = {
    'KristenAndSara_1280x720_600': 1,
    'Johnny_1280x720_600': 1,
    'BQSquare_416x240_600': 1,
    'BasketballDrill_832x480_500': 14,
    'BQMall_832x480_600': 0,
}
# seqname = 'Johnny_1280x720_600'
# data/test_18/QP37/lq/FourPeople_1280x720_600.yuv

# data/test_18/QP37/pd/BasketballPass_416x240_500.yuv
# data/test_18/QP37/pd/data/test_18/QP37/pd/Kimono_1920x1080_240.yuv.yuv
seqnames = [
# "PeopleOnStreet_2560x1600_150",
# "ParkScene_1920x1080_240",
# "Traffic_2560x1600_150",
# "BQTerrace_1920x1080_600",
# "BasketballDrive_1920x1080_500",
# "RaceHorses_832x480_300",
# "BasketballDrill_832x480_500",
# "Cactus_1920x1080_500",
# "PartyScene_832x480_500",
# "BQMall_832x480_600",
# "Johnny_1280x720_600",
# "FourPeople_1280x720_600",
# "RaceHorses_416x240_300",
"BasketballPass_416x240_500",
# "Kimono_1920x1080_240",
# "BlowingBubbles_416x240_500",
# "BQSquare_416x240_600",
# "KristenAndSara_1280x720_600",
]
# seqname = 'RaceHorses_832x480_300'
qp=37
for seqname in seqnames:
    raw_yuv_path = f'data/test_18/gt/{seqname}.yuv'
    lq_yuv_path =  f'data/test_18/QP{qp}/lq/{seqname}.yuv'
    pd_yuv_path =  f'data/test_18/QP{qp}/pd/{seqname}.yuv'
    vname = lq_yuv_path.split("/")[-1].split('.')[0]
    _, wxh, nfs = vname.split('_')
    nfs = 30
    w, h = int(wxh.split('x')[0]), int(wxh.split('x')[1])


    torch.cuda.set_device(0)
    log_fp = open(f'visual_log/{seqname}.txt', 'w')
    def save_img(y, u, v, idx, outputfolder):
            # ===== U, V =====
            Y = y[idx]
            U = u[idx]
            V = v[idx]

            # Upsample
            H, W = Y.shape
            U_up = cv2.resize(U, (W, H), interpolation=cv2.INTER_LINEAR)
            V_up = cv2.resize(V, (W, H), interpolation=cv2.INTER_LINEAR)

            # YUV → RGB
            yuv = np.stack([Y, U_up, V_up], axis=2)
            # yuv = np.stack([Y, U_up, V_up], axis=2)
            rgb = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
            # print(rgb.shape)
            # Save PNG
            img = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
            img.save(f"{outputfolder}/frame_{idx:04d}.png")

    for modeltype in ['MDFI', 'OVQE', 'TGAF', 'TVQE']:
        with open(f'configs/train_{modeltype}.yml', 'r') as fp:
            opts_dict = yaml.load(fp, Loader=yaml.FullLoader)
        if modeltype == 'MDFI':
            ckp_path = f'exp/{modeltype}_QP{qp}/best_weight.pth'
        else:
            ckp_path = f'exp/{modeltype}_QP{qp}/best_weight.pth'
        getmodel = getattr(models, modeltype)
        if modeltype == 'TVQE' or modeltype == 'TGAF':
            model = getmodel(opts_dict['network'])
        else:
            model = getmodel()
        # print(model)
        print(f'loading model {ckp_path}...')
        checkpoint = torch.load(ckp_path, map_location='cpu')
        if 'module.' in list(checkpoint['state_dict'].keys())[0]:
            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                name = k[7:]
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(checkpoint['state_dict'])

        print(f'> model {ckp_path} loaded.')
        model = model.cuda()
        model.eval()
        radius = 3
        # Load video
        print('loading raw and low-quality yuv...')
        gt_y, gt_u, gt_v = utils.import_yuv(seq_path=raw_yuv_path, h=h, w=w, tot_frm=nfs, start_frm=0, only_y=False)
        lq_y, lq_u, lq_v = utils.import_yuv(seq_path=lq_yuv_path, h=h, w=w, tot_frm=nfs, start_frm=0, only_y=False)
        lq_y_copy = lq_y
        pd_y = utils.import_yuv(seq_path=pd_yuv_path, h=h, w=w, tot_frm=nfs, start_frm=0, only_y=True)
        lq_y = lq_y.astype(np.float32) / 255.
        gt_y = gt_y.astype(np.float32) / 255.
        print('> yuv loaded.')
        print(lq_y.shape)
        # Criterion
        criterion = utils.PSNR()
        unit = 'dB'

        # Test
        
        ori_psnr_counter = utils.Counter()
        enh_psnr_counter = utils.Counter()
        ori_ssim_counter = utils.Counter()
        enh_ssim_counter = utils.Counter()
        lq_y = torch.from_numpy(lq_y)
        lq_y = torch.unsqueeze(lq_y, 0).cuda()
        if modeltype == 'MDFI':
            pd_y = pd_y.astype(np.float32) / 255.
            pd_y = torch.from_numpy(pd_y)
            pd_y = torch.unsqueeze(pd_y, 0).cuda()
        with torch.no_grad():
            if modeltype == 'MDFI':
                enc_all = model(lq_y, pd_y)
            elif modeltype == 'OVQE' or modeltype == 'STFF':
                enc_all = model(lq_y)
            else:
                enc_all = torch.from_numpy(np.zeros([1, nfs, 1, h, w]))
                for idx in range(nfs):
            # chọn indices cho cửa sổ 7 frame
                    idxs = [min(max(i, 0), nfs - 1) for i in range(idx - radius, idx + radius + 1)]  # len=7
                    idx_tensor = torch.tensor(idxs, device=lq_y.device, dtype=torch.long)
                    clip = lq_y.index_select(1, idx_tensor)  # (7, H, W)
                    with torch.no_grad():
                        pred = model(clip)  # giả sử output (1, 1, H, W)
                    enc_all[:, idx, 0, :, :] = pred[:, 0, :, :]
        # Open output file
        # out_file = open(f"output/{modeltype}_{seqname}.yuv", "wb")  # Tạo hoặc ghi đè lên tệp
        # out_folder = f"/tmp/output/{modeltype}_{seqname}" 
        # RAW_folder = f"/tmp/output/RAW_{seqname}" 
        # VVC_folder = f"/tmp/output/VVC_{seqname}" 
        # os.makedirs(out_folder, exist_ok=True)
        # os.makedirs(RAW_folder, exist_ok=True)
        # os.makedirs(VVC_folder, exist_ok=True)
        print(enc_all.shape)
        print(lq_y.shape)
        # pbar = tqdm(total=nfs, ncols=80)
        enc_all = np.float32(enc_all.cpu())
        lq_y = np.float32(lq_y.cpu())
        # log_fp.write(f'{modeltype}')
        for idx in range(nfs):

            # # Chuyển đổi và lưu khung hình enc_all
            # enh_frm = enc_all[0, idx, 0, :, :].cpu().numpy()
            # enh_frm = (enh_frm * 255.0).round().astype(np.uint8)

            # # ===== U, V =====
            # # Y = lq_y_copy[idx]
            # U = lq_u[idx]
            # V = lq_v[idx]

            # # Upsample
            # H, W = enh_frm.shape
            # U_up = cv2.resize(U, (W, H), interpolation=cv2.INTER_LINEAR)
            # V_up = cv2.resize(V, (W, H), interpolation=cv2.INTER_LINEAR)

            # # YUV → RGB
            # yuv = np.stack([enh_frm, U_up, V_up], axis=2)
            # # yuv = np.stack([Y, U_up, V_up], axis=2)
            # rgb = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
            # # print(rgb.shape)
            # # Save PNG
            # img = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
            # img.save(f"{out_folder}/frame_{idx:04d}.png")
            # save_img(lq_y_copy, lq_u, lq_v, idx, VVC_folder)
            # save_img(gt_y, gt_u, gt_v, idx, RAW_folder)
            # pbar.update()
            batch_ori = utils.calculate_psnr(lq_y[0, idx,...], gt_y[idx],data_range=1.0)
            batch_perf = utils.calculate_psnr(enc_all[0, idx, 0,:,:], gt_y[idx],data_range=1.0)
            ssim_ori = utils.calculate_ssim(lq_y[0, idx,...], gt_y[idx],data_range=1.0)
            ssim_perf = utils.calculate_ssim(enc_all[0, idx, 0,:,:], gt_y[idx], data_range=1.0)
            msg = "Model {:<5} Frame {:<2} psnr [{:.3f}] enh_psnr [{:.3f}] ssim [{:.5f}] enh_ssim [{:.5f}]".format(
                modeltype, idx, batch_ori, batch_perf,ssim_ori, ssim_perf
                )
            print(msg)
            log_fp.write(msg + '\n')
            log_fp.flush()
            ori_psnr_counter.accum(volume=batch_ori)
            enh_psnr_counter.accum(volume=batch_perf)
            ori_ssim_counter.accum(volume=ssim_ori)
            enh_ssim_counter.accum(volume=ssim_perf)
        # out_file.close()  # Đóng tệp sau khi hoàn tất
        # pbar.close()
        # ori_ = ori_psnr_counter.get_ave()
        # enh_ = enh_psnr_counter.get_ave()
        # print('ave ori [{:.3f}] {:s}, enh [{:.3f}] {:s}, delta [{:.3f}] {:s}'.format(
        #     ori_, unit, enh_, unit, (enh_ - ori_) , unit
        #     ))
        print('> done.')
