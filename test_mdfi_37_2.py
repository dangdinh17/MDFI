import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import torch
import numpy as np
from collections import OrderedDict
import math
from models.mdfi import MDFI
import utils
from tqdm import tqdm
import glob
import os.path as op
import numpy as np
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '--type', type=int, default=0,
    help='Weight type.'
)
args = parser.parse_args()
weight_type = args.type
QP = 37
if weight_type == 0:
    ckp_path = f'exp/MDFI_QP37_64_1/ckp_11000.pth'
elif weight_type == 1:
    ckp_path = 'exp/QP37_training_fixLR/best_psnr_weight.pth'
elif weight_type == 2:
    ckp_path = 'exp/QP37_training_fixLR/best_ssim_weight.pth'
# ckp_path = 'model/ckp_500000.pth'
# ckp_path = 'models/ckp_20000.pth'
# gt_dir = '/kaggle/input/testhighres/gt/Out'
# lq_dir = '/kaggle/input/testhighres/lr/Out'
# pd_dir = '/kaggle/input/testhighres/pd/Out'
gt_dir = './data/test_18/gt'
lq_dir = f'./data/test_18/QP{QP}/lq'
pd_dir = f'./data/test_18/QP{QP}/pd'
# gt_dir = '/kaggle/input/testset/QP37/gt'
# lq_dir = '/kaggle/input/testset/QP37/lr'
# pd_dir = '/kaggle/input/testset/QP37/pd'
log_fp = open(f'log/log_test_MDFI_QP{QP}_64_1.txt', 'a')
gt_video_list = sorted(glob.glob(op.join(gt_dir, '*.yuv')))
lq_video_list = sorted(glob.glob(op.join(lq_dir, '*.yuv')))
pd_video_list = sorted(glob.glob(op.join(pd_dir, '*.yuv')))
torch.cuda.set_device(0)

resolution_to_divide_block = {
    '2560x1600': [150],
    '1920x1080': [500, 240],
    '1280x720': [105],
    '832x480': [105, 300, 600],
    '416x240': [105, 300, 500],
}

def get_divide_block(wxh, nfs):
    if wxh in resolution_to_divide_block:
        divide_block_list = resolution_to_divide_block[wxh]
    else:
        divide_block_list = nfs

    for block_size in divide_block_list:
        if nfs == block_size:
            return block_size
    return divide_block_list[0]  # Returns the first value in the list as the default



def save_results_to_excel(results, excel_path):
    df = pd.DataFrame(results)
    df.to_excel(excel_path, index=False)
    print("Results saved to Excel file.")
    

def main():

    model = MDFI(out_nc=64, num_sft=1)
    msg = f'loading model {ckp_path}...'
    print(msg)
        # , map_location='cpu'   ,map_location={'cuda:0': 'cuda:1'}
    checkpoint = torch.load(ckp_path, map_location='cpu')
    if 'module.' in list(checkpoint['state_dict'].keys())[0]:  # multi-gpu training
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            name = k[7:]  # remove module
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:  # single-gpu training
        model.load_state_dict(checkpoint['state_dict'])

    msg = f'> model {ckp_path} loaded.'
    print(msg)
    model = model.cuda()
    model.eval()

    # ==========
    # Load entire video
    # ==========
    results = []
    testing_timer = utils.system.Timer()
    for cdx in range(8, len(gt_video_list)):
        raw_yuv_path = gt_video_list[cdx]
        lq_yuv_path = lq_video_list[cdx]
        pd_yuv_path = pd_video_list[cdx]
        vname = raw_yuv_path.split("/")[-1].split('.')[0]
        _, wxh, nfs = vname.split('_')
        nfs = int(nfs)
        w, h = int(wxh.split('x')[0]), int(wxh.split('x')[1])
        # divide_bolck = get_divide_block(wxh, nfs)
        divide_bolck = 25
        divide = math.ceil(nfs / divide_bolck)
        add_frame = 0

        msg = f'loading raw and low-quality yuv...'
        print(msg)
        raw_y = utils.import_yuv(
            seq_path=raw_yuv_path, h=h, w=w, tot_frm=nfs, start_frm=0, only_y=True
            )
        raw_y = raw_y.astype(np.float32) / 255.

        lq_y = utils.import_yuv(
            seq_path=lq_yuv_path, h=h, w=w, tot_frm=nfs, start_frm=0, only_y=True
        )
        lq_y = lq_y.astype(np.float32) / 255.

        pd_y = utils.import_yuv(
            seq_path=pd_yuv_path, h=h, w=w, tot_frm=nfs, start_frm=0, only_y=True
        )
        pd_y = pd_y.astype(np.float32) / 255.

        msg = '> yuv loaded.'
        print(msg)



        # ==========
        # Test
        # ==========
        unit = 'dB'
        pbar = tqdm(total=nfs, ncols=80)
        ori_psnr_counter = utils.Counter()
        enh_psnr_counter = utils.Counter()

        ori_ssim_counter = utils.Counter()
        enh_ssim_counter = utils.Counter()

        lq_y = torch.from_numpy(lq_y)
        lq_y = torch.unsqueeze(lq_y, 0).cuda()

        pd_y = torch.from_numpy(pd_y)
        pd_y = torch.unsqueeze(pd_y, 0).cuda()
        
        enhanced = torch.from_numpy(np.zeros([1, nfs, 1, h, w]))
        with torch.no_grad():
            if h<=720:
                for ccc in range(divide):
                    if ccc == 0:
                        enc_all = model(lq_y[:, ccc * divide_bolck:(ccc + 1) * divide_bolck + add_frame, :,:].contiguous(),
                                        pd_y[:, ccc * divide_bolck:(ccc + 1) * divide_bolck + add_frame, :,:].contiguous())
                        enhanced[:, ccc * divide_bolck:(ccc + 1) * divide_bolck, :, :, :] = enc_all[:,:divide_bolck,:, :,:]
                    elif ccc == divide - 1:
                        enc_all = model(lq_y[:, ccc * divide_bolck - add_frame:, :, :].contiguous(),
                                        pd_y[:, ccc * divide_bolck - add_frame:, :, :].contiguous())
                        enhanced[:, ccc * divide_bolck:, :, :, :] = enc_all[:, add_frame:, :, :, :]
                    else:
                        enc_all = model(lq_y[:, ccc * divide_bolck - add_frame:(ccc + 1) * divide_bolck + add_frame, :,:].contiguous(),
                                        pd_y[:, ccc * divide_bolck - add_frame:(ccc + 1) * divide_bolck + add_frame, :,:].contiguous())
                        enhanced[:, ccc * divide_bolck:(ccc + 1) * divide_bolck, :, :, :] = enc_all[:,add_frame:divide_bolck + add_frame, :, :,:]
            else:
                add_h_w = 4
                for bbb in range(2):
                    if bbb == 0:
                        for ccc in range(divide):
                            if ccc == 0:
                                enc_all = model(lq_y[:, ccc * divide_bolck:(ccc + 1) * divide_bolck + add_frame, :,:int(w / 2) + add_h_w].contiguous(),
                                                pd_y[:, ccc * divide_bolck:(ccc + 1) * divide_bolck + add_frame, :,:int(w / 2) + add_h_w].contiguous(),)
                                enhanced[:, ccc * divide_bolck:(ccc + 1) * divide_bolck, :, :, :int(w / 2)] = enc_all[:,:divide_bolck,:, :,:int(w / 2)]
                            elif ccc == divide - 1:
                                enc_all = model(lq_y[:, ccc * divide_bolck - add_frame:, :, :int(w / 2) + add_h_w].contiguous(),
                                                pd_y[:, ccc * divide_bolck - add_frame:, :, :int(w / 2) + add_h_w].contiguous())
                                enhanced[:, ccc * divide_bolck:, :, :, :int(w / 2)] = enc_all[:, add_frame:, :, :,:int(w / 2)]
                            else:
                                enc_all = model(lq_y[:, ccc * divide_bolck - add_frame:(ccc + 1) * divide_bolck + add_frame, :,:int(w / 2) + add_h_w].contiguous(),
                                                pd_y[:, ccc * divide_bolck - add_frame:(ccc + 1) * divide_bolck + add_frame, :,:int(w / 2) + add_h_w].contiguous())
                                enhanced[:, ccc * divide_bolck:(ccc + 1) * divide_bolck, :, :, :int(w / 2)] = enc_all[:,add_frame:divide_bolck + add_frame,:, :,:int(w / 2)]
                    else:
                        for ccc in range(divide):
                            if ccc == 0:
                                enc_all = model(lq_y[:, ccc * divide_bolck:(ccc + 1) * divide_bolck + add_frame, :,int(w / 2) - add_h_w:w].contiguous(),
                                                pd_y[:, ccc * divide_bolck:(ccc + 1) * divide_bolck + add_frame, :,int(w / 2) - add_h_w:w].contiguous(),)
                                enhanced[:, ccc * divide_bolck:(ccc + 1) * divide_bolck, :, :, int(w / 2):   w] = enc_all[:,:divide_bolck,:, :, add_h_w:]
                            elif ccc == divide - 1:
                                enc_all = model(lq_y[:, ccc * divide_bolck - add_frame:, :, int(w / 2) - add_h_w:w].contiguous(),
                                                pd_y[:, ccc * divide_bolck - add_frame:, :, int(w / 2) - add_h_w:w].contiguous())
                                enhanced[:, ccc * divide_bolck:, :, :, int(w / 2):w] = enc_all[:, add_frame:, :, :, add_h_w:]
                            else:
                                enc_all = model(lq_y[:, ccc * divide_bolck - add_frame:(ccc + 1) * divide_bolck + add_frame, :,int(w / 2) - add_h_w:w].contiguous(),
                                                pd_y[:, ccc * divide_bolck - add_frame:(ccc + 1) * divide_bolck + add_frame, :,int(w / 2) - add_h_w:w].contiguous())
                                enhanced[:, ccc * divide_bolck:(ccc + 1) * divide_bolck, :, :, int(w / 2):w] = enc_all[:, add_frame:divide_bolck + add_frame,:, :, add_h_w:]

        enhanced = np.float32(enhanced.cpu())
        lq_y = np.float32(lq_y.cpu())
        iteration_time = testing_timer.get_interval()
        for idx in range(nfs):
            batch_ori = utils.calculate_psnr(lq_y[0, idx,...], raw_y[idx],data_range=1.0)
            batch_perf = utils.calculate_psnr(enhanced[0, idx, 0,:,:], raw_y[idx],data_range=1.0)
            ssim_ori = utils.calculate_ssim(lq_y[0, idx,...], raw_y[idx],data_range=1.0)
            ssim_perf = utils.calculate_ssim(enhanced[0, idx, 0,:,:], raw_y[idx], data_range=1.0)

            ori_psnr_counter.accum(volume=batch_ori)
            enh_psnr_counter.accum(volume=batch_perf)
            ori_ssim_counter.accum(volume=ssim_ori)
            enh_ssim_counter.accum(volume=ssim_perf)

            # display
            pbar.set_description(
                "[{:.3f}] {:s} -> [{:.3f}] {:s}"
                .format(batch_ori, unit, batch_perf, unit)
                )
            pbar.update()

        pbar.close()
        ori_ = ori_psnr_counter.get_ave()
        enh_ = enh_psnr_counter.get_ave()
        ori_ssim = ori_ssim_counter.get_ave()
        enh_ssim = enh_ssim_counter.get_ave()
        msg = "VideoName {:<29} ave: ori [{:.3f}] {:s}, enh [{:.3f}] {:s}, delta [{:.3f}] {:s}  ave ori_ssim [{:.5f}], enh_ssim [{:.5f}], delta_ssim [{:.4f}], time [{:.1f}]".format(
            vname,ori_, unit, enh_, unit, (enh_ - ori_) , unit, ori_ssim, enh_ssim, (enh_ssim - ori_ssim)*100, iteration_time
            )
        print(msg)
        log_fp.write(msg + '\n')
        log_fp.flush()
        results.append({
            "VideoName": vname,
            "ori_psnr": ori_,
            "enh_psnr": enh_,
            "delta_psnr": enh_ - ori_,
            "ori_ssim": ori_ssim,
            "enh_ssim": enh_ssim,
            "delta_ssim": (enh_ssim - ori_ssim) * 100,
        })

if __name__ == '__main__':
    main()
