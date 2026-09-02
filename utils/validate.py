import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import torch
import numpy as np
from collections import OrderedDict
import math
import utils
from tqdm import tqdm
import glob
import os.path as op
import numpy as np
import os
import pandas as pd

def validate(model, gt_dir, lq_dir, pd_dir):
    gt_video_list = sorted(glob.glob(op.join(gt_dir, '*.yuv')))
    lq_video_list = sorted(glob.glob(op.join(lq_dir, '*.yuv')))
    pd_video_list = sorted(glob.glob(op.join(pd_dir, '*.yuv')))
    model.eval()

    # ==========
    # Load entire video
    # ==========
    avg_delta_psnr, avg_delta_ssim = 0, 0

    for cdx in tqdm(range(len(gt_video_list))):
        torch.cuda.empty_cache()
        raw_yuv_path = gt_video_list[cdx]
        lq_yuv_path = lq_video_list[cdx]
        pd_yuv_path = pd_video_list[cdx]
        vname = raw_yuv_path.split("/")[-1].split('.')[0]
        _, wxh, nfs = vname.split('_')
        nfs = int(nfs)
        w, h = int(wxh.split('x')[0]), int(wxh.split('x')[1])
        divide_bolck = 15
        divide = math.ceil(nfs / divide_bolck)
        add_frame = 0

        raw_y = utils.import_yuv(
            seq_path=raw_yuv_path, h=h, w=w, tot_frm=nfs, start_frm=0, only_y=True, verbose=False
            )
        raw_y = raw_y.astype(np.float32) / 255.

        lq_y = utils.import_yuv(
            seq_path=lq_yuv_path, h=h, w=w, tot_frm=nfs, start_frm=0, only_y=True, verbose=False
        )
        lq_y = lq_y.astype(np.float32) / 255.

        pd_y = utils.import_yuv(
            seq_path=pd_yuv_path, h=h, w=w, tot_frm=nfs, start_frm=0, only_y=True, verbose=False
        )
        pd_y = pd_y.astype(np.float32) / 255.

        # msg = '> yuv loaded.'
        # print(msg)

        # ==========
        # Test
        # ==========
        unit = 'dB'
        # pbar = tqdm(total=nfs, ncols=80)
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
        #     pbar.set_description(
        #         "[{:.3f}] {:s} -> [{:.3f}] {:s}"
        #         .format(batch_ori, unit, batch_perf, unit)
        #         )
        #     pbar.update()

        # pbar.close()
        ori_ = ori_psnr_counter.get_ave()
        enh_ = enh_psnr_counter.get_ave()
        ori_ssim = ori_ssim_counter.get_ave()
        enh_ssim = enh_ssim_counter.get_ave()
        avg_delta_psnr += enh_ - ori_
        avg_delta_ssim += enh_ssim - ori_ssim
        # print(enh_ - ori_, enh_ssim - ori_ssim)
    # print(len(gt_video_list))
    
    avg_delta_psnr = avg_delta_psnr / len(gt_video_list)
    avg_delta_ssim = avg_delta_ssim / len(gt_video_list)
    return avg_delta_psnr, avg_delta_ssim
        
def validate_ovqe(model, gt_dir, lq_dir, pd_dir):
    gt_video_list = sorted(glob.glob(op.join(gt_dir, '*.yuv')))
    lq_video_list = sorted(glob.glob(op.join(lq_dir, '*.yuv')))
    pd_video_list = sorted(glob.glob(op.join(pd_dir, '*.yuv')))
    model.eval()

    # ==========
    # Load entire video
    # ==========
    avg_delta_psnr, avg_delta_ssim = 0, 0

    for cdx in tqdm(range(len(gt_video_list))):
        torch.cuda.empty_cache()
        raw_yuv_path = gt_video_list[cdx]
        lq_yuv_path = lq_video_list[cdx]
        pd_yuv_path = pd_video_list[cdx]
        vname = raw_yuv_path.split("/")[-1].split('.')[0]
        _, wxh, nfs = vname.split('_')
        nfs = int(nfs)
        w, h = int(wxh.split('x')[0]), int(wxh.split('x')[1])
        divide_bolck = 15
        divide = math.ceil(nfs / divide_bolck)
        add_frame = 0

        raw_y = utils.import_yuv(
            seq_path=raw_yuv_path, h=h, w=w, tot_frm=nfs, start_frm=0, only_y=True, verbose=False
            )
        raw_y = raw_y.astype(np.float32) / 255.

        lq_y = utils.import_yuv(
            seq_path=lq_yuv_path, h=h, w=w, tot_frm=nfs, start_frm=0, only_y=True, verbose=False
        )
        lq_y = lq_y.astype(np.float32) / 255.

        pd_y = utils.import_yuv(
            seq_path=pd_yuv_path, h=h, w=w, tot_frm=nfs, start_frm=0, only_y=True, verbose=False
        )
        pd_y = pd_y.astype(np.float32) / 255.

        # msg = '> yuv loaded.'
        # print(msg)

        # ==========
        # Test
        # ==========
        unit = 'dB'
        # pbar = tqdm(total=nfs, ncols=80)
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
                        enc_all = model(lq_y[:, ccc * divide_bolck:(ccc + 1) * divide_bolck + add_frame, :,:].contiguous())
                        enhanced[:, ccc * divide_bolck:(ccc + 1) * divide_bolck, :, :, :] = enc_all[:,:divide_bolck,:, :,:]
                    elif ccc == divide - 1:
                        enc_all = model(lq_y[:, ccc * divide_bolck - add_frame:, :, :].contiguous())
                        enhanced[:, ccc * divide_bolck:, :, :, :] = enc_all[:, add_frame:, :, :, :]
                    else:
                        enc_all = model(lq_y[:, ccc * divide_bolck - add_frame:(ccc + 1) * divide_bolck + add_frame, :,:].contiguous())
                        enhanced[:, ccc * divide_bolck:(ccc + 1) * divide_bolck, :, :, :] = enc_all[:,add_frame:divide_bolck + add_frame, :, :,:]
            else:
                add_h_w = 4
                for bbb in range(2):
                    if bbb == 0:
                        for ccc in range(divide):
                            if ccc == 0:
                                enc_all = model(lq_y[:, ccc * divide_bolck:(ccc + 1) * divide_bolck + add_frame, :,:int(w / 2) + add_h_w].contiguous())
                                enhanced[:, ccc * divide_bolck:(ccc + 1) * divide_bolck, :, :, :int(w / 2)] = enc_all[:,:divide_bolck,:, :,:int(w / 2)]
                            elif ccc == divide - 1:
                                enc_all = model(lq_y[:, ccc * divide_bolck - add_frame:, :, :int(w / 2) + add_h_w].contiguous())
                                enhanced[:, ccc * divide_bolck:, :, :, :int(w / 2)] = enc_all[:, add_frame:, :, :,:int(w / 2)]
                            else:
                                enc_all = model(lq_y[:, ccc * divide_bolck - add_frame:(ccc + 1) * divide_bolck + add_frame, :,:int(w / 2) + add_h_w].contiguous())
                                enhanced[:, ccc * divide_bolck:(ccc + 1) * divide_bolck, :, :, :int(w / 2)] = enc_all[:,add_frame:divide_bolck + add_frame,:, :,:int(w / 2)]
                    else:
                        for ccc in range(divide):
                            if ccc == 0:
                                enc_all = model(lq_y[:, ccc * divide_bolck:(ccc + 1) * divide_bolck + add_frame, :,int(w / 2) - add_h_w:w].contiguous())
                                enhanced[:, ccc * divide_bolck:(ccc + 1) * divide_bolck, :, :, int(w / 2):   w] = enc_all[:,:divide_bolck,:, :, add_h_w:]
                            elif ccc == divide - 1:
                                enc_all = model(lq_y[:, ccc * divide_bolck - add_frame:, :, int(w / 2) - add_h_w:w].contiguous())
                                enhanced[:, ccc * divide_bolck:, :, :, int(w / 2):w] = enc_all[:, add_frame:, :, :, add_h_w:]
                            else:
                                enc_all = model(lq_y[:, ccc * divide_bolck - add_frame:(ccc + 1) * divide_bolck + add_frame, :,int(w / 2) - add_h_w:w].contiguous())
                                enhanced[:, ccc * divide_bolck:(ccc + 1) * divide_bolck, :, :, int(w / 2):w] = enc_all[:, add_frame:divide_bolck + add_frame,:, :, add_h_w:]

        enhanced = np.float32(enhanced.cpu())
        lq_y = np.float32(lq_y.cpu())
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
        #     pbar.set_description(
        #         "[{:.3f}] {:s} -> [{:.3f}] {:s}"
        #         .format(batch_ori, unit, batch_perf, unit)
        #         )
        #     pbar.update()

        # pbar.close()
        ori_ = ori_psnr_counter.get_ave()
        enh_ = enh_psnr_counter.get_ave()
        ori_ssim = ori_ssim_counter.get_ave()
        enh_ssim = enh_ssim_counter.get_ave()
        avg_delta_psnr += enh_ - ori_
        avg_delta_ssim += enh_ssim - ori_ssim
        # print(enh_ - ori_, enh_ssim - ori_ssim)
    # print(len(gt_video_list))
    
    avg_delta_psnr = avg_delta_psnr / len(gt_video_list)
    avg_delta_ssim = avg_delta_ssim / len(gt_video_list)
    return avg_delta_psnr, avg_delta_ssim
        
def validate_tgaf(model, gt_dir, lq_dir, pd_dir):
    gt_video_list = sorted(glob.glob(op.join(gt_dir, '*.yuv')))
    lq_video_list = sorted(glob.glob(op.join(lq_dir, '*.yuv')))
    pd_video_list = sorted(glob.glob(op.join(pd_dir, '*.yuv')))
    model.eval()

    # ==========
    # Load entire video
    # ==========
    avg_delta_psnr, avg_delta_ssim = 0, 0

    for cdx in tqdm(range(len(gt_video_list))):
        torch.cuda.empty_cache()
        raw_yuv_path = gt_video_list[cdx]
        lq_yuv_path = lq_video_list[cdx]
        pd_yuv_path = pd_video_list[cdx]
        vname = raw_yuv_path.split("/")[-1].split('.')[0]
        _, wxh, nfs = vname.split('_')
        nfs = int(nfs)
        w, h = int(wxh.split('x')[0]), int(wxh.split('x')[1])
        divide_bolck = 15
        divide = math.ceil(nfs / divide_bolck)
        add_frame = 0

        raw_y = utils.import_yuv(
            seq_path=raw_yuv_path, h=h, w=w, tot_frm=nfs, start_frm=0, only_y=True, verbose=False
            )
        raw_y = raw_y.astype(np.float32) / 255.

        lq_y = utils.import_yuv(
            seq_path=lq_yuv_path, h=h, w=w, tot_frm=nfs, start_frm=0, only_y=True, verbose=False
        )
        lq_y = lq_y.astype(np.float32) / 255.

        pd_y = utils.import_yuv(
            seq_path=pd_yuv_path, h=h, w=w, tot_frm=nfs, start_frm=0, only_y=True, verbose=False
        )
        pd_y = pd_y.astype(np.float32) / 255.

        # msg = '> yuv loaded.'
        # print(msg)

        # ==========
        # Test
        # ==========
        unit = 'dB'
        # pbar = tqdm(total=nfs, ncols=80)
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
            for idx in range(nfs):
        # chọn indices cho cửa sổ 7 frame
                idxs = [min(max(i, 0), nfs - 1) for i in range(idx - 3, idx + 3 + 1)]  # len=7
                # print(idxs)
                # stack thành input clip
                idx_tensor = torch.tensor(idxs, device=lq_y.device, dtype=torch.long)
                clip = lq_y.index_select(1, idx_tensor)  # (7, H, W)
                # clip = clip.unsqueeze(0)  # (1, 7, 1, H, W)
                # print(clip.shape)
                with torch.no_grad():
                    pred = model(clip)  # giả sử output (1, 1, H, W)

                enhanced[:, idx, 0, :, :] = pred[:, 0, :, :]
        enhanced = np.float32(enhanced.cpu())
        lq_y = np.float32(lq_y.cpu())
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
        #     pbar.set_description(
        #         "[{:.3f}] {:s} -> [{:.3f}] {:s}"
        #         .format(batch_ori, unit, batch_perf, unit)
        #         )
        #     pbar.update()

        # pbar.close()
        ori_ = ori_psnr_counter.get_ave()
        enh_ = enh_psnr_counter.get_ave()
        ori_ssim = ori_ssim_counter.get_ave()
        enh_ssim = enh_ssim_counter.get_ave()
        avg_delta_psnr += enh_ - ori_
        avg_delta_ssim += enh_ssim - ori_ssim
        # print(enh_ - ori_, enh_ssim - ori_ssim)
    # print(len(gt_video_list))
    
    avg_delta_psnr = avg_delta_psnr / len(gt_video_list)
    avg_delta_ssim = avg_delta_ssim / len(gt_video_list)
    return avg_delta_psnr, avg_delta_ssim