import torch
import numpy as np
import argparse
from collections import OrderedDict
from mdfi import MDFI
import utils
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_name', type=str, required=True,
                        help="Tên video theo định dạng: BasketballDrill_832x480_500 (không có đuôi .yuv)")
    parser.add_argument('--nfs', type=int, default=30,
                        help="Số frame của video")
    return parser.parse_args()

def main():
    args = parse_args()
    vname = args.video_name
    nfs = args.nfs
    ckp_path = 'exp//QP37_training_fixLR/best_psnr_weight.pth'
    raw_yuv_path = f'data/test_18/gt/{vname}.yuv'
    lq_yuv_path = f'data/test_18/QP37/lq/{vname}.yuv'
    pd_yuv_path = f'data/test_18/QP37/pd/{vname}.yuv'
    
    _, wxh, _ = vname.split('_')
    w, h = int(wxh.split('x')[0]), int(wxh.split('x')[1])
    torch.cuda.set_device(0)

    model = MDFI()
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

    print('loading raw and low-quality yuv...')
    raw_y, raw_u, raw_v = utils.import_yuv(seq_path=raw_yuv_path, h=h, w=w, tot_frm=nfs, start_frm=0, only_y=False)
    lq_y, lq_u, lq_v = utils.import_yuv(seq_path=lq_yuv_path, h=h, w=w, tot_frm=nfs, start_frm=0, only_y=False)
    pd_y, pd_u, pd_v = utils.import_yuv(seq_path=lq_yuv_path, h=h, w=w, tot_frm=nfs, start_frm=0, only_y=False)

    raw_y = raw_y.astype(np.float32) / 255.
    lq_y = lq_y.astype(np.float32) / 255.
    pd_y = pd_y.astype(np.float32) / 255

    raw_u = raw_u.astype(np.float32) / 255.
    lq_u = lq_u.astype(np.float32) / 255.
    pd_u = pd_u.astype(np.float32) / 255.

    raw_v = raw_v.astype(np.float32) / 255.
    lq_v = lq_v.astype(np.float32) / 255.
    pd_v = lq_v.astype(np.float32) / 255.
    print('> yuv loaded.')

    criterion = utils.PSNR()
    unit = 'dB'

    pbar = tqdm(total=nfs, ncols=80)
    ori_psnr_counter = utils.Counter()
    enh_psnr_counter = utils.Counter()

    lq_y = torch.from_numpy(lq_y)[None, ...].cuda()
    lq_u = torch.from_numpy(lq_u)[None, ...].cuda()
    lq_v = torch.from_numpy(lq_v)[None, ...].cuda()
    pd_y = torch.from_numpy(pd_y)[None, ...].cuda()
    pd_u = torch.from_numpy(pd_u)[None, ...].cuda()
    pd_v = torch.from_numpy(pd_v)[None, ...].cuda()

    with torch.no_grad():
        enc_all_y = model(lq_y, pd_y)
        enc_all_u = model(lq_u, pd_u)
        enc_all_v = model(lq_v, pd_v)

    out_file = open(f"{vname}.yuv", "wb")

    for idx in range(nfs):
        gt_frm = torch.from_numpy(raw_y[idx])
        batch_ori = criterion(lq_y[0, idx, ...].cpu(), gt_frm)
        batch_perf = criterion(enc_all_y[0, idx, 0, :, :].cpu(), gt_frm)
        print(f"Frame {idx+1:03d}: Ori PSNR = {batch_ori:.3f} dB, Enh PSNR = {batch_perf:.3f} dB")
        ori_psnr_counter.accum(volume=batch_ori)
        enh_psnr_counter.accum(volume=batch_perf)

        enh_frm_y = (enc_all_y[0, idx, 0, :, :].cpu().numpy() * 255).round().astype(np.uint8)
        enh_frm_u = (enc_all_u[0, idx, 0, :, :].cpu().numpy() * 255).round().astype(np.uint8)
        enh_frm_v = (enc_all_v[0, idx, 0, :, :].cpu().numpy() * 255).round().astype(np.uint8)

        out_file.write(enh_frm_y.tobytes())
        out_file.write(enh_frm_u.tobytes())
        out_file.write(enh_frm_v.tobytes())

        pbar.set_description(
            "[{:.3f}] {:s} -> [{:.3f}] {:s}".format(batch_ori, unit, batch_perf, unit))
        pbar.update()

    out_file.close()
    pbar.close()
    ori_ = ori_psnr_counter.get_ave()
    enh_ = enh_psnr_counter.get_ave()
    print('ave ori [{:.3f}] {:s}, enh [{:.3f}] {:s}, delta [{:.3f}] {:s}'.format(
        ori_, unit, enh_, unit, (enh_ - ori_), unit))
    print('> done.')

if __name__ == '__main__':
    main()
