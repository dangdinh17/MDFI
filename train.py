import os
import math
from torch.cuda.amp import autocast, GradScaler
import time
import yaml
import argparse
import torch
import torch.optim as optim
import os.path as op
import numpy as np
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
from collections import OrderedDict
import utils  # my tool box
import dataset
from models.mdfi import MDFI

from PIL import Image
import numpy as np

# Chuẩn hóa về [0, 1] nếu cần
def normalize(tensor):
    t_min = tensor.min()
    t_max = tensor.max()
    if t_max - t_min == 0:
        return torch.zeros_like(tensor)
    return (tensor - t_min) / (t_max - t_min)

def tensor_to_pil(t):
    t = normalize(t).numpy()
    t = (t * 255).astype(np.uint8)
    return Image.fromarray(t, mode='L')

def receive_arg():
    """Process all hyper-parameters and experiment settings.
    Record in opts_dict."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--opt_path', type=str, default='option_R3_mfqev2_1D.yml',
        help='Path to option YAML file.'
    )
    parser.add_argument(
        '--local_rank', type=int, default=0,
        help='Distributed launcher requires.'
    )
    args = parser.parse_args()

    with open(args.opt_path, 'r') as fp:
        opts_dict = yaml.load(fp, Loader=yaml.FullLoader)

    opts_dict['opt_path'] = args.opt_path
    opts_dict['train']['rank'] = args.local_rank

    if opts_dict['train']['exp_name'] == None:
        opts_dict['train']['exp_name'] = utils.get_timestr()

    opts_dict['train']['log_path'] = op.join(
        "exp", opts_dict['train']['exp_name'], "log.log"
    )
    opts_dict['train']['checkpoint_save_path_pre'] = op.join(
        "exp", opts_dict['train']['exp_name'], "ckp_"
    )

    opts_dict['train']['num_gpu'] = torch.cuda.device_count()
    if opts_dict['train']['num_gpu'] > 1:
        opts_dict['train']['is_dist'] = True
    else:
        opts_dict['train']['is_dist'] = False

    return opts_dict

def main():
    # ==========
    # parameters
    # ==========

    opts_dict = receive_arg()
    rank = opts_dict['train']['rank']
    unit = opts_dict['train']['criterion']['unit']
    num_iter = int(opts_dict['train']['num_iter'])
    interval_print = int(opts_dict['train']['interval_print'])
    interval_val = int(opts_dict['train']['interval_val'])

    # ==========
    # init distributed training
    # ==========
    if opts_dict['train']['is_dist']:
        utils.init_dist(
            local_rank=rank,
            backend='nccl'
        )
    pass

    if rank == 0:
        log_dir = op.join("exp", opts_dict['train']['exp_name'])
        print("log_dir", log_dir)
        utils.mkdir(log_dir)
        log_fp = open(opts_dict['train']['log_path'], 'w')

        # log all parameters
        msg = (
            f"{'<' * 10} Hello {'>' * 10}\n"
            f"Timestamp: [{utils.get_timestr()}]\n"
            f"\n{'<' * 10} Options {'>' * 10}\n"
            f"{utils.dict2str(opts_dict)}"
        )
        print(msg)
        log_fp.write(msg + '\n')
        log_fp.flush()

    # ==========
    # TO-DO: init tensorboard
    # ==========
    pass

    seed = opts_dict['train']['random_seed']
    # >I don't know why should rs + rank
    utils.set_random_seed(seed + rank)

    torch.backends.cudnn.benchmark = True  # speed up
    # torch.backends.cudnn.deterministic = True  # if reproduce


    # create datasets
    train_ds_type = opts_dict['dataset']['train']['type']
    radius = opts_dict['network']['radius']
    assert train_ds_type in dataset.__all__, \
        "Not implemented!"
    train_ds_cls = getattr(dataset, train_ds_type)
    train_ds = train_ds_cls(
        opts_dict=opts_dict['dataset']['train'],
        radius=radius
        )

    # create datasamplers
    train_sampler = utils.DistSampler(
        dataset=train_ds,
        num_replicas=opts_dict['train']['num_gpu'],
        rank=rank,
        ratio=opts_dict['dataset']['train']['enlarge_ratio']
    )

    # create dataloaders
    train_loader = utils.create_dataloader(
        dataset=train_ds,
        opts_dict=opts_dict,
        sampler=train_sampler,
        phase='train',
        seed=opts_dict['train']['random_seed']
    )

    assert train_loader is not None

    batch_size = opts_dict['dataset']['train']['batch_size_per_gpu'] * \
                 opts_dict['train']['num_gpu']  # divided by all GPUs
    num_iter_per_epoch = math.ceil(len(train_ds) * \
                                   opts_dict['dataset']['train']['enlarge_ratio'] / batch_size)
    num_epoch = math.ceil(num_iter / num_iter_per_epoch)

    # create dataloader prefetchers
    tra_prefetcher = utils.CPUPrefetcher(train_loader)

    # ==========
    # create model    ,find_unused_parameters=True
    # ==========
    model = MDFI()
    model = model.to(rank)
    if opts_dict['train']['is_dist']:
        model = DDP(model, device_ids=[rank],find_unused_parameters=True)

    # load pre-trained generator      ,, map_location='cpu'  ,map_location={'cuda:0':'cuda:1'})
    # # Load pre-trained generator


    # # Load pre-trained generator
    # ckp_path = 'model/pre_weight.pth'
    # checkpoint = torch.load(ckp_path, map_location='cpu')
    # state_dict = checkpoint['state_dict']

    # # Xử lý multi-GPU ↔ single-GPU
    # if ('module.' in list(state_dict.keys())[0]) and (not opts_dict['train']['is_dist']):  
    #     new_state_dict = OrderedDict()
    #     for k, v in state_dict.items():
    #         name = k[7:]  # Loại bỏ 'module.'
    #         new_state_dict[name] = v
    # elif ('module.' not in list(state_dict.keys())[0]) and (opts_dict['train']['is_dist']):  
    #     new_state_dict = OrderedDict()
    #     for k, v in state_dict.items():
    #         name = 'module.' + k  # Thêm 'module.'
    #         new_state_dict[name] = v
    # else:
    #     new_state_dict = state_dict

    # # Chỉ giữ lại các trọng số có cùng kích thước
    # model_dict = model.state_dict()
    # filtered_state_dict = {k: v for k, v in new_state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}

    # # Cập nhật trọng số
    # model_dict.update(filtered_state_dict)
    # model.load_state_dict(model_dict, strict=False)

    # print(f'Loaded from {ckp_path} with strict=False, skipping mismatched layers')

    ckp_path = 'model/ckp_360000.pth'
    checkpoint = torch.load(ckp_path, map_location='cpu')
    state_dict = checkpoint['state_dict']
    if ('module.' in list(state_dict.keys())[0]) and (not opts_dict['train']['is_dist']):  # multi-gpu pre-trained -> single-gpu training
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove module
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        print(f'loaded from1 {ckp_path}')
    elif ('module.' not in list(state_dict.keys())[0]) and (opts_dict['train']['is_dist']):  # single-gpu pre-trained -> multi-gpu training
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = 'module.' + k  # add module
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        print(f'loaded from2 {ckp_path}')
    else:  # the same way of training  ,strict=False
        model.load_state_dict(state_dict)
        print(f'loaded from3 {ckp_path}')
    # Tạo thư mục nếu chưa tồn tại
    import os
    os.makedirs("img", exist_ok=True)
    i = 0

    # ==========
    # define loss func & optimizer & scheduler & scheduler & criterion
    # ==========
    assert opts_dict['train']['loss'].pop('type') == 'CharbonnierLoss', \
        "Not implemented."
    loss_func = utils.CharbonnierLoss(**opts_dict['train']['loss'])


    # define optimizer
    assert opts_dict['train']['optim'].pop('type') == 'Adam', \
        "Not implemented."
    optimizer = optim.Adam(
        model.parameters(),
        **opts_dict['train']['optim']
    )

    # define scheduler
    if opts_dict['train']['scheduler']['is_on']:
        assert opts_dict['train']['scheduler'].pop('type') == \
               'CosineAnnealingRestartLR', "Not implemented."
        del opts_dict['train']['scheduler']['is_on']
        scheduler = utils.CosineAnnealingRestartLR(
            optimizer,
            **opts_dict['train']['scheduler']
        )
        opts_dict['train']['scheduler']['is_on'] = True

    # define criterion
    assert opts_dict['train']['criterion'].pop('type') == \
           'PSNR', "Not implemented."
    criterion = utils.PSNR()

    start_iter = 0  # should be restored
    start_epoch = start_iter // num_iter_per_epoch

    # display and log
    if rank == 0:
        msg = (
            f"\n{'<' * 10} Dataloader {'>' * 10}\n"
            f"total iters: [{num_iter}]\n"
            f"total epochs: [{num_epoch}]\n"
            f"iter per epoch: [{num_iter_per_epoch}]\n"
            f"start from iter: [{start_iter}]\n"
            f"start from epoch: [{start_epoch}]"
        )
        print(msg)
        log_fp.write(msg + '\n')
        log_fp.flush()

    if opts_dict['train']['is_dist']:
        torch.distributed.barrier()  # all processes wait for ending

    if rank == 0:
        msg = f"\n{'<' * 10} Training {'>' * 10}"
        print(msg)
        log_fp.write(msg + '\n')

    model.train()
    num_iter_accum = start_iter
    for current_epoch in range(start_epoch, num_epoch + 1):
        if opts_dict['train']['is_dist']:
            train_sampler.set_epoch(current_epoch)

        # fetch the first batch
        tra_prefetcher.reset()
        train_data = tra_prefetcher.next()
        while train_data is not None:
            num_iter_accum += 1
            if num_iter_accum > num_iter:
                break

            # Giả sử: gt_data: [B, C, H, W], lq_data/pd_data: [B, T, C, H, W]
            gt_data = train_data['gt'].to(rank)
            lq_data = train_data['lq'].to(rank)
            pd_data = train_data['pd'].to(rank)
            b, t, c, h, w = lq_data.shape

            # Gộp kênh RGB lại thành chuỗi theo T
            gt1_data = torch.cat([gt_data[:, :, i, ...] for i in range(c)], dim=1)  # (B, T, H, W)
            lq_data = torch.cat([lq_data[:, :, i, ...] for i in range(c)], dim=1)  # (B, T, H, W)
            pd_data = torch.cat([pd_data[:, :, i, ...] for i in range(c)], dim=1)  # (B, T, H, W)

            # # i là chỉ số batch (ví dụ lần load thứ mấy)
            # frame_count = 15   # chỉ lấy 15 frame đầu tiên

            # # Danh sách ảnh PIL
            # lq_imgs = [tensor_to_pil(lq_data[0][j].detach().cpu()) for j in range(frame_count)]
            # pd_imgs = [tensor_to_pil(pd_data[0][j].detach().cpu()) for j in range(frame_count)]
            # gt_imgs = [tensor_to_pil(gt1_data[0][j].detach().cpu()) for j in range(frame_count)]

            # # Kích thước ảnh
            # w, h = lq_imgs[0].size

            # # Tạo ảnh hàng ngang
            # lq_row = Image.new('L', (w * frame_count, h))
            # pd_row = Image.new('L', (w * frame_count, h))
            # gt_row = Image.new('L', (w * frame_count, h))

            # for j in range(frame_count):
            #     lq_row.paste(lq_imgs[j], (j * w, 0))
            #     pd_row.paste(pd_imgs[j], (j * w, 0))
            #     gt_row.paste(gt_imgs[j], (j * w, 0))

            # # Ghép 3 hàng theo chiều dọc
            # combined = Image.new('L', (w * frame_count, h * 3))
            # combined.paste(lq_row, (0, 0))
            # combined.paste(pd_row, (0, h))
            # combined.paste(gt_row, (0, 2 * h))

            # # Lưu ảnh
            # combined.save(f"img/{i}.png")
            # print(f"Đã lưu ảnh {i} gồm 15 frame theo chiều ngang và 3 dòng lq | pd | gt vào 'img/'")

            # i += 1
            # if (i >= 100):
            #     print("Đã lưu đủ 100 ảnh, dừng lại.")
            #     return
            enhanced = model(lq_data, pd_data)

            center_idx = gt_data.shape[1] // 2  # thường là 3 nếu T = 7

            # Lấy frame trung tâm của cả prediction và GT
            # gt_center = gt_data[:, center_idx, ...]         # shape: (B, C, H, W)
            # enhanced_center = enhanced[:, center_idx, ...]  # shape: (B, C, H, W)

            # Tính loss chỉ cho khung hình giữa
            loss = loss_func(enhanced, gt_data)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            # update learning rate
            if opts_dict['train']['scheduler']['is_on']:
                scheduler.step()  # should after optimizer.step()

            if (num_iter_accum % interval_print == 0) and (rank == 0):
                # display & log
                lr = optimizer.param_groups[0]['lr']
                if num_iter_accum == 150000:
                    optimizer.param_groups[0]["lr"] = 5e-5
                elif num_iter_accum == 250000:
                    optimizer.param_groups[0]["lr"] = 2e-5
                elif num_iter_accum == 350000:
                    optimizer.param_groups[0]["lr"] = 1e-5
                loss_item = loss.item()
                msg = (
                    f"iter: [{num_iter_accum}]/{num_iter}, "
                    f"epoch: [{current_epoch}]/{num_epoch - 1}, "
                    "lr: [{:.3f}]x1e-4, loss: [{:.4f}]".format(
                        lr * 1e4, loss_item
                    )
                )
                print(msg)
                log_fp.write(msg + '\n')

            if ((num_iter_accum % interval_val == 0) or \
                (num_iter_accum == num_iter)) and (rank == 0):
                # save model
                checkpoint_save_path = (
                    f"{opts_dict['train']['checkpoint_save_path_pre']}"
                    f"{num_iter_accum}"
                    ".pth"
                )
                state = {
                    'num_iter_accum': num_iter_accum,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                if opts_dict['train']['scheduler']['is_on']:
                    state['scheduler'] = scheduler.state_dict()
                torch.save(state, checkpoint_save_path)

                # log
                msg = (
                    "> model saved at {:s}\n"
                ).format(
                    checkpoint_save_path
                )
                print(msg)
                log_fp.write(msg + '\n')
                log_fp.flush()

            if opts_dict['train']['is_dist']:
                torch.distributed.barrier()  # all processes wait for ending

            # fetch next batch
            train_data = tra_prefetcher.next()

    if rank == 0:
        msg = (
            f"\n{'<' * 10} Goodbye {'>' * 10}\n"
            f"Timestamp: [{utils.get_timestr()}]"
        )
        print(msg)
        log_fp.write(msg + '\n')

        log_fp.close()


if __name__ == '__main__':
    main()
