import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import math
import time
import yaml
import argparse
import torch
import torch.optim as optim
try:
    from torch.amp import autocast, GradScaler
except:
    from torch.cuda.amp import autocast, GradScaler

import numpy as np
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
from collections import OrderedDict
import utils  # my tool box
import dataset
from models import *
import os.path as op
from PIL import Image
import numpy as np
from comet_ml import Experiment, ExistingExperiment

def skip_batches(tra_prefetcher, start_iters):
    for _ in tqdm(range(start_iters), unit='batch'):
        try:
            _ = tra_prefetcher.next()
        except StopIteration:
            break
    return tra_prefetcher.next()

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
        '--opt_path', type=str, default='configs/train_TGAF.yml',
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

    opts_dict['train']['log_path'] = os.path.join(
        "exp", opts_dict['train']['exp_name'], "log.log"
    )
    opts_dict['train']['checkpoint_save_path_pre'] = os.path.join(
        "exp", opts_dict['train']['exp_name'], "ckp_"
    )
    opts_dict['train']['best_model'] = os.path.join("exp", opts_dict['train']['exp_name'], 'best_weight.pth')
    opts_dict['train']['best_psnr_model'] = os.path.join("exp", opts_dict['train']['exp_name'], 'best_psnr_weight.pth')
    opts_dict['train']['best_ssim_model'] = os.path.join("exp", opts_dict['train']['exp_name'], 'best_ssim_weight.pth')


    opts_dict['train']['num_gpu'] = torch.cuda.device_count()
    # opts_dict['train']['num_gpu'] = 1

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
    
    #config dataset
    QP = opts_dict['QP']
    # enc_type = opts_dict['dataset']['enc_type']
    opts_dict['dataset']['train']['gt_path'] = os.path.join(opts_dict['dataset']['train']['gt_path'])
    opts_dict['dataset']['train']['lq_path'] = os.path.join(f'QP{QP}', opts_dict['dataset']['train']['lq_path'])
    opts_dict['dataset']['train']['pd_path'] = os.path.join(f'QP{QP}', opts_dict['dataset']['train']['pd_path'])
    
    opts_dict['dataset']['val']['gt_path'] = os.path.join(opts_dict['dataset']['val']['root'], 
                                                          opts_dict['dataset']['val']['gt_path'])
    opts_dict['dataset']['val']['lq_path'] = os.path.join(opts_dict['dataset']['val']['root'], f'QP{QP}',
                                                          opts_dict['dataset']['val']['lq_path'])
    opts_dict['dataset']['val']['pd_path'] = os.path.join(opts_dict['dataset']['val']['root'], f'QP{QP}',
                                                          opts_dict['dataset']['val']['pd_path'])
    # ==========
    # init distributed training
    # ==========
    if opts_dict['train']['is_dist']:
        utils.init_dist(
            local_rank=rank,
            backend='nccl'
        )
    pass
    using_comet = opts_dict['comet_logging'].pop('using')
    previous_experiment = opts_dict['comet_logging'].pop('previous_experiment')

    if using_comet:
        if previous_experiment:
            experiment = ExistingExperiment(previous_experiment=previous_experiment, **opts_dict['comet_logging'])    
        else:
            experiment = Experiment(**opts_dict['comet_logging']) 

        experiment.set_name(opts_dict['train']['exp_name'])

    if rank == 0:
        log_dir = os.path.join("exp", opts_dict['train']['exp_name'])
        if not os.path.exists(log_dir):
            print("log_dir", log_dir)
            utils.mkdir(log_dir)        
        log_fp = open(opts_dict['train']['log_path'], 'a')

        # log all parameters
        msg = (
            f"{'<' * 10} Hello {'>' * 10}\n"
            f"Timestamp: [{utils.get_timestr()}]\n"
            f"\n{'<' * 10} Options {'>' * 10}\n"
            f"{utils.dict2str(opts_dict)}"
        )
        print(msg)
        if not previous_experiment:
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


    # ==========
    # create model    ,find_unused_parameters=True
    # ==========
    # model = STFF_L()
    model = TGAF(opts_dict['network'])
    # model = TVQE(opts_dict=opts_dict['network'])
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

    ckp_path = opts_dict['train']['load_best_weight']
    if os.path.exists(ckp_path):
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
    # import os
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
    
    #Mixed Precision training
    amp = opts_dict['AMP']
    if amp:
        scaler = GradScaler()
        
    start_iter = 0  # should be restored
    best_psnr = -float('inf')
    best_ssim = -float('inf')
    # load checkpoint
    if os.path.isfile(opts_dict['train']['checkpoint']):
        checkpoint = torch.load(opts_dict['train']['checkpoint'], map_location="cpu")
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_psnr = checkpoint['best_psnr']
        if 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
        if 'num_iter_accum' in checkpoint:
            start_iter = checkpoint['num_iter_accum']
            print(f"Resume from iteration: {start_iter}")
        if 'best_ssim' in checkpoint:
            best_ssim = checkpoint['best_ssim']
        if 'amp' in checkpoint:
            scaler.load_state_dict(checkpoint['amp'])
    
    # create datasets
    train_ds_type = opts_dict['dataset']['train']['type']
    radius = opts_dict['network']['radius']
    assert train_ds_type in dataset.__all__, \
        "Not implemented!"
    train_ds_cls = getattr(dataset, train_ds_type)
    train_ds = train_ds_cls(opts_dict=opts_dict['dataset']['train'], radius=radius)
    print(len(train_ds))
    # create datasamplers
    train_sampler = utils.ResumeDistSampler(
        dataset=train_ds,
        num_replicas=opts_dict['train']['num_gpu'],
        start_iter=start_iter,
        rank=rank,
        ratio=opts_dict['dataset']['train']['enlarge_ratio']
    )
    # train_sampler = utils.DistSampler(
    #     dataset=train_ds,
    #     num_replicas=opts_dict['train']['num_gpu'],
    #     rank=rank,
    #     ratio=opts_dict['dataset']['train']['enlarge_ratio']
    # )
    # create dataloaders
    train_loader = utils.create_dataloader(
        dataset=train_ds,
        opts_dict=opts_dict,
        sampler=train_sampler,
        phase='train',
        seed=opts_dict['train']['random_seed']
    )
    val_gt, val_lr, val_pd = opts_dict['dataset']['val']['gt_path'], opts_dict['dataset']['val']['lq_path'], opts_dict['dataset']['val']['pd_path']
    assert train_loader is not None

    batch_size = opts_dict['dataset']['train']['batch_size_per_gpu'] * \
                 opts_dict['train']['num_gpu']  # divided by all GPUs
    num_iter_per_epoch = math.ceil(len(train_ds) * \
                                   opts_dict['dataset']['train']['enlarge_ratio'] / batch_size)
    num_epoch = math.ceil(num_iter / num_iter_per_epoch)
    start_epoch = start_iter // num_iter_per_epoch
    # start_epoch = 0
    # create dataloader prefetchers
    tra_prefetcher = utils.CPUPrefetcher(train_loader)

    
        
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

    avg_psnr, avg_ssim = utils.validate_tgaf(model, val_gt, val_lr, val_pd)
    msg = (
            f"Validating: delta PSNR: [{avg_psnr:.4f}], delta SSIM: [{avg_ssim * 100:.4f}], "
        )
    if avg_psnr > best_psnr and avg_ssim > best_ssim:
        best_psnr = avg_psnr
        best_ssim = avg_ssim
        msg += f'\nSaved best model with best Delta PSNR = {avg_psnr:.4f} and Delta SSIM = {avg_ssim*100:.4f}\n'
        state = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        if opts_dict['train']['scheduler']['is_on']:
            state['scheduler'] = scheduler.state_dict()
        torch.save(state, opts_dict['train']['best_model'])
    print(msg)
    log_fp.write(msg + '\n')
    if rank == 0:
        msg = f"\n{'<' * 10} Training {'>' * 10}"
        print(msg)
        log_fp.write(msg + '\n')
        total_train_timer = utils.system.Timer()  # total time of each epoch

    training_timer = utils.system.Timer()
    
    model.train()
    num_iter_accum = start_iter
    for current_epoch in range(start_epoch, num_epoch + 1):
        if opts_dict['train']['is_dist']:
            train_sampler.set_epoch(current_epoch)

        # fetch the first batch
        tra_prefetcher.reset()
        train_data = tra_prefetcher.next()
        # start = time.time()
        # train_data = skip_batches(tra_prefetcher, start_iter)
        # end = time.time()

        while train_data is not None:
            num_iter_accum += 1
            if num_iter_accum > num_iter:
                break

            # Giả sử: gt_data: [B, C, H, W], lq_data/pd_data: [B, T, C, H, W]
            gt_data = train_data['gt'].to(rank)
            lq_data = train_data['lq'].to(rank)
            b, t, c, h, w = lq_data.shape

            # Gộp kênh RGB lại thành chuỗi theo T
            gt1_data = torch.cat([gt_data[:, :, i, ...] for i in range(c)], dim=1)  # (B, T, H, W)
            lq_data = torch.cat([lq_data[:, :, i, ...] for i in range(c)], dim=1)  # (B, T, H, W)
            optimizer.zero_grad()

            if amp:
                with autocast():
                    enhanced = model(lq_data)
                    # print(enhanced.shape, gt_data.shape)
                    loss = loss_func(enhanced, gt_data)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                enhanced = model(lq_data)
                # print(enhanced.shape, gt_data.shape)
                loss = loss_func(enhanced, gt_data)
                
                loss.backward()
                optimizer.step()
            
            if using_comet:
                experiment.log_metric("train_loss", loss.item(), step=num_iter_accum)
                # experiment.log_metric("train_psnr", psnr, step=train_step)

            # update learning rate
            if opts_dict['train']['scheduler']['is_on']:
                scheduler.step()  # should after optimizer.step()

            if ((num_iter_accum % interval_val == 0) or \
                (num_iter_accum == num_iter)) and (rank == 0):
                # save model
                iteration_time = training_timer.get_interval()

                lr = optimizer.param_groups[0]['lr']
                if num_iter_accum == 150000:
                    optimizer.param_groups[0]["lr"] = 5e-5
                elif num_iter_accum == 250000:
                    optimizer.param_groups[0]["lr"] = 2e-5
                elif num_iter_accum == 350000:
                    optimizer.param_groups[0]["lr"] = 1e-5
                loss_item = loss.item()
                avg_psnr, avg_ssim = utils.validate_tgaf(model, val_gt, val_lr, val_pd)
                if using_comet:
                    experiment.log_metric("delta PSNR", avg_psnr, step=num_iter_accum//interval_val)
                    experiment.log_metric("delta SSIM", avg_ssim, step=num_iter_accum//interval_val)

                msg = (
                    f"iter: [{num_iter_accum}]/{num_iter}, "
                    f"epoch: [{current_epoch}]/{num_epoch - 1}, "
                    f"lr: [{lr * 1e4:.3f}]x1e-4, loss: [{loss_item:.4f}], "
                    f"delta PSNR: [{avg_psnr:.4f}], delta SSIM: [{avg_ssim * 100:.4f}], "
                    f"iteration time: [{iteration_time:.4f}] s\n"
                )
                # print('validating')
                # print(avg_psnr)
                if avg_psnr > best_psnr and avg_ssim > best_ssim:
                    best_psnr = avg_psnr
                    best_ssim = avg_ssim
                    msg += f'\nSaved best model with best Delta PSNR = {avg_psnr:.4f} and Delta SSIM = {avg_ssim*100:.4f}\n'
                    state = {
                        'num_iter_accum': num_iter_accum,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }
                    if opts_dict['train']['scheduler']['is_on']:
                        state['scheduler'] = scheduler.state_dict()
                    torch.save(state, opts_dict['train']['best_model'])
                elif avg_psnr > best_psnr:
                    best_psnr = avg_psnr
                    msg += f'\nSaved best model with best Delta PSNR = {avg_psnr:.4f}\n'
                    state = {
                        'num_iter_accum': num_iter_accum,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }
                    if opts_dict['train']['scheduler']['is_on']:
                        state['scheduler'] = scheduler.state_dict()
                    torch.save(state, opts_dict['train']['best_psnr_model'])
                elif avg_ssim > best_ssim:
                    best_ssim = avg_ssim
                    msg += f'\nSaved best model with best Delta SSIM = {avg_ssim*100:.4f}\n'
                    state = {
                        'num_iter_accum': num_iter_accum,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }
                    if opts_dict['train']['scheduler']['is_on']:
                        state['scheduler'] = scheduler.state_dict()
                    torch.save(state, opts_dict['train']['best_ssim_model']) 
                training_timer.restart()
                checkpoint_save_path = (
                    f"{opts_dict['train']['checkpoint_save_path_pre']}"
                    f"{num_iter_accum}"
                    ".pth"
                )
                state = {
                    'num_iter_accum': num_iter_accum,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_psnr': best_psnr,
                    'best_ssim': best_ssim
                }
                if opts_dict['train']['scheduler']['is_on']:
                    state['scheduler'] = scheduler.state_dict()
                if amp:
                    state['amp'] = scaler.state_dict()
                torch.save(state, checkpoint_save_path)

                # log
                msg += "> model saved at {:s}\n".format(checkpoint_save_path)
                
                print(msg)
                log_fp.write(msg + '\n')
                log_fp.flush()

            if opts_dict['train']['is_dist']:
                torch.distributed.barrier()  # all processes wait for ending

            # fetch next batch
            train_data = tra_prefetcher.next()
            
    experiment.end()
    if rank == 0:
        total_time = total_train_timer.get_interval() / 3600
        total_day = total_train_timer.get_interval() / (24 * 3600)

        msg_hours = "TOTAL TIME: [{:.4f}] h".format(total_time)
        msg_days = "TOTAL TIME: [{:.4f}] days".format(total_day)

        print(msg_hours)
        print(msg_days)
        log_fp.write(msg_hours + '\n')
        log_fp.write(msg_days + '\n')

        goodbye_msg = (f"\n{'<' * 10} Goodbye {'>' * 10}\n"
                       f"Timestamp: [{utils.get_timestr()}]")
        print(goodbye_msg)
        log_fp.write(goodbye_msg + '\n')

        log_fp.close()


if __name__ == '__main__':
    main()
