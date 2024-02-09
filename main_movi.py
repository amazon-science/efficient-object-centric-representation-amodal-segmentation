import os, argparse
from re import S
import numpy as np


import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
import pandas as pd 
import random

from model.network import Amodal_imgcrop_simplebevcrop_bid
from model.loss import iou, focal_loss
from dataset.dataloader_movi import get_dataloader
from model.utils import save_checkpoint, make_experiment, load_parameters, generate, generate2, count

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True

def train(args,summary,model,loader,iters,epoch):
    loader.sampler.set_epoch(epoch)
    for obj_patches, infos in train_loader:
        obj_patches_crop = infos['obj_patches_crop'].to(device,non_blocking=True)
        calib = infos['calib'].to(device,non_blocking=True)
        fm_crop = infos['fm_crop'].to(device,non_blocking=True)
        vm_crop = infos['vm_crop'].to(device,non_blocking=True)
        obj_position = infos['obj_position'].to(device,non_blocking=True)
        pred = model(obj_patches_crop, calib)
        pred_fm = pred['full_mask']
        if args.decoder.find('vm')>-1:
            pred_vm = pred['vis_mask']

        loss_fm = focal_loss(pred_fm, fm_crop,args.loss_gamma,1)
        loss = loss_fm 
        if args.decoder.find('vm')>-1:
            loss_vm = focal_loss(pred_vm, vm_crop,args.loss_gamma,1)
            loss += args.vis_lambda*loss_vm

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if dist.get_rank() == 0:
            summary.add_scalar('train/loss', float(loss.item()), iters)
            summary.add_scalar('train/loss_fm', float(loss_fm.item()), iters)
            if args.decoder.find('vm')>-1:
                summary.add_scalar('train/loss_vm', float(loss_vm.item()), iters)

        iters+=1

    if dist.get_rank() == 0:
        summary.add_images('train/pred_fm', (pred_fm[0]>0.5).float(), epoch, dataformats='NCHW')
        summary.add_images('train/gt_fm', fm_crop[0], epoch, dataformats='NCHW')
        if args.decoder.find('vm')>-1:
            summary.add_images('train/pred_vm', (pred_vm[0]>0.5).float(), epoch, dataformats='NCHW')
            summary.add_images('train/gt_vm', vm_crop[0], epoch, dataformats='NCHW')
    return iters

def validate(args,summary,model,loader,epoch,device):

    model.eval()
    loader.sampler.set_epoch(epoch)
    Loss = []
    IOU = torch.tensor(0, device=device, requires_grad=False).float()
    IOU_occ = torch.tensor(0, device=device, requires_grad=False).float()
    num = torch.tensor(0, device=device, requires_grad=False).float()
    num_occ = torch.tensor(0, device=device, requires_grad=False).float()
    with torch.no_grad():
        for obj_patches, infos in loader:
            obj_patches_crop = infos['obj_patches_crop'].to(device,non_blocking=True)
            vm = infos['vm_nocrop'].to(device,non_blocking=True)
            calib = infos['calib'].to(device,non_blocking=True)
            fm = infos['fm_nocrop'].to(device,non_blocking=True)
            pred = model(obj_patches_crop, calib)
    
            obj_position = infos['obj_position'].to(device,non_blocking=True)
            fm_crop = infos['fm_crop'].to(device,non_blocking=True)
            vm_crop = infos['vm_crop'].to(device,non_blocking=True)
            pred_fm = pred['full_mask']
            if args.decoder.find('vm')>-1:
                pred_vm = pred['vis_mask']
            if args.decoder.find('occ')>-1:
                pred_occ = pred['occ_mask']

            loss = focal_loss(pred_fm, fm_crop,args.loss_gamma,1) 
            if args.decoder.find('vm')>-1:
                loss_vm = focal_loss(pred_vm, vm_crop,args.loss_gamma,1)
                loss += args.vis_lambda*loss_vm
            if args.decoder.find('occ')>-1:
                loss_occ = focal_loss(pred_occ, fm_crop-vm_crop,args.loss_gamma,1)
                loss += args.occ_lambda*loss_occ

            full_mask_ori = generate(pred_fm,vm,obj_position,logit=True)

            sum_iou = iou(full_mask_ori,fm,average=False)
            occ_iou,num_ = iou(full_mask_ori-vm,fm-vm,average=False,return_num=True)
            Loss.append(loss.item())
            IOU+=sum_iou.item()
            num+=vm.shape[1]
            num_occ+=num_.item()
            IOU_occ+=occ_iou.item()

    dist.all_reduce(IOU)
    dist.all_reduce(IOU_occ)
    dist.all_reduce(num)
    dist.all_reduce(num_occ)

    if dist.get_rank() == 0:
        summary.add_scalar('val/loss', np.mean(np.array(Loss)), epoch)
        summary.add_scalar('val/iou_fm', IOU.item()/num.item(), epoch)
        summary.add_scalar('val/iou_occ', IOU_occ.item()/num_occ.item(), epoch)

        summary.add_images('val/pred_fm', (pred_fm[0]>0.5).float(), epoch, dataformats='NCHW')
        summary.add_images('val/gt_fm', fm_crop[0], epoch, dataformats='NCHW')

        if args.decoder.find('vm')>-1:
            summary.add_images('val/pred_vm', (pred_vm[0]>0.5).float(), epoch, dataformats='NCHW')
            summary.add_images('val/gt_vm', vm_crop[0], epoch, dataformats='NCHW')

    model.train()
    return IOU.item()/num.item(), IOU_occ.item()/num_occ.item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--savedir",
        type=str,
        default="/home/ubuntu/amodal_movi_bev/experiment",
        help="directory to save experiments to")
    parser.add_argument(
        "--name", type=str,
        default="tiim_220613",
        help="name of experiment")
    

    parser.add_argument('--enlarge_coef', type=float, default=2)
    parser.add_argument('--patch_h', type=int, default=128)
    parser.add_argument('--patch_w', type=int, default=128)
    parser.add_argument('--fm_h', type=int, default=121)
    parser.add_argument('--fm_w', type=int, default=121)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--f_dim', type=int, default=16)
    parser.add_argument('--num_slot', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=2)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default=torch.float32)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument('--epochs', type=int, default=101)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--seq_len', type=int, default=12)
    parser.add_argument('--save_interval', type=int, default=10)
    parser.add_argument('--average', type=int, default=1)
    parser.add_argument('--loss_gamma', type=float, default=2.)
    parser.add_argument('--vis_lambda', type=float, default=1.0)
    parser.add_argument('--part', type=int, default=0)
    parser.add_argument('--train_len', type=int, default=0)
    parser.add_argument('--decoder', type=str, default='fm', choices=['fm','fm+vm','fm+occ','fm+vm+occ'])
    parser.add_argument('--pos_type', type=str, default='random', choices=['random','camera_position', 'sincos'])

    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--wd', type=float, default=5e-4)
    parser.add_argument('--scheduler', type=str, default='exp', choices=['step','exp'])
    parser.add_argument('--lr-decay', type=int, default=20,
                        help='After how epochs to decay LR by a factor of gamma.')
    parser.add_argument('--gamma', type=float, default=1.00,
                        help='LR decay factor.')
    
    parser.add_argument('--dataset', type=str, default="kubasic")

    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument("--local_rank", default=-1, type=int)

    args = parser.parse_args()
    set_seed(args.seed)

    local_rank = args.local_rank
    
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    if dist.get_rank()==0:
        summary = make_experiment(args)
    else:
        summary = None

    if args.dataset.find('kubasic')>-1:
        args.camera_position = [8.48113, -7.50764, 2.5]
    elif args.dataset.find('movid')>-1:
        args.camera_position = [8.48113, -7.50764, 1.5]

    model = Amodal_imgcrop_simplebevcrop_bid(args)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    if args.scheduler=='step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_decay, gamma=args.gamma)
    elif args.scheduler=='exp':
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)
    epoch_ckpt, iters = load_parameters(args,model,optimizer, scheduler)

    train_loader, val_loader = get_dataloader(args, "train")
    
    if dist.get_rank()==0:
        print('start training from {}-th epoch'.format(epoch_ckpt))

    for epoch in range(epoch_ckpt+1,args.epochs+1):
        
        iters = train(args,summary,model,train_loader,iters,epoch)
        
        scheduler.step()
        
        miou, miou_occ = validate(args,summary,model,val_loader,epoch,device)


        if dist.get_rank()==0:
            print('epoch {}, iou:{}, iou_occ:{}'.format(epoch,miou,miou_occ))

        if epoch % args.save_interval == 0 and dist.get_rank()==0:
            save_checkpoint(args, epoch, model, optimizer, scheduler, iters)
