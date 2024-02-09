import os, argparse
from re import S
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
import pandas as pd 

from model.network import Amodal_imgcrop_simplebevcrop_bid
from model.loss import iou, focal_loss
from dataset.dataloader_movi import get_dataloader
from model.utils import save_checkpoint, make_experiment, load_parameters, generate, generate2

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def test(summary,model,loader,epoch,args,device):

    model.eval()
    iters = 1
    Loss = []
    IOU = torch.tensor(0, device=device, requires_grad=False).float()
    IOU_full = torch.tensor(0, device=device, requires_grad=False).float()
    IOU_occ = torch.tensor(0, device=device, requires_grad=False).float()
    num = torch.tensor(0, device=device, requires_grad=False).float()
    num_occ = torch.tensor(0, device=device, requires_grad=False).float()
    with torch.no_grad():
        for obj_patches, infos in loader:
            obj_patches_crop = infos['obj_patches_crop'].to(device,non_blocking=True)
            vm = infos['vm_crop'].to(device,non_blocking=True)
            calib = infos['calib'].to(device,non_blocking=True)
            fm = infos['fm_crop'].to(device,non_blocking=True)
            pred = model(obj_patches_crop, calib)
            pred_fm = pred['full_mask']

            loss = focal_loss(pred_fm, fm, args.loss_gamma,args.average)
            sum_iou = iou(pred_fm,fm,average=False)
            Loss.append(loss.item())
            IOU+=sum_iou.item()
            num+=vm.shape[1]

            vm_nocrop = infos['vm_nocrop'].to(device,non_blocking=True)
            obj_position = infos['obj_position'].to(device,non_blocking=True)
            fm_nocrop = infos['fm_nocrop'].to(device,non_blocking=True)
            
            full_mask_ori = generate(pred_fm,vm_nocrop,obj_position,logit=True)
            
            iou_full = iou(full_mask_ori,fm_nocrop,average=False)
            iou_occ, num_ = iou(full_mask_ori-vm_nocrop,fm_nocrop-vm_nocrop,average=False,return_num=True)
            IOU_full+=iou_full.item()
            IOU_occ +=iou_occ.item()
            num_occ+=num_.item()

            summary.add_scalar('test/iou_full', IOU_full.item()/num.item(), iters)
            summary.add_scalar('test/iou_occ', IOU_occ.item()/num_occ.item(), iters)
            iters+=1

    dist.all_reduce(IOU)
    dist.all_reduce(IOU_full)
    dist.all_reduce(IOU_occ)
    dist.all_reduce(num)
    dist.all_reduce(num_occ)

    if dist.get_rank() == 0:
        summary.add_text('test_'+epoch+'/result_ddp', 'loss:{} mean_iou:{} iou_full:{} iou_occ:{} param_e:{} fm_num:{}'.format(np.mean(np.array(Loss)),IOU.item()/num.item(),IOU_full.item()/num.item(),IOU_occ.item()/num_occ.item(),epoch, num.item()))

        summary.add_images('test_'+epoch+'/gt_fm', fm[0], 0, dataformats='NCHW')
        summary.add_images('test_'+epoch+'/pred_fm', (pred_fm[0]>0.5).float(), 0, dataformats='NCHW')
        summary.add_images('test_'+epoch+'/gt_fm_nocrop', fm_nocrop[0], 0, dataformats='NCHW')
        summary.add_images('test_'+epoch+'/pred_fm_nocrop', full_mask_ori[0], 0, dataformats='NCHW')
        summary.close()




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--savedir",
        type=str,
        default="/home/ubuntu/amodal_movi_bev/experiment",
        help="directory to save experiments to")
    parser.add_argument(
        "--name", type=str,
        default="debug",
        help="name of experiment")
    

    parser.add_argument('--enlarge_coef', type=float, default=2)
    parser.add_argument('--patch_h_nocrop', type=int, default=256)
    parser.add_argument('--patch_w_nocrop', type=int, default=256)
    parser.add_argument('--patch_h', type=int, default=128)
    parser.add_argument('--patch_w', type=int, default=128)
    parser.add_argument('--fm_h', type=int, default=121)
    parser.add_argument('--fm_w', type=int, default=121)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--f_dim', type=int, default=16)
    parser.add_argument('--num_slot', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=2)

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default=torch.float32)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--seq_len', type=int, default=8)
    parser.add_argument('--save_interval', type=int, default=10)
    parser.add_argument('--average', type=int, default=1)
    parser.add_argument('--loss_gamma', type=float, default=2.)
    parser.add_argument('--part', type=int, default=0)
    parser.add_argument('--decoder', type=str, default='fm', choices=['fm','fm+vm','fm+occ','fm+vm+occ'])
    parser.add_argument('--pos_type', type=str, default='random', choices=['random','camera_position', 'sincos'])

    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--wd', type=float, default=5e-4)
    parser.add_argument('--scheduler', type=str, default='exp', choices=['step','exp'])
    parser.add_argument('--lr-decay', type=int, default=20,
                        help='After how epochs to decay LR by a factor of gamma.')
    parser.add_argument('--gamma', type=float, default=0.95,
                        help='LR decay factor.')

    parser.add_argument('--dataset', type=str, default="kubasic")

    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument('--param', type=str, required=True)

    args = parser.parse_args()
    print(args)

    local_rank = args.local_rank
    
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    if dist.get_rank()==0:
        savedir = os.path.join(args.savedir, args.name)
        summary = SummaryWriter(savedir)
    else:
        summary = None

    if args.dataset.find('kubasic')>-1:
        args.camera_position = [8.48113, -7.50764, 2.5]
    elif args.dataset.find('movid')>-1:
        args.camera_position = [8.48113, -7.50764, 1.5]

    model = Amodal_imgcrop_simplebevcrop_bid(args)
    #model = nn.DataParallel(model.to(device))
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    msg = model.load_state_dict(torch.load(args.param,map_location=device)['model'])
    print(msg)

    test_loader = get_dataloader(args, "test")

    epoch = args.param[-11:-7]


    test(summary,model,test_loader,epoch,args,device)