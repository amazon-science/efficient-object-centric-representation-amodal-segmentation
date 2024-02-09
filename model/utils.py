import torch
import torch.nn as nn
import os
from torch.utils.tensorboard import SummaryWriter
import json
import torch.nn.functional as F

def sin_cos_pos(seq_len,num_hiddens):
    P = torch.zeros((1, seq_len, num_hiddens))
    X = torch.arange(seq_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
    P[:, :, 0::2] = torch.sin(X)
    P[:, :, 1::2] = torch.cos(X)
    return P

class pos_mlp(nn.Module):
    def __init__(self,args,obj='slot'):
        super(pos_mlp, self).__init__()
        layers = []
        if obj=='slot':
            self.num = args.num_slot
        elif obj=='feature':
            self.num = args.f_dim**2
            
        for _ in self.num:
            layer = nn.Sequential(
                            nn.Linear(3,args.d_model//2),
                            nn.LeakyReLU(),
                            nn.Linear(args.d_model//2,args.d_model),
                            nn.LeakyReLU()
                            )
            layers.append(layer)
        self.layers = nn.ModuleList(layers)
        
        self.d_model = args.d_model

    def forward(self,x):
        y = []
        for l in self.layers:
            y.append(l(x))
        y = torch.cat(y,dim=1)
        return y.view(-1, self.num, self.d_model)


def generate(fm_pred,vm_nocrop,obj_position,logit=True):
    #device = vm_nocrop.device
    #fm_final = torch.zeros(vm_nocrop.shape).to(device,non_blocking=True)
    num = []
    fm_final = []
    _, t, _, H, W = vm_nocrop.shape
    for i in range(t):
        obj = obj_position[0][i]
        h = int((obj[1]-obj[0]+1).item())
        w = int((obj[3]-obj[2]+1).item())

        pred = F.interpolate(fm_pred[0][i:i+1],size=(h,w),mode='bilinear', align_corners=True)
        if logit:
            pred = (pred>0.5).float()
        pred = F.pad(pred, (int(obj[2].item()),W-int(obj[3].item())-1,int(obj[0].item()),H-int(obj[1].item())-1), 'constant', 0)
        fm_final.append(pred)
        num.append(h*w)
    fm_final = torch.cat(fm_final).unsqueeze(0)
    if logit:
        return fm_final
    else:
        return fm_final, num

def generate2(fm_pred,vm_nocrop,loss_mask,obj_position,logit=True):
    #device = vm_nocrop.device
    #fm_final = torch.zeros(vm_nocrop.shape).to(device,non_blocking=True)
    num = []
    fm_final = []
    _, t, _, H, W = vm_nocrop.shape
    for i in range(t):
        obj = obj_position[0][i]
        h = int((obj[1]-obj[0]+1).item())
        w = int((obj[3]-obj[2]+1).item())

        pred = F.interpolate(fm_pred[0][i:i+1],size=(h,w),mode='bilinear', align_corners=True)
        if logit:
            pred = (pred>0.5).float()
        pred = F.pad(pred, (int(obj[2].item()),W-int(obj[3].item())-1,int(obj[0].item()),H-int(obj[1].item())-1), 'constant', 0)
        fm_final.append(pred)
        num.append(h*w)
    fm_final = torch.cat(fm_final).unsqueeze(0)
    if logit:
        fm_final = vm_nocrop * loss_mask + (1-loss_mask) * fm_final
        return fm_final
    else:
        return fm_final, num

def count(obj_position):
    num = []
    _, t, _ = obj_position.shape
    for i in range(t):
        obj = obj_position[0][i]
        h = int((obj[1]-obj[0]+1).item())
        w = int((obj[3]-obj[2]+1).item())
        num.append(h*w)
    return num


def save_checkpoint(args, epoch, model, optimizer, scheduler, iters):
    ckpt = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optim": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "iters":iters
    }
    ckpt_file = os.path.join(
        args.savedir, args.name, "checkpoint-{:04d}.pth.gz".format(epoch)
    )
    print("==> Saving checkpoint '{}'".format(ckpt_file))
    torch.save(ckpt, ckpt_file)

def make_experiment(args):
    print("\n" + "#" * 80)
    
    print(
        "Creating experiment '{}' in directory:\n  {}".format(args.name, args.savedir)
    )
    print("#" * 80)
    print("\nConfig:")
    for key in sorted(args.__dict__):
        print("  {:12s} {}".format(key + ":", args.__dict__[key]))
    print("#" * 80)

    # Create a new directory for the experiment
    savedir = os.path.join(args.savedir, args.name)
    os.makedirs(savedir, exist_ok=True)

    # # Create tensorboard summary writer
    summary = SummaryWriter(savedir)

    # # Save configuration to file
    k_v = {}
    for k in list(vars(args).keys()):
        if k!='dtype':
            k_v[k]=vars(args)[k]

    with open(os.path.join(savedir, "config.txt"), "w") as fp:
        json.dump(k_v, fp)

    return summary

def load_parameters(args, model, optimizer, scheduler):
    model_dir = os.path.join(args.savedir, args.name)
    checkpt_fn = sorted(
        [
            f
            for f in os.listdir(model_dir)
            if os.path.isfile(os.path.join(model_dir, f)) and ".pth.gz" in f
        ]
    )
    if len(checkpt_fn) != 0:
        model_pth = os.path.join(model_dir, checkpt_fn[-1])
        ckpt = torch.load(model_pth, map_location=torch.device('cpu'))
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optim"])
        scheduler.load_state_dict(ckpt["scheduler"])
        epoch_ckpt = ckpt["epoch"]
        iters = ckpt["iters"]
        print("starting training from {}".format(checkpt_fn[-1]))
    else:
        epoch_ckpt = 0
        iters = 1
        pass

    return epoch_ckpt, iters