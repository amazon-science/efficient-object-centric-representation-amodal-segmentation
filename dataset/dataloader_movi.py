from pickle import OBJ
import numpy as np
import torch
from torch import distributed
from torch.utils import data
from tqdm import utils
from .dataset_movi import Movi
from torch.utils.data import DataLoader

def Movi_collate_fn(batch):
    keys_infos = batch[0].keys()
    new_batch_infos = {}

    for k in keys_infos:
        tmp = [val[k] for val in batch]
        new_batch_infos[k] = torch.stack(tmp,dim=0)

    return new_batch_infos["input_obj_patches"], new_batch_infos


def get_dataloader(args, mode):
    train_dir = "/home/ubuntu/"+args.dataset+"/train"
    test_dir = "/home/ubuntu/"+args.dataset+"/test"
    if mode == "train":
        
        train_set = Movi(data_dir=train_dir, args=args, mode="train")
        val_set = Movi(data_dir=test_dir, args=args, mode="test")
        
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
        train_loader = DataLoader(train_set, args.batch_size, shuffle=False, 
                                  sampler=train_sampler, collate_fn=Movi_collate_fn,
                                  num_workers=args.num_workers)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_set)
        val_loader = DataLoader(val_set, args.batch_size, shuffle=False, 
                                sampler=val_sampler, collate_fn=Movi_collate_fn,
                                num_workers=args.num_workers)
        
        return train_loader, val_loader
    elif mode == "test":
        
        test_set = Movi(data_dir=test_dir, args=args, mode="test")
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_set)
        test_loader = DataLoader(test_set, args.batch_size, shuffle=False, sampler=test_sampler, collate_fn=Movi_collate_fn)
        return test_loader
    else:
        raise
