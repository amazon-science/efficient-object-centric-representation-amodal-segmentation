import json, os, pickle
from unicodedata import lookup

import numpy as np
import torch

import matplotlib.pyplot as plt
from skimage import transform
from pycocotools import mask as coco_mask
import copy

import sys
sys.path.append('..')
from model.calibration import Calibration


class Kitti(object):
    def __init__(self, data_dir, args, mode, subtest=None):

        self.mode = mode
        self.device = "cpu"
        self.dtype = args.dtype
        self.part = args.part

        self.data_dir = data_dir
          
        self.data_summary = pickle.load(open(os.path.join(self.data_dir, self.mode+"_data.pkl"), "rb"))
        self.full_summary = pickle.load(open(os.path.join(self.data_dir, self.mode+"_data_full_mask.pkl"), "rb"))
        self.kitti_file = pickle.load(open(os.path.join('/home/ubuntu/data/KINS_Video_Car/car_data', self.mode+"_Kins_Kitti_file_matching.pkl"), "rb"))

        self.video_obj_lists = []
        for video_name in self.data_summary.keys():
            for obj_id in self.data_summary[video_name].keys():
                self.video_obj_lists.append(video_name + "-" + str(obj_id))

        self.img_path = "/home/ubuntu/data/KINS_Video_Car/Kitti/raw_video"
        self.calib_dir = '/home/ubuntu/data/KINS_Video_Car/Kitti/calib/'

        c = [3,26,28,29,30]
        self.calib_id = {}
        for i in c:
            calib = Calibration(self.calib_dir+str(i)+'.txt')
            self.calib_id[i] = torch.tensor([[calib.fu,0,calib.cu],[0,calib.fv,calib.cv],[0,0,1]])
        self.flow_path = "/home/ubuntu/data/KINS_Video_Car/Kitti/Kitti_flow"
        self.flow_reverse_path =  "/home/ubuntu/data/KINS_Video_Car/Kitti/Kitti_reverse_flow"

        self.fm_h = args.fm_h
        self.fm_w = args.fm_w

        self.patch_h = args.patch_h
        self.patch_w = args.patch_w
        self.enlarge_coef = args.enlarge_coef
        self.train_seq_len = args.seq_len

        self.cur_video_name = None
        self.video_frames = None
        self.video_name_id_map = dict(zip(list(self.data_summary.keys()), range(len(self.data_summary))))
    
    def __len__(self):
        return len(self.video_obj_lists)
            
    def __getitem__(self, idx, specified_V_O_id=None):
        if specified_V_O_id is None:
            video_name, obj_id = self.video_obj_lists[idx].split("-")
        else:
            video_name, obj_id = specified_V_O_id.split("-")
        if video_name != self.cur_video_name:
            self.video_frames, self.calib_path, self.check_fm= self.getImg(video_name) 
            self.cur_video_name = video_name
        video_frames = self.video_frames

        obj_patches = []
        obj_patches_crop = []
        obj_position = []
        obj_rate = []
        counts = []
        imgs = []
        calib_paths = []
        vm_crop = []
        fm_crop = []
        vm_nocrop = []
        fm_nocrop = []
        loss_mask_weight = []
        loss_mask_weight_crop = []
        fm_avail = []

        # for evaluation 
        video_ids = []
        object_ids = []
        frame_ids = []
        obj_dict = self.data_summary[video_name][int(obj_id)]
        full_dict = self.full_summary[video_name][int(obj_id)]
        timesteps = list(obj_dict.keys())
        assert np.all(np.diff(sorted(timesteps))==1)
        start_t, end_t = min(timesteps), max(timesteps)

        if (self.mode != "test" or self.part) and end_t - start_t > self.train_seq_len - 1:
            start_t = np.random.randint(start_t, end_t-(self.train_seq_len-2))
            end_t = start_t + self.train_seq_len - 1
        
        for t_step in range(start_t, end_t+1):
            Image_H, Image_W = obj_dict[t_step]["VM"]["size"]
            vm = coco_mask.decode(obj_dict[t_step]["VM"]).astype(bool)
            fm = coco_mask.decode(full_dict[t_step]["FM"]).astype(bool)
            vx_min, vx_max, vy_min, vy_max = obj_dict[t_step]["VM_bbox"]
            # enlarge the bbox
            x_center = (vx_min + vx_max) // 2
            y_center = (vy_min + vy_max) // 2
            x_len = int((vx_max - vx_min) * self.enlarge_coef)
            y_len = int((vy_max - vy_min) * self.enlarge_coef)
            vx_min = max(0, x_center - x_len // 2)
            vx_max = min(Image_H, x_center + x_len // 2)
            vy_min = max(0, y_center - y_len // 2)
            vy_max = min(Image_W, y_center + y_len // 2)
            obj_position.append([vx_min, vx_max, vy_min, vy_max])
            obj_rate.append([vx_min/Image_H, vx_max/Image_H, vy_min/Image_W, vy_max/Image_W])
            vm_crop.append(vm[vx_min:vx_max+1, vy_min:vy_max+1])
            vm_nocrop.append(vm)
            fm_crop.append(fm[vx_min:vx_max+1, vy_min:vy_max+1])
            fm_nocrop.append(fm)
            fm_avail.append(float((fm[vx_min:vx_max+1, vy_min:vy_max+1]).sum()>0))
            # get loss mask
            loss_mask = (1 - coco_mask.decode(obj_dict[t_step]["loss_mask"])).astype(bool)
            loss_mask_weight.append(loss_mask)
            loss_mask_weight_crop.append(loss_mask[vx_min:vx_max+1, vy_min:vy_max+1])

            # get patches and flow crop
            img = video_frames[t_step]
            patch_crop = video_frames[t_step][vx_min:vx_max+1, vy_min:vy_max+1]
            patch = video_frames[t_step]
            calib = self.calib_id[self.calib_path[t_step]]

            obj_patches.append(patch)
            obj_patches_crop.append(patch_crop)

            calib_paths.append(calib)
            imgs.append(img)
            # for evaluation
            video_ids.append(self.video_name_id_map[video_name])
            object_ids.append(int(obj_id))
            frame_ids.append(t_step)
            counts.append(1)
        
        num_pad = max(self.train_seq_len - (end_t - start_t + 1), 0)
        for _  in range(num_pad):
            obj_position.append(copy.deepcopy(obj_position[-1]))
            obj_rate.append(copy.deepcopy(obj_rate[-1]))
            vm_crop.append(copy.deepcopy(vm_crop[-1]))
            fm_crop.append(copy.deepcopy(fm_crop[-1]))
            vm_nocrop.append(copy.deepcopy(vm_nocrop[-1]))
            fm_nocrop.append(copy.deepcopy(fm_nocrop[-1]))
            imgs.append(copy.deepcopy(imgs[-1]))
            calib_paths.append(copy.deepcopy(calib_paths[-1]))

            loss_mask_weight.append(copy.deepcopy(loss_mask_weight[-1]))
            loss_mask_weight_crop.append(copy.deepcopy(loss_mask_weight_crop[-1]))
            obj_patches.append(copy.deepcopy(obj_patches[-1]))
            obj_patches_crop.append(copy.deepcopy(obj_patches_crop[-1]))
            fm_avail.append(copy.deepcopy(fm_avail[-1]))

            video_ids.append(video_ids[-1])
            object_ids.append(object_ids[-1])
            frame_ids.append(frame_ids[-1] + 1)
            counts.append(0)
        
        obj_rate = torch.from_numpy(np.array(obj_rate)).to(self.dtype).to(self.device)
        obj_position = torch.from_numpy(np.array(obj_position)).to(self.dtype).to(self.device)
        counts = torch.from_numpy(np.array(counts)).to(self.dtype).to(self.device)
        loss_mask_weight = torch.from_numpy(np.array(loss_mask_weight)).to(self.dtype).to(self.device).unsqueeze(1)
        calib_paths = torch.stack(calib_paths).to(self.dtype).to(self.device)
        fm_avail = torch.from_numpy(np.array(fm_avail)).to(self.dtype).to(self.device)
        obj_patches_crop = self.rescale_patch(obj_patches_crop)

        fm_crop = self.fm_rescale(fm_crop)
        vm_crop = self.fm_rescale(vm_crop)
        loss_mask_weight_crop = self.fm_rescale(loss_mask_weight_crop)
        loss_mask_weight_crop = torch.from_numpy(np.array(loss_mask_weight_crop)).to(self.dtype).to(self.device).unsqueeze(1)
        obj_temp = np.stack(obj_patches, axis=0) # Seq_len * patch_h * patch_w * 3
        obj_temp_crop = np.stack(obj_patches_crop, axis=0)

        obj_patches = torch.from_numpy(obj_temp).to(self.dtype).to(self.device).permute(0,3,1,2)
        obj_patches_crop = torch.from_numpy(obj_temp_crop).to(self.dtype).to(self.device).permute(0,3,1,2)
        vm_crop = torch.from_numpy(np.array(vm_crop)).to(self.dtype).to(self.device).unsqueeze(1)
        fm_crop = torch.from_numpy(np.array(fm_crop)).to(self.dtype).to(self.device).unsqueeze(1)
        vm_nocrop = torch.from_numpy(np.array(vm_nocrop)).to(self.dtype).to(self.device).unsqueeze(1)
        fm_nocrop = torch.from_numpy(np.array(fm_nocrop)).to(self.dtype).to(self.device).unsqueeze(1)
        imgs = torch.from_numpy(np.array(imgs)).to(self.dtype).to(self.device).permute(0,3,1,2)

        video_ids = torch.from_numpy(np.array(video_ids)).to(self.dtype).to(self.device)
        object_ids = torch.from_numpy(np.array(object_ids)).to(self.dtype).to(self.device)
        frame_ids = torch.from_numpy(np.array(frame_ids)).to(self.dtype).to(self.device)

        obj_data = {"input_obj_patches": obj_patches, 
                    "obj_patches_crop": obj_patches_crop,
                    "raw_imgs": imgs,
                    "vm_crop" : vm_crop,  
                    "fm_avail" : fm_avail,
                    "calib": calib_paths,
                    "fm_crop": fm_crop,
                    "obj_position": obj_position, 
                    "obj_rate": obj_rate, 
                    "loss_mask": loss_mask_weight, 
                    "loss_mask_crop": loss_mask_weight_crop,
                    "vm_nocrop": vm_nocrop,
                    "fm_nocrop": fm_nocrop,
                    "counts": counts,
                    "video_ids": video_ids,
                    "object_ids": object_ids,
                    "frame_ids": frame_ids,
                    }

        return obj_data


    def get_image_instances(self, video_name):
        objects_data = []
        for obj_id in self.data_summary[video_name].keys():
            specified_V_O_id = video_name + "-" + str(obj_id)
            obj_data = self.__getitem__(idx=None, specified_V_O_id=specified_V_O_id)
            timesteps = self.data_summary[video_name][obj_id].keys()
            obj_data["st"] = min(timesteps)
            obj_data["et"] = max(timesteps)
            obj_data["obj_id"] = obj_id
            objects_data.append(obj_data)
        return objects_data


        
    def fm_rescale(self, masks):     
        
        for i, m in enumerate(masks):
            if m is None:
                continue
            h, w = masks[i].shape[:2]
            m = transform.rescale(m, (self.fm_h/h, self.fm_w/w))
            cur_h, cur_w = m.shape[:2]
            to_pad = ((0, max(self.fm_h-cur_h, 0)), (0, max(self.fm_w-cur_w, 0)))
            m = np.pad(m, to_pad)[:self.fm_h,:self.fm_w]
            masks[i] = m
        return masks


    def rescale(self, masks, obj_patches=None):
        mask_count = [np.sum(m)  if m is not None else 0 for m in masks]
        idx = np.argmax(mask_count)

        h, w = masks[idx].shape[:2]
        for i, m in enumerate(masks):
            if m is None:
                continue
            m = transform.rescale(m, (self.patch_h/h, self.patch_w/w))
            cur_h, cur_w = m.shape[:2]
            to_pad = ((0, max(self.patch_h-cur_h, 0)), (0, max(self.patch_w-cur_w, 0)))
            m = np.pad(m, to_pad)[:self.patch_h,:self.patch_w]
            masks[i] = m

        for i, obj_pt in enumerate(obj_patches):
            obj_pt = transform.rescale(obj_pt, (self.patch_h/h, self.patch_w/w, 1))
            cur_h, cur_w = obj_pt.shape[:2]
            to_pad = ((0, max(self.patch_h-cur_h, 0)), (0, max(self.patch_w-cur_w, 0)), (0, 0))
            obj_pt = np.pad(obj_pt, to_pad)[:self.patch_h,:self.patch_w,:3]
            obj_patches[i] = obj_pt
              

        return masks, obj_patches

    def rescale_patch(self, obj_patches=None):
        for i, obj_pt in enumerate(obj_patches):
            h,w = obj_pt.shape[:2]
            obj_pt = transform.rescale(obj_pt, (self.patch_h/h, self.patch_w/w, 1))
            cur_h, cur_w = obj_pt.shape[:2]
            to_pad = ((0, max(self.patch_h-cur_h, 0)), (0, max(self.patch_w-cur_w, 0)), (0, 0))
            obj_pt = np.pad(obj_pt, to_pad)[:self.patch_h,:self.patch_w,:3]
            obj_patches[i] = obj_pt
        return obj_patches
    
    def getImg(self, v_id):
        imgs = []
        paths = []
        check_fm = []
        cur_img_path = os.path.join(self.img_path, "_".join(v_id.split("_")[:3]), v_id)
        imgs_list = os.listdir(cur_img_path)
        imgs_list = [val for val in imgs_list if val.find(".png") > -1]
        imgs_list = sorted(imgs_list, key=lambda x: int(x[:-4]))
        for sub_path in imgs_list:
            if sub_path.find("png") > -1:
                img_path = os.path.join(cur_img_path, sub_path)
                paths.append(int(v_id[:10][-2:]))
                check_fm.append(img_path)
                img_tmp = plt.imread(img_path)
                imgs.append(img_tmp)
        return imgs, paths, check_fm

    def getFlow(self, flow_path, v_id, shift):
        flows = []
        cur_flow_path = os.path.join(flow_path, "_".join(v_id.split("_")[:3]), v_id)
        flows_list = os.listdir(cur_flow_path)
        flows_list = sorted(flows_list, key=lambda x: int(x[:-4]))
        for sub_path in flows_list:
            if sub_path.find("flo") > -1:
                sub_flow_path = os.path.join(cur_flow_path, sub_path)
                f_tmp = self.readFlow(sub_flow_path) 
                flows.append(f_tmp)
        if shift > 0:
            flows = flows + flows[-shift:]
        elif shift < 0:
            flows = flows[:-shift] + flows
        return flows
    

    def readFlow(self, fn):
        with open(fn, 'rb') as f:
            magic = np.fromfile(f, np.float32, count=1)
            if 202021.25 != magic:
                print('Magic number incorrect. Invalid .flo file')
                return None
            else:
                w = np.fromfile(f, np.int32, count=1)
                h = np.fromfile(f, np.int32, count=1)
                data = np.fromfile(f, np.float32, count=2*int(w)*int(h))
                flow = np.resize(data, (int(h), int(w), 2))
                return flow

