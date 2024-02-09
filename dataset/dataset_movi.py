import json, os

import numpy as np
import torch
from PIL import Image
import imageio
import matplotlib.pyplot as plt
from skimage import transform
from torch._C import dtype
import copy


class Movi(object):
    def __init__(self, data_dir, args, mode):

        self.mode = mode
        self.device = "cpu"
        self.dtype = args.dtype

        self.data_dir = data_dir

        self.instance_list = np.load(self.data_dir+'/instance_list.npy').tolist()
        self.len = len(self.instance_list)

        self.patch_h = args.patch_h
        self.patch_w = args.patch_w
        self.enlarge_coef = args.enlarge_coef
        self.train_seq_len = args.seq_len
        self.fm_h = args.fm_h
        self.fm_w = args.fm_w

        self.cur_video_name = None
        self.num_frames = None
        self.camera = None
        self.height = None
        self.width = None
        self.depth = None
        self.forward_flow = None
        self.backward_flow = None
        self.video_frames = None
        self.video_path = None
        self.instance = None
        self.vis_instances_num = None
    
    def __len__(self):

        return self.len
            
    def __getitem__(self, idx, specified_V_O_id=None):
        if specified_V_O_id is None:
            video_name, obj = self.instance_list[idx].split("-")
        else:
            video_name, obj = specified_V_O_id.split("-")

        obj = int(obj)

        if video_name != self.cur_video_name:
            self.cur_video_name = video_name
            self.video_path = os.path.join(self.data_dir,video_name)
            data_ranges = self.read_json(os.path.join(self.video_path,'data_ranges.json'))
            metadata = self.read_json(os.path.join(self.video_path,'metadata.json'))

            self.num_frames = metadata["metadata"]["num_frames"]
            self.camera = self.format_camera_information(metadata)
            self.height = metadata['metadata']['height']
            self.width = metadata['metadata']['width']
            self.depth = self.get_depth(self.video_path) #[t,h,w,1]
            self.video_frames, self.bev_frames = self.get_img(self.video_path) #[t,h,w,3]
            self.instances = [self.format_instance_information(obj) for obj in metadata["instances"]]

            metadata_bev = self.read_json(os.path.join(self.video_path,'metadata_bev.json'))
            self.instances_bev = [self.format_instance_information(obj) for obj in metadata_bev["instances"]]
            n = 0
            for j in self.instances:
                if j['bbox_frames'] != []:
                    n+=1
            self.vis_instances_num = n


        vis_mask_paths = [self.video_path+'/'+f"segmentation_full_{f:05d}.png" for f in range(self.num_frames)]
        vis_mask = [np.array(Image.open(frame_path))==obj+1 for frame_path in vis_mask_paths] #[t,h,w]

        full_mask_paths = [self.video_path+'/'+"segmentation_"+str(obj)+f"_{f:05d}.png" for f in range(self.num_frames)]
        full_mask = [np.array(Image.open(frame_path))==obj+1 for frame_path in full_mask_paths] #[t,h,w]

        video_frames = self.video_frames
        bev_frames = self.bev_frames
        
        full_mask_bev_paths = [self.video_path+'/'+"segmentation_bev_"+str(obj)+f"_{f:05d}.png" for f in range(self.num_frames)]
        full_mask_bev = [np.array(Image.open(frame_path))==obj+1 for frame_path in full_mask_bev_paths]

        obj_patches = []
        obj_patches_crop = []
        obj_position = []
        obj_rate = []
        vm_crop = []
        vm_nocrop = []
        fm_crop = []
        fm_nocrop = []
        calib = []
        depth = []
        camera_position = []
        bev_img = []
        bev_mask = []

        timesteps = self.instances[int(obj)]['bbox_frames']
        start_t, end_t = 0, len(timesteps)-1
        if self.mode != "test" and len(timesteps) > self.train_seq_len:
            start_t = np.random.randint(start_t, end_t-self.train_seq_len+2)
            end_t = start_t + self.train_seq_len - 1
        
        for t_step in range(start_t, end_t+1):
            index = t_step
            t_step = timesteps[t_step]
            Image_H, Image_W = self.height, self.width
            
            #some objects will move out the field of view in some frames
            
            xmin, ymin, xmax, ymax = self.instances[int(obj)]["bboxes"][index]
            vx_min, vy_min, vx_max, vy_max = int(Image_H*xmin), int(Image_W*ymin), int(Image_H*xmax), int(Image_W*ymax)
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

            # get mask
            vm_crop.append(vis_mask[t_step][vx_min:vx_max+1, vy_min:vy_max+1])
            fm_crop.append(full_mask[t_step][vx_min:vx_max+1, vy_min:vy_max+1])

            # get patches and flow crop
            patch_crop = video_frames[t_step][vx_min:vx_max+1, vy_min:vy_max+1]
            patch = video_frames[t_step]

        

            vm_nocrop.append(vis_mask[t_step])
            fm_nocrop.append(full_mask[t_step])
            obj_patches.append(patch)
            obj_patches_crop.append(patch_crop)
            calib.append(torch.from_numpy(self.camera["intrinsics"]))
            depth.append(self.depth[t_step])
            camera_position.append(self.camera["positions"][t_step])


            xmin, ymin, xmax, ymax = self.instances_bev[int(obj)]["bboxes"][t_step]
            vx_min, vy_min, vx_max, vy_max = int(Image_H*xmin), int(Image_W*ymin), int(Image_H*xmax), int(Image_W*ymax)
            # enlarge the bbox
            x_center = (vx_min + vx_max) // 2
            y_center = (vy_min + vy_max) // 2
            x_len = int((vx_max - vx_min) * self.enlarge_coef)
            y_len = int((vy_max - vy_min) * self.enlarge_coef)
            vx_min = max(0, x_center - x_len // 2)
            vx_max = min(Image_H, x_center + x_len // 2)
            vy_min = max(0, y_center - y_len // 2)
            vy_max = min(Image_W, y_center + y_len // 2)
            bev_crop = bev_frames[t_step][vx_min:vx_max+1, vy_min:vy_max+1]
            bev_mask_crop = full_mask_bev[t_step][vx_min:vx_max+1, vy_min:vy_max+1]
            bev_img.append(bev_crop)
            bev_mask.append(bev_mask_crop)

        if self.mode!='test':
            num_pad = max(self.train_seq_len - (end_t - start_t + 1), 0)
        else:
            num_pad = 0
        for _  in range(num_pad):
            obj_position.append(copy.deepcopy(obj_position[-1]))
            obj_rate.append(copy.deepcopy(obj_rate[-1]))
            vm_crop.append(copy.deepcopy(vm_crop[-1]))
            vm_nocrop.append(copy.deepcopy(vm_nocrop[-1]))
            fm_crop.append(copy.deepcopy(fm_crop[-1]))
            fm_nocrop.append(copy.deepcopy(fm_nocrop[-1]))

            obj_patches.append(copy.deepcopy(obj_patches[-1]))
            obj_patches_crop.append(copy.deepcopy(obj_patches_crop[-1]))
            calib.append(copy.deepcopy(calib[-1]))
            depth.append(copy.deepcopy(depth[-1]))
            camera_position.append(copy.deepcopy(camera_position[-1]))

            bev_img.append(copy.deepcopy(bev_img[-1]))
            bev_mask.append(copy.deepcopy(bev_mask[-1]))
        
        obj_patches_crop = self.rescale_patch(obj_patches_crop)
        bev_img = self.rescale_patch(bev_img)
        fm_crop = self.fm_rescale(fm_crop)
        vm_crop = self.fm_rescale(vm_crop)
        bev_mask = self.fm_rescale(bev_mask)

        obj_position = torch.from_numpy(np.array(obj_position)).to(self.dtype).to(self.device)
        obj_rate = torch.from_numpy(np.array(obj_rate)).to(self.dtype).to(self.device)

        obj_temp = np.stack(obj_patches, axis=0) # Seq_len * patch_h * patch_w * 3
        obj_temp_crop = np.stack(obj_patches_crop, axis=0) # Seq_len * patch_h * patch_w * 3
        bev_img = np.stack(bev_img, axis=0)

        obj_patches = torch.from_numpy(obj_temp).to(self.dtype).to(self.device).permute(0,3,1,2)
        obj_patches_crop = torch.from_numpy(obj_temp_crop).to(self.dtype).to(self.device).permute(0,3,1,2)
        depth = torch.from_numpy(np.array(depth)).to(self.dtype).to(self.device).permute(0,3,1,2)
        vm_crop = torch.from_numpy(np.array(vm_crop)).to(self.dtype).to(self.device).unsqueeze(1)
        fm_crop = torch.from_numpy(np.array(fm_crop)).to(self.dtype).to(self.device).unsqueeze(1)
        vm_nocrop = torch.from_numpy(np.array(vm_nocrop)).to(self.dtype).to(self.device).unsqueeze(1)
        fm_nocrop = torch.from_numpy(np.array(fm_nocrop)).to(self.dtype).to(self.device).unsqueeze(1)
        calib = torch.stack(calib).to(self.dtype).to(self.device)
        camera_position = torch.from_numpy(np.array(camera_position)).to(self.dtype).to(self.device)
        bev_mask = torch.from_numpy(np.array(bev_mask)).to(self.dtype).to(self.device).unsqueeze(1)
        bev_img = torch.from_numpy(bev_img).to(self.dtype).to(self.device).permute(0,3,1,2)

        obj_data = {"input_obj_patches": obj_patches, 
                    "obj_patches_crop": obj_patches_crop,
                    "vm_nocrop" : vm_nocrop,  
                    "fm_nocrop" : fm_nocrop,
                    "vm_crop" : vm_crop,  
                    "fm_crop": fm_crop,
                    "calib": calib,
                    "obj_position": obj_position, 
                    "obj_rate": obj_rate, 
                    "depth": depth,
                    "camera_position": camera_position,
                    "bev_img": bev_img,
                    "bev_mask": bev_mask,
                    }

        return obj_data

        
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

    def rescale_patch(self, obj_patches=None):
        for i, obj_pt in enumerate(obj_patches):
            h,w = obj_pt.shape[:2]
            obj_pt = transform.rescale(obj_pt, (self.patch_h/h, self.patch_w/w, 1))
            cur_h, cur_w = obj_pt.shape[:2]
            to_pad = ((0, max(self.patch_h-cur_h, 0)), (0, max(self.patch_w-cur_w, 0)), (0, 0))
            obj_pt = np.pad(obj_pt, to_pad)[:self.patch_h,:self.patch_w,:3]
            obj_patches[i] = obj_pt
        return obj_patches

    def rescale(self, masks, obj_patches=None, flows=None, flows_reverse=None):
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
              
        for i, flow in enumerate(flows):
            flow = transform.rescale(flow, (self.patch_h/h, self.patch_w/w, 1))
            cur_h, cur_w = flow.shape[:2]
            to_pad = ((0, max(self.patch_h-cur_h, 0)), (0, max(self.patch_w-cur_w, 0)), (0, 0))
            flow = np.pad(flow, to_pad)[:self.patch_h,:self.patch_w,:2]
            flows[i] = flow
        
        for i, flow_r in enumerate(flows_reverse):
            flow_r = transform.rescale(flow_r, (self.patch_h/h, self.patch_w/w, 1))
            cur_h, cur_w = flow_r.shape[:2]
            to_pad = ((0, max(self.patch_h-cur_h, 0)), (0, max(self.patch_w-cur_w, 0)), (0, 0))
            flow_r = np.pad(flow_r, to_pad)[:self.patch_h,:self.patch_w,:2]
            flows_reverse[i] = flow_r

        return masks, obj_patches, flows, flows_reverse
    

    def read_json(self,dir_):
        with open(dir_) as f:
            data = json.load(f)
        return data

    def format_camera_information(self,metadata):
        return {
            "intrinsics": np.array(metadata["camera"]["K"], np.float32),
            "extrinsics": np.array(metadata["camera"]["R"], np.float32),
            "focal_length": metadata["camera"]["focal_length"],
            "sensor_width": metadata["camera"]["sensor_width"],
            "field_of_view": metadata["camera"]["field_of_view"],
            "positions": metadata["camera"]["positions"],
            "quaternions": np.array(metadata["camera"]["quaternions"], np.float32),
        }

    def read_tiff(self, filename):
        img = imageio.imread(filename, format="tiff")
        if img.ndim == 2:
            img = img[:, :, None]
        return img
    
    def get_depth(self, video_path):
        depth_paths = [video_path+'/'+f"depth_full_{f:05d}.tiff" for f in range(self.num_frames)]
        depth_frames = np.array([self.read_tiff(frame_path) for frame_path in depth_paths])
        return depth_frames

    def get_flow(self,video_path,data_ranges):
        forward_flow_min, forward_flow_max = data_ranges["forward_flow"]["min"], data_ranges["forward_flow"]["max"]
        backward_flow_min, backward_flow_max = data_ranges["backward_flow"]["min"], data_ranges["backward_flow"]["max"]
        forward_flow_paths = [video_path+'/'+f"forward_flow_full_{f:05d}.png" for f in range(self.num_frames)]
        backward_flow_paths = [video_path+'/'+f"backward_flow_full_{f:05d}.png" for f in range(self.num_frames)]
        forward_flow_frames = np.array([np.array(Image.open(frame_path),np.uint16)[...,:2]/65535*(forward_flow_max-forward_flow_min)+forward_flow_min 
                for frame_path in forward_flow_paths])
        backward_flow_frames = np.array([np.array(Image.open(frame_path),np.uint16)[...,:2]/65535*(backward_flow_max-backward_flow_min)+backward_flow_min
                for frame_path in backward_flow_paths])
        return forward_flow_frames, backward_flow_frames

    def get_img(self,video_path):
        rgba_paths = [video_path+'/'+f"rgba_full_{f:05d}.png" for f in range(self.num_frames)]
        rgba_frames = np.array([plt.imread(frame_path)[...,:3] for frame_path in rgba_paths])
        rgba_bev_paths = [video_path+'/'+f"rgba_bev_full_{f:05d}.png" for f in range(self.num_frames)]
        rgba_bev_frames = np.array([plt.imread(frame_path)[...,:3] for frame_path in rgba_bev_paths])
        return rgba_frames, rgba_bev_frames

    def format_instance_information(self, obj):
        return {
            "bboxes_3d": obj["bboxes_3d"],
            "bboxes": obj["bboxes"],
            "bbox_frames": obj["bbox_frames"],
        }

