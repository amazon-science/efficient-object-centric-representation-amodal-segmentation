import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import scipy.linalg as linalg
import numpy as np
from .downsample import DownChannel, Down

class BEVTrans(nn.Module):
    def __init__(self, args, downconv='DownSample', width = [-10,10], height = [-3.5,1.5], depth = [0,20], 
                width_resolution = 100, height_resolution = 5, depth_resolution = 100, C=256):
        super(BEVTrans, self).__init__()
        self.width = width
        self.height = height
        self.depth = depth

        self.width_len = width[1]-width[0]
        self.height_len = height[1]-height[0]
        self.depth_len = depth[1]-depth[0]

        self.width_resolution = width_resolution
        self.height_resolution = height_resolution
        self.depth_resolution = depth_resolution

        self.depth_span = torch.linspace(depth[0]+self.depth_len/depth_resolution/2, depth[1]-self.depth_len/depth_resolution/2, steps=depth_resolution)
        self.width_span = torch.linspace(width[0]+self.width_len/width_resolution/2, width[1]-self.width_len/width_resolution/2, steps=width_resolution)
        self.height_span = torch.linspace(height[0]+self.height_len/height_resolution/2, height[1]-self.height_len/height_resolution/2, steps=height_resolution)

        X, Y, Z = torch.meshgrid(self.width_span, self.height_span, self.depth_span)
        feature_3d_position = torch.stack((X, Y, Z), 3).view(-1,3).numpy()

        if hasattr(args, "camera_position"):
            camera_pos = args.camera_position
            axis = [1,0,0]
            radian = np.arctan(camera_pos[2]/math.sqrt(camera_pos[0]**2+camera_pos[1]**2))
            rot_matrix = linalg.expm(np.cross(np.eye(3), axis / linalg.norm(axis) * radian))

            self.feature_3d_position = torch.from_numpy(rot_matrix@feature_3d_position.T).to(torch.float32).cuda()
        else:
            self.feature_3d_position = torch.stack((X, Y, Z), 3).view(-1,3).to(torch.float32).cuda().T
        
        if downconv == 'DownChannel':
            in_channel = C*height_resolution
            self.conv = DownChannel([in_channel, in_channel//2], 
                                    [in_channel//2, C])
        elif downconv == 'DownSample':
            in_channel = C*height_resolution
            self.conv = Down([in_channel, in_channel//2], 
                             [in_channel//2, C])

    def forward(self, X, calib):
        bs = X.shape[0]
        C = X.shape[1]
        feature_3d = []
        
        for i in range(bs):
            img_pos = calib[i]@self.feature_3d_position
            img_z = img_pos.clone()[2:3,:]
            img_pos = 2*((img_pos/img_z)[:2]-0.5)
            img_pos = img_pos.T.view(1,self.width_resolution,self.height_resolution*self.depth_resolution,2)
            feature_3d.append(img_pos)
        feature_3d = torch.cat(feature_3d,dim=0)
        feature_3d = F.grid_sample(X, grid=feature_3d, mode='bilinear', align_corners=True)
        feature_3d = feature_3d.view(bs,C,self.width_resolution,self.height_resolution,self.depth_resolution)
        feature_3d = feature_3d.permute(0,1,3,4,2).reshape(bs,-1,self.depth_resolution,self.width_resolution)
        return self.conv(feature_3d)