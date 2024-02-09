import torch
import torch.nn as nn

from .fpn import FPN50
from .transformer import TransformerDecoder, TransformerDecoderLayer
from .decoder import decoder_16to121_bid
from .utils import sin_cos_pos
from .bev import BEVTrans

class Amodal_imgcrop_simplebevcrop_bid(nn.Module):
    def __init__(self,args):
        super(Amodal_imgcrop_simplebevcrop_bid, self).__init__()
        self.d_model = args.d_model
        self.fea_dim = args.f_dim

        self.feature_extractor = FPN50(first_channel=3)
        self.bev_generator = BEVTrans(args, width_resolution = 64, depth_resolution = 64)

        self.slot_update_bev = TransformerDecoder(TransformerDecoderLayer(args.d_model, args.n_head), args.num_layers)
        self.slot_update_feature = TransformerDecoder(TransformerDecoderLayer(args.d_model, args.n_head), args.num_layers)
        self.feature_update = TransformerDecoder(TransformerDecoderLayer(args.d_model, args.n_head), args.num_layers)

        self.fea_fc = nn.Linear(256,self.d_model)
        self.bev_fc = nn.Linear(256,self.d_model)

        self.dec_fm = decoder_16to121_bid(args)
        self.dec_type = args.decoder
        if args.decoder.find('vm')>-1:
            self.dec_vm = decoder_16to121_bid(args)
        if args.decoder.find('occ')>-1:
            self.dec_occ = decoder_16to121_bid(args)

        self.slot = nn.Embedding(args.num_slot, args.d_model)
        self.pos_type = args.pos_type
        if args.pos_type == 'random':
            self.slot_pos = nn.Embedding(args.num_slot, args.d_model)
            self.fea_pos = nn.Embedding(args.f_dim**2, args.d_model)
            self.bev_pos = nn.Embedding(args.f_dim**2, args.d_model)
        elif args.pos_type == 'sincos':
            self.slot_pos = nn.Parameter(sin_cos_pos(args.num_slot, args.d_model), requires_grad=False)
            self.fea_pos = nn.Parameter(sin_cos_pos(args.f_dim**2, args.d_model), requires_grad=False)
            self.bev_pos = nn.Parameter(sin_cos_pos(args.f_dim**2, args.d_model), requires_grad=False)

    def forward(self, obj_pt_crop, calibration):

        bs, t, c,h,w = obj_pt_crop.shape 

        ans = {'full_mask':[], "vis_mask":[], "occ_mask":[]}
        features = []
        bevs = []

        for i in range(t):
            obj_crop = obj_pt_crop[:,i,...]
            calib = calibration[:,i,...]

            feature_maps = self.feature_extractor(obj_crop)
            cropped_feature = feature_maps[0]
            bev = self.bev_generator(cropped_feature, calib)

            
            _, bev_channel, bev_h, bev_w = bev.shape
            bev = bev.view(bs,bev_channel,-1).transpose(1,2)
            bev = self.bev_fc(bev)

            _, feature_channel, feature_h, feature_w = cropped_feature.shape 
            cropped_feature = cropped_feature.view(bs,feature_channel,-1).transpose(1,2)
            cropped_feature = self.fea_fc(cropped_feature)

            features.append(cropped_feature)
            bevs.append(bev)


        if self.pos_type == 'random':
            slots_pos = torch.stack([self.slot_pos.weight]*bs)
            fea_pos = torch.stack([self.fea_pos.weight]*bs)
            bev_pos = torch.stack([self.bev_pos.weight]*bs)

        elif self.pos_type == 'sincos':
            slots_pos = torch.cat([self.slot_pos]*bs)
            fea_pos = torch.cat([self.fea_pos]*bs)
            bev_pos = torch.cat([self.bev_pos]*bs)

        slots = torch.stack([self.slot.weight]*bs)
        forward_features = []
        for feature, bev in zip(features,bevs):
            slots = self.slot_update_bev(slots, bev, pos=bev_pos, query_pos=slots_pos)
            slots = self.slot_update_feature(slots, feature, pos=fea_pos, query_pos=slots_pos)
            update_feature = self.feature_update(feature, slots, pos=slots_pos, query_pos=fea_pos)

            update_feature = update_feature.transpose(1,2).view(bs,self.d_model,self.fea_dim,self.fea_dim)
            forward_features.append(update_feature)

        slots = torch.stack([self.slot.weight]*bs)
        backward_features = []
        for feature, bev in zip(reversed(features),reversed(bevs)):
            slots = self.slot_update_bev(slots, bev, pos=bev_pos, query_pos=slots_pos)
            slots = self.slot_update_feature(slots, feature, pos=fea_pos, query_pos=slots_pos)
            update_feature = self.feature_update(feature, slots, pos=slots_pos, query_pos=fea_pos)

            update_feature = update_feature.transpose(1,2).view(bs,self.d_model,self.fea_dim,self.fea_dim)
            backward_features.insert(0, update_feature)

        for i in range(t):
            update_feature = torch.cat([forward_features[i],backward_features[i]],dim=1)   
            full_mask = self.dec_fm(update_feature)
            ans['full_mask'].append(full_mask.unsqueeze(1))
            if self.dec_type.find('vm')>-1:
                vis_mask = self.dec_vm(update_feature)
                ans['vis_mask'].append(vis_mask.unsqueeze(1))
            if self.dec_type.find('occ')>-1:
                occ_mask = self.dec_occ(update_feature)
                ans['occ_mask'].append(occ_mask.unsqueeze(1))
        
        re = {'full_mask':torch.cat(ans['full_mask'],dim=1)}
        if self.dec_type.find('vm')>-1:
            re['vis_mask'] = torch.cat(ans['vis_mask'],dim=1)
        if self.dec_type.find('occ')>-1:
            re['occ_mask'] = torch.cat(ans['occ_mask'],dim=1)

        return re

if __name__ == '__main__':
    class A:
        def __init__(self):
            self.d_model = 512
            self.f_dim = 16
            self.n_head = 8
            self.num_layers = 1
            self.num_slot = 1
            self.camera_position = [1.0,0.0,0.0]
            self.decoder = 'fm+vm'
            self.pos_type = 'sincos'

    args = A()

    net = Amodal_imgcrop_simplebevcrop_bid(args).cuda()
    x = torch.rand(1,3,128,128).unsqueeze(0).cuda()
    calib = torch.eye(3).unsqueeze(0).unsqueeze(0).cuda()
    y = net(x,calib)
    print(y)