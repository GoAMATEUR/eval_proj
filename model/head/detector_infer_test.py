import os
import torch
import pdb
import numpy as np
import pdb
from torch import imag, nn
from shapely.geometry import Polygon
from torch.nn import functional as F
from model.layers.utils import (
    nms_hm,
    select_topk,
    select_point_of_interest,
)

PI = np.pi

class postprocess(nn.Module):
    def __init__(self,input_width=640,input_height=320, det_thres=0.29):
        super(postprocess, self).__init__()
        self.regression_head_cfg = [['2d_dim'], ['3d_offset'],['corner_offset'],['3d_dim'], ['ori_cls', 'ori_offset'], ['depth']]
        self.regression_channel_cfg = [[4, ], [2, ], [20], [3, ], [8, 8], [1, ]]
        self.input_width = input_width
        self.input_height = input_height
        self.down_ratio = 4
        self.depth_mode = 'inv_sigmoid'
        self.depth_range =  [0.1, 200]
        self.alpha_centers = torch.tensor([0, PI / 2, PI, - PI / 2]).to('cuda') 
        self.multibin = True
        self.orien_bin_size = 4
        self.dim_modes = ['exp', True, False]
        self.output_width = self.input_width  // self.down_ratio
        self.output_height = self.input_height // self.down_ratio
        self.max_detection = 50 
        self.head_conv = 256
        self.det_threshold = det_thres ## 0.25改为0.29
        self.dim_mean =  torch.as_tensor(((3.99331126, 1.54370861, 1.64175497),
                               (0.295, 1.6, 0.3175),
                               (1.34645161, 1.55322581, 0.3883871),
                               (2.503, 1.72 , 1.077),
                               (9.1775, 2.95, 2.3425),
                               (10.3655102,3.31632653,2.45469388),
                               (6.016911083,3.412001685,2.2783185),
                               (4.824963,2.046904,1.78939),
                                (8.8040879,2.9161930,2.07649252))).to('cuda')
        self.dim_std =  torch.as_tensor(((0.35223078,0.19156938,0.12989004),
                                (0.08789198,0.14142136,0.06647368),
                                (0.23795565,0.06148103,0.05400698),
                                (0.60405381,0.07483315,0.22275772),
                                (0.82898658,0.05     , 0.06684871),
                                (2.53467307,0.54747917,0.31232572),
                               (3.68131074,1.02058,0.6805663),
                                (1.232056,0.318495,0.242334),
                               (4.68605842,1.237353625,0.853123))).to('cuda')

    
    def decode_dimension(self, cls_id, dims_offset):
        cls_id = cls_id.flatten().long()
        if cls_id.is_cuda is False:
            cls_dimension_mean = self.dim_mean[cls_id, :].to('cpu')
        else:
            cls_dimension_mean = self.dim_mean[cls_id, :]

        if self.dim_modes[0] == 'exp':
            dims_offset = dims_offset.exp()

        if self.dim_modes[2]:
            cls_dimension_std = self.dim_std[cls_id, :]
            dimensions = dims_offset * cls_dimension_std + cls_dimension_mean
        else:
            dimensions = dims_offset * cls_dimension_mean
            
        return dimensions
    
    def decode_depth(self, depths_offset):

        if self.depth_mode == 'exp':
            depth = depths_offset.exp()
        elif self.depth_mode == 'linear':
            depth = depths_offset * self.depth_ref[1] + self.depth_ref[0]
        elif self.depth_mode == 'inv_sigmoid':
            depth = 1 / torch.sigmoid(depths_offset) - 1
        else:
            raise ValueError

        if self.depth_range is not None:
            depth = torch.clamp(depth, min=self.depth_range[0], max=self.depth_range[1])

        return depth

    def decode_location_flatten(self, points, offsets, depths, calibs, batch_idxs):
        batch_size = 1# len(calibs)
        gts = torch.unique(batch_idxs, sorted=True).tolist()
        # locations = points.new_zeros(points.shape[0], 3).float()
        locations = torch.zeros((points.shape[0], 3), dtype=points.dtype, device=points.device).float()
        points = (points + offsets) * self.down_ratio #- pad_size[batch_idxs]

        for idx, gt in enumerate(gts):
            corr_pts_idx = torch.nonzero(batch_idxs == gt).squeeze(-1)
            calib = calibs
            # print("calib P", calib.P,calib.b_x,calib.b_y, calib.f_u, calib.f_v, calib.c_u, calib.c_v)
            # pdb.set_trace()
            # concatenate uv with depth
            corr_pts_depth = torch.cat((points[corr_pts_idx], depths[corr_pts_idx, None]), dim=1)
            locations[corr_pts_idx] = calib.project_image_to_rect(corr_pts_depth)

        return locations
    
    def decode_axes_orientation(self, vector_ori, locations):
    
        if self.multibin:
            pred_bin_cls = vector_ori[:, : self.orien_bin_size * 2].view(-1, self.orien_bin_size, 2)
            pred_bin_cls = torch.softmax(pred_bin_cls, dim=2)[..., 1]
            # orientations = vector_ori.new_zeros(vector_ori.shape[0])
            orientations = torch.zeros(vector_ori.shape[0], device=vector_ori.device, dtype=vector_ori.dtype)
            for i in range(self.orien_bin_size):
                mask_i = (pred_bin_cls.argmax(dim=1) == i)
                s = self.orien_bin_size * 2 + i * 2
                e = s + 2
                pred_bin_offset = vector_ori[mask_i, s : e]
                orientations[mask_i] = torch.atan(pred_bin_offset[:, 0]/ pred_bin_offset[:, 1]) + self.alpha_centers[i]
        else:
            axis_cls = torch.softmax(vector_ori[:, :2], dim=1)
            axis_cls = axis_cls[:, 0] < axis_cls[:, 1]
            head_cls = torch.softmax(vector_ori[:, 2:4], dim=1)
            head_cls = head_cls[:, 0] < head_cls[:, 1]
            # cls axis
            orientations = self.alpha_centers[axis_cls + head_cls * 2]
            sin_cos_offset = F.normalize(vector_ori[:, 4:])
            orientations += torch.atan(sin_cos_offset[:, 0] / sin_cos_offset[:, 1])

        locations = locations.view(-1, 3)
        rays = torch.atan(locations[:, 0] / locations[:, 2])
        alphas = orientations
        rotys = alphas + rays

        larger_idx = (rotys > PI).nonzero()
        small_idx = (rotys < -PI).nonzero()
        if len(larger_idx) != 0:
                rotys[larger_idx] -= 2 * PI
        if len(small_idx) != 0:
                rotys[small_idx] += 2 * PI

        larger_idx = (alphas > PI).nonzero()
        small_idx = (alphas < -PI).nonzero()
        if len(larger_idx) != 0:
                alphas[larger_idx] -= 2 * PI
        if len(small_idx) != 0:
                alphas[small_idx] += 2 * PI

        return rotys, alphas
    
    def decode_orientation(self, vector_ori):

        pred_bin_cls = vector_ori[:, : self.orien_bin_size * 2].view(-1, self.orien_bin_size, 2)
        pred_bin_cls = torch.softmax(pred_bin_cls, dim=2)[..., 1]
        
        orientations = torch.zeros(vector_ori.shape[0], device=vector_ori.device, dtype=vector_ori.dtype)
        
        for i in range(self.orien_bin_size):
            mask_i = (pred_bin_cls.argmax(dim=1) == i)
            s = self.orien_bin_size * 2 + i * 2
            e = s + 2
            pred_bin_offset = vector_ori[mask_i, s : e]
            orientations[mask_i] = torch.atan(pred_bin_offset[:, 0]/ pred_bin_offset[:, 1]) + self.alpha_centers[i]
        alphas = orientations

        larger_idx = (alphas > PI).nonzero()
        small_idx = (alphas < -PI).nonzero()
        if len(larger_idx) != 0:
                alphas[larger_idx] -= 2 * PI
        if len(small_idx) != 0:
                alphas[small_idx] += 2 * PI

        return alphas 

    def decode_box2d_fcos(self,centers, pred_offset, out_size=None):
        box2d_center = centers.view(-1, 2)
        #box2d = box2d_center.new(box2d_center.shape[0], 4).zero_()
        box2d = torch.zeros((centers.shape[0],4),device=centers.device, dtype=centers.dtype)
        # left, top, right, bottom
        box2d[:, :2] = box2d_center - pred_offset[:, :2]
        box2d[:, 2:] = box2d_center + pred_offset[:, 2:]
        box2d = box2d * self.down_ratio
        box2d[:,0::2] = torch.clamp(box2d[:,0::2], min=0, max=self.input_width)
        box2d[:,1::2] = torch.clamp(box2d[:,1::2], min=0, max=self.input_height)

        return box2d
    
    def decode_box2d_cxcywh(self, centers, pred_offset, pad_size=None, out_size=None):
        box2d_center = centers.view(-1, 2)
        box2d = box2d_center.new(box2d_center.shape[0], 4).zero_()
        box2d_center = (box2d_center+pred_offset[:, :2])* self.down_ratio
        # left, top, right, bottom
        box2d[:, :2] = box2d_center - pred_offset[:, 2:]* self.down_ratio/2.0
        box2d[:, 2:] = box2d_center + pred_offset[:, 2:]* self.down_ratio/2.0
        
        # print(pad_size)
        # print(box2d)
        # for inference
        if pad_size is not None:
            N = box2d.shape[0]
            out_size = out_size[0]
            # upscale and subtract the padding
            box2d = box2d - pad_size.repeat(1, 2)
            # clamp to the image bound
            box2d[:, 0::2].clamp_(min=0, max=out_size[0].item() - 1)
            box2d[:, 1::2].clamp_(min=0, max=out_size[1].item() - 1)

        return box2d

    def del_dul_id(self,scores, indexs):
        valid_mask =  []
        gts = torch.unique(indexs, sorted=True).tolist()
        for idx, gt in enumerate(gts):
            corr_pts_idx = torch.nonzero(indexs == gt).squeeze(-1)
            topk_scores, topk_inds = torch.topk(scores[corr_pts_idx], 1)
            valid_mask.append(corr_pts_idx[topk_inds])
#             print(topk_inds)             
        return torch.as_tensor(valid_mask)

    def forward(self,output_cls,output_regs,calib):

        heatmap = nms_hm(output_cls,kernel=5)
        scores, indexs, clses, ys, xs = select_topk(heatmap, K=self.max_detection)
        
        pred_bbox_points = torch.cat([xs.view(-1, 1), ys.view(-1, 1)], dim=1)
        pred_regression_pois = select_point_of_interest(output_regs.shape[0], indexs, output_regs).view(-1, output_regs.shape[1])
        scores = scores.view(-1)
        indexs = indexs.view(-1)
        # print("score", scores)
        valid_mask = scores >= self.det_threshold
        # print(valid_mask)
        # pdb.set_trace()
        if valid_mask.sum() == 0:
            return None,None,None,None,None,None,None,None,None,None,None 
        
        visualize_preds ={}
        clses = clses.view(-1)[valid_mask]
        # print(clses)
        pred_bbox_points = pred_bbox_points[valid_mask]
        # print("pred_bbox_points :",pred_bbox_points)
        pred_regression_pois = pred_regression_pois[valid_mask]
        scores = scores[valid_mask]
        indexs = indexs[valid_mask]
        
        valid_mask_duplicate = self.del_dul_id(clses, indexs)
        #print(valid_mask_duplicate)
        clses = clses[valid_mask_duplicate]
        # print(clses)
        pred_bbox_points = pred_bbox_points[valid_mask_duplicate]
        pred_regression_pois = pred_regression_pois[valid_mask_duplicate]
        scores = scores[valid_mask_duplicate]
        indexs = indexs[valid_mask_duplicate]
        # print(scores)
#         print("indexs ",indexs)
#         print("pred_regression_pois.shape : ",pred_regression_pois.shape)
        # print("pred_2d_reg before",pred_regression_pois[:, 0:4])
        pred_2d_reg = F.relu(pred_regression_pois[:, 0:4])
        # print("pred_2d_reg ",pred_2d_reg)
        pred_offset_3D = (pred_regression_pois[:,4:6])
        # print("pred_offset_3D ",pred_offset_3D)
        pred_dimensions_offsets = pred_regression_pois[:, 22:25]
        pred_orientation = torch.cat((pred_regression_pois[:,25:33], pred_regression_pois[:,33:41]), dim=1)

        ppred_bbox_points = pred_bbox_points + pred_offset_3D
        
        pred_box2d = self.decode_box2d_fcos(ppred_bbox_points, pred_2d_reg)
        
        pred_dimensions = self.decode_dimension(clses, pred_dimensions_offsets)
        # print("pred_dimensions ",pred_dimensions)
        pred_depths_offset = pred_regression_pois[:, 41].squeeze(-1)
        pred_depths = self.decode_depth(pred_depths_offset)
        # print("pred_depths ",pred_depths)
        pred_depths = pred_depths.reshape(-1)
        pred_keypoint_offset = pred_regression_pois[:, 6:14]
        pred_keypoint_offset = pred_keypoint_offset.view(-1, 4, 2)			
        pred_keypoint_visible = pred_regression_pois[:, 14:22]
        pred_keypoint_visible = pred_keypoint_visible.view(-1, 4, 2)
        pred_center_type = pred_regression_pois[:, 42:47]
        
        pred_center_type = torch.softmax(pred_center_type, dim=1)
        pred_center_type = pred_center_type.argmax(dim=1)
        # print("pred_center_type ",pred_center_type)
        # pdb.set_trace()
        #pred_h = (pred_box2d[:,3]-pred_box2d[:,1])/2 
        #print("pred_h: ",pred_h.shape)
         #+ pred_h.reshape(-1,1)
        #print("ppred_bbox_points: ",ppred_bbox_points.shape)
        # print("pred_keypoint_offset: ",pred_keypoint_offset)
        # print("pred_keypoint: ",(pred_keypoint_offset.reshape(-1, 4,2) + ppred_bbox_points.reshape(-1, 1, 2)))
        
        # print(bbox_dim.shape)
        # print(ppred_bbox_points.shape)
        # print(pred_keypoint_offset.shape)
        # print(bbox_dim.view(-1,1,2))
        box_center = (pred_box2d[:,:2] + pred_box2d[:,2:]) / 2
        
        ## 25D
        pred_keypoint = pred_keypoint_offset.reshape(-1, 4,2)* 4 + pred_bbox_points.reshape(-1, 1, 2)*4 ## pred_bbox_points for aiv
        # pred_keypoint = pred_keypoint_offset.reshape(-1, 4,2)* 4 + ppred_bbox_points.reshape(-1, 1, 2)*4 ## ppred_bbox points for hh
        
        # bbox_dim = pred_box2d[:,2:] - pred_box2d[:,:2]
        # pred_keypoint = pred_keypoint_offset.view(-1, 4, 2)*bbox_dim.view(-1,1,2) + ppred_bbox_points.view(-1, 1, 2) * 4
        pred_keypoint_visible = torch.softmax(pred_keypoint_visible, dim=2)
        pred_keypoint_visible = pred_keypoint_visible[:,:, 0] < pred_keypoint_visible[:,:, 1]
        # print("pred_keypoint_visible: ",pred_keypoint_visible.shape)
        batch_idxs = pred_depths.new_zeros(pred_depths.shape[0]).long()
        pred_locations = self.decode_location_flatten(pred_bbox_points, pred_offset_3D, pred_depths, calib, batch_idxs)
        # print("pred_locations :",pred_locations)
        pred_rotys, pred_alphas = self.decode_axes_orientation(pred_orientation, pred_locations)
        # print("pred_rotys : ",pred_rotys, " pred_alphas : ",pred_alphas)
        pred_locations[:, 1] += pred_dimensions[:, 1] / 2
        clses = clses.view(-1, 1)
        pred_alphas = pred_alphas.view(-1, 1)
        pred_rotys = pred_rotys.view(-1, 1)
        scores = scores.view(-1, 1)
        # change dimension back to h,w,l
        pred_dimensions = pred_dimensions.roll(shifts=-1, dims=1)
        ppred_bbox_points = pred_bbox_points*4
        
        
        return clses,pred_alphas,pred_rotys, pred_box2d, pred_dimensions,scores ,pred_locations,pred_keypoint,pred_keypoint_visible,ppred_bbox_points,pred_center_type




