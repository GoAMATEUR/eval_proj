import torch
import pdb
import numpy as np
import torch
import pdb
import numpy as np
from torch import nn
from torch.nn import functional as F

from utils.registry import Registry
from model import registry
from model.layers.utils import sigmoid_hm
from model.make_layers import group_norm, _fill_fc_weights
from model.layers.utils import select_point_of_interest
# from model.backbone.DCNv2.dcn_v2 import DCNv2

# from inplace_abn import InPlaceABN

@registry.PREDICTOR.register("Base_Predictor")
class _predictor(nn.Module):
    def __init__(self, cfg, in_channels):
        super(_predictor, self).__init__()
        # ("Car", "Cyclist", "Pedestrian")
        classes = len(cfg.DATASETS.DETECT_CLASSES)
        
        self.regression_head_cfg = cfg.MODEL.HEAD.REGRESSION_HEADS
        self.regression_channel_cfg = cfg.MODEL.HEAD.REGRESSION_CHANNELS
        self.output_width = cfg.INPUT.WIDTH_TRAIN // cfg.MODEL.BACKBONE.DOWN_RATIO
        self.output_height = cfg.INPUT.HEIGHT_TRAIN // cfg.MODEL.BACKBONE.DOWN_RATIO
        
        self.head_conv = cfg.MODEL.HEAD.NUM_CHANNEL

        use_norm = cfg.MODEL.HEAD.USE_NORMALIZATION
        if use_norm == 'BN': norm_func = nn.BatchNorm2d
        elif use_norm == 'GN': norm_func = group_norm
        else: norm_func = nn.Identity

        # the inplace-abn is applied to reduce GPU memory and slightly increase the batch-size
        self.use_inplace_abn = False
        self.bn_momentum = cfg.MODEL.HEAD.BN_MOMENTUM
        self.abn_activision = 'leaky_relu'

        ###########################################
        ###############  Cls Heads ################
        ########################################### 

        if not self.use_inplace_abn:
            
            self.class_head = nn.Sequential(
                nn.Conv2d(in_channels, self.head_conv, kernel_size=3, padding=1, bias=False),
                norm_func(self.head_conv, momentum=self.bn_momentum), nn.ReLU(inplace=True),
                nn.Conv2d(self.head_conv, classes, kernel_size=1, padding=1 // 2, bias=True),
            )
        
        self.class_head[-1].bias.data.fill_(- np.log(1 / cfg.MODEL.HEAD.INIT_P - 1))

        ###########################################
        ############  Regression Heads ############
        ########################################### 
        
        # init regression heads
        self.reg_features = nn.ModuleList()
        self.reg_heads = nn.ModuleList()

        # init regression heads
        for idx, regress_head_key in enumerate(self.regression_head_cfg):
            if not self.use_inplace_abn:
                
                feat_layer = nn.Sequential(nn.Conv2d(in_channels, self.head_conv, kernel_size=3, padding=1, bias=False),
                                    norm_func(self.head_conv, momentum=self.bn_momentum), nn.ReLU(inplace=True))
            
            self.reg_features.append(feat_layer)
            # init output head
            head_channels = self.regression_channel_cfg[idx]
            head_list = nn.ModuleList()
            for key_index, key in enumerate(regress_head_key):
                key_channel = head_channels[key_index]
                output_head = nn.Conv2d(self.head_conv, key_channel, kernel_size=1, padding=1 // 2, bias=True)

                if key.find('uncertainty') >= 0 and cfg.MODEL.HEAD.UNCERTAINTY_INIT:
                    torch.nn.init.xavier_normal_(output_head.weight, gain=0.01)
                
                # since the edge fusion is applied to the offset branch, we should save the index of this branch
                if key == '3d_offset': self.offset_index = [idx, key_index]

                _fill_fc_weights(output_head, 0)
                head_list.append(output_head)

            self.reg_heads.append(head_list)

        ###########################################
        ##############  Edge Feature ##############
        ###########################################

        # edge feature fusion
        self.enable_edge_fusion = cfg.MODEL.HEAD.ENABLE_EDGE_FUSION
        self.edge_fusion_kernel_size = cfg.MODEL.HEAD.EDGE_FUSION_KERNEL_SIZE
        self.edge_fusion_relu = cfg.MODEL.HEAD.EDGE_FUSION_RELU

        if self.enable_edge_fusion:
            trunc_norm_func = nn.BatchNorm1d if cfg.MODEL.HEAD.EDGE_FUSION_NORM == 'BN' else nn.Identity
            trunc_activision_func = nn.ReLU(inplace=True) if self.edge_fusion_relu else nn.Identity()
            
            self.trunc_heatmap_conv = nn.Sequential(
                nn.Conv1d(self.head_conv, self.head_conv, kernel_size=self.edge_fusion_kernel_size, padding=self.edge_fusion_kernel_size // 2, padding_mode='replicate'),
                trunc_norm_func(self.head_conv, momentum=self.bn_momentum), trunc_activision_func, nn.Conv1d(self.head_conv, classes, kernel_size=1),
            )
            
            self.trunc_offset_conv = nn.Sequential(
                nn.Conv1d(self.head_conv, self.head_conv, kernel_size=self.edge_fusion_kernel_size, padding=self.edge_fusion_kernel_size // 2, padding_mode='replicate'),
                trunc_norm_func(self.head_conv, momentum=self.bn_momentum), trunc_activision_func, nn.Conv1d(self.head_conv, 2, kernel_size=1),
            )

    def forward(self, features):
        b, c, h, w = features.shape

        # output classification
        feature_cls = self.class_head[:-1](features)
        output_cls = self.class_head[-1](feature_cls)

        output_regs = []
        # output regression
        for i, reg_feature_head in enumerate(self.reg_features):
            reg_feature = reg_feature_head(features)

            for j, reg_output_head in enumerate(self.reg_heads[i]):
                output_reg = reg_output_head(reg_feature)
                output_regs.append(output_reg)

        output_cls = sigmoid_hm(output_cls)
        output_regs = torch.cat(output_regs, dim=1)

        return  output_cls,  output_regs

def make_predictor(cfg, in_channels):
    func = registry.PREDICTOR[cfg.MODEL.HEAD.PREDICTOR]
    return func(cfg, in_channels)