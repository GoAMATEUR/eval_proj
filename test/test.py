
import sys
from tkinter import N
sys.path.append('/home/utopilot/workspace/infer/infer_test/')
import os,cv2
import torch
import uuid,pdb
from torch import nn
from config import cfg
from PIL import Image
from skimage import transform as trans 
import numpy as np
import datetime
from config import TYPE_ID_CONVERSION
from utils.check_point import DetectronCheckpointer
from model.backbone import build_backbone
from engine import (
    default_argument_parser,
    default_setup,
    launch,
)
import re
from shutil import copyfile
from utils.kitti_utils import  read_label,Calibration
from model.head.detector_predictor_test import make_predictor
from model.head.detector_infer_test import postprocess
from utils.nms2d import nms_eara
from utils.vis3d import draw_projected_box3d, draw_bev_box3d
from utils.visualize_infer import show_result_keypoints,corner_to_3dboundingbox 
import argparse
from tqdm import tqdm

# TYPE_ID_COLOR = {
#     "VAN" : (0, 0, 255),
#     "CAR": (0, 255, 0),
#     "PD": (255, 0, 0),
#     "Rider": (0, 122, 255),
#     "Three": (0, 122, 122),
#     "TRUCK":(122, 122, 255),
#     "BUS": (122, 122, 0),
#     "SPECIALCAR":(122, 0, 122),
#     "trailerback":(0,0,0),
# }
TYPE_ID_COLOR = {
    "VAN" : (0, 0, 255),
    "CAR": (0, 255, 0),
    "PD": (255, 0, 0),
    "Rider": (0, 122, 255),
    "Three": (0, 122, 122),
    "TRUCK":(255, 0, 0),
    "BUS": (122, 122, 0),
    "SPECIALCAR":(122, 0, 122),
    "trailerback":(0,0,0),
    "STACKER": (114, 114, 114)
}

TYPE_ID_CONVERSION = {
    'CAR': 0,
    'PD': 1,
    'Rider': 2,
    'BUS': 3,
    'TRUCK': 4,
    'Three': 5,
    'trailerback':6,
    'VAN':7,
    'SPECIALCAR': 8,
    "STACKER":9
}
TYPE_ID_INVERSE = {
    0:'CAR',
    1:'PD',
    2:'Rider',
    3:'BUS',
    4:'TRUCK',
    5:'Three',
    6:'trailerback',
    7:'VAN',
    8: "SPECIALCAR",
    9: "STACKER"
}


def draw_2d(image, bbox_2d, thickness=3):
    cv2.rectangle(image, (int(bbox_2d[0]), int(bbox_2d[1])),
                  (int(bbox_2d[2]), int(bbox_2d[3])), (0, 255, 0), thickness)
    return image

def box_to_string2(
                    name,
                    wlh,
                    center,
                    bbox_2d,
                    truncation,
                    occlusion,
                    alpha,
                    yaw,
                    cube,
                    vislines,
                    vid):
        """
        Convert box in KITTI image frame to official label string fromat.
        :param name: KITTI name of the box.
        :param box: Box class in KITTI image frame.
        :param bbox_2d: Optional, 2D bounding box obtained by projected Box into image (xmin, ymin, xmax, ymax).
            Otherwise set to KITTI default.
        :param truncation: Optional truncation, otherwise set to KITTI default.
        :param occlusion: Optional occlusion, otherwise set to KITTI default.
        :param alpha: Optional alpha, otherwise set to KITTI default.
        :return: KITTI string representation of box.
        """
        # Prepare output.
        name += ' '
    #         print('name',name)
    #         print('cube',cube)
        trunc = '{:.2f} '.format(truncation)
        occ = '{:d} '.format(occlusion)
        a = '{:.2f} '.format(alpha)
        bb = '{:.2f} {:.2f} {:.2f} {:.2f} '.format(bbox_2d[0], bbox_2d[1], bbox_2d[2], bbox_2d[3])
        hwl = '{:.2f} {:.2f} {:.2f} '.format(wlh[2], wlh[0], wlh[1])  # height, width, length.
        xyz = '{:.2f} {:.2f} {:.2f} '.format(center[0], center[1], center[2])  # x, y, z.
        y = '{:.2f} '.format(yaw)  # Yaw angle.
        if cube is not None:
            modified_cube = [ [-1,-1] if c is None else c for c in cube]
        else:
            modified_cube = [[-1,-1],[-1,-1],[-1,-1],[-1,-1],[-1,-1],[-1,-1],[-1,-1],[-1,-1]]
    #         print('modified_cube',modified_cube)
        cub = '{:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} '.format(modified_cube[0][0], modified_cube[0][1],modified_cube[1][0],\
        modified_cube[1][1],modified_cube[2][0], modified_cube[2][1],modified_cube[3][0],modified_cube[3][1],modified_cube[4][0], modified_cube[4][1],modified_cube[5][0],\
        modified_cube[5][1],modified_cube[6][0], modified_cube[6][1],modified_cube[7][0],modified_cube[7][1])
        modified_vis = [-1 if v is None else v for v in vislines]
        vis = '{:.2f} {:.2f} {:.2f} {:.2f} '.format(modified_vis[0], modified_vis[1], modified_vis[2], modified_vis[3])
        vid = '{:d}'.format(vid)

        output = name + trunc + occ + a + bb + hwl + xyz + y + cub + vis + vid
    #         print(output)


        return output

# def box_to_string(name,
#                     wlh,
#                     center,
#                     bbox_2d,
#                     truncation,
#                     occlusion,
#                     alpha,
#                     yaw):
#         """
#         Convert box in KITTI image frame to official label string fromat.
#         :param name: KITTI name of the box.
#         :param box: Box class in KITTI image frame.
#         :param bbox_2d: Optional, 2D bounding box obtained by projected Box into image (xmin, ymin, xmax, ymax).
#             Otherwise set to KITTI default.
#         :param truncation: Optional truncation, otherwise set to KITTI default.
#         :param occlusion: Optional occlusion, otherwise set to KITTI default.
#         :param alpha: Optional alpha, otherwise set to KITTI default.
#         :return: KITTI string representation of box.
#         """
#         # Prepare output.
#         name += ' '
#         trunc = '{:.2f} '.format(truncation)
#         occ = '{:d} '.format(occlusion)
#         a = '{:.2f} '.format(alpha)
#         bb = '{:.2f} {:.2f} {:.2f} {:.2f} '.format(bbox_2d[0], bbox_2d[1], bbox_2d[2], bbox_2d[3])
#         hwl = '{:.2} {:.2f} {:.2f} '.format(wlh[2], wlh[0], wlh[1])  # height, width, length.
#         xyz = '{:.2f} {:.2f} {:.2f} '.format(center[0], center[1], center[2])  # x, y, z.
#         y = '{:.2f}'.format(yaw)  # Yaw angle.
        

#         output = name + trunc + occ + a + bb + hwl + xyz + y
       

#         return output

def get_3rd_point(point_a, point_b):
    d = point_a - point_b
    point_c = point_b + np.array([-d[1], d[0]])
    return point_c

def get_transfrom_matrix(center_scale, output_size):
    center, scale = center_scale[0], center_scale[1]
    # todo: further add rot and shift here.
    src_w = scale[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    src_dir = np.array([src_w * -0.5, 0])
    dst_dir = np.array([dst_w * -0.5, 0])

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center
    src[1, :] = center + src_dir
    dst[0, :] = np.array([dst_w * 0.5, dst_h * 0.5])
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2, :] = get_3rd_point(dst[0, :], dst[1, :])

    affine_cv = cv2.getAffineTransform(src,dst)
    #img=  cv2.warpAffine(img, M, (cols, rows))
    get_matrix = trans.estimate_transform("affine", src, dst)
    matrix = get_matrix.params

    return matrix.astype(np.float32),affine_cv

def normalize_img(img, mean,std ):
    #img = img[...,[2, 1, 0]]
    img = np.transpose(img,(2,0,1))

    img = torch.from_numpy(img/255.0).float()
    for t,m,s in zip(img,mean,std):
        t.sub_(m).div_(s)

    return img 

def preprocess_PIL(img,input_width,input_height):
    center = np.array([i / 2 for i in img.size], dtype=np.float32)
    size = np.array([i for i in img.size], dtype=np.float32)
    center_size = [center, size]
    trans_affine,affine_opencv = get_transfrom_matrix(
        center_size,
        [input_width, input_height]
    )
    img =cv2.warpAffine(np.array(img), affine_opencv, (input_width, input_height))
    return img

def box3d_to_corners(locs, dims, roty):
    # 3d bbox template
    h, w, l = dims
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

    # rotation matirx
    R = np.array([[np.cos(roty), 0, np.sin(roty)],
                  [0, 1, 0],
                  [-np.sin(roty), 0, np.cos(roty)]])

    corners3d = np.vstack([x_corners, y_corners, z_corners])  # (3, 8)
    corners3d = np.dot(R, corners3d).T
    corners3d = corners3d + locs

    return corners3d

def preprocess_centersize(img,mean,std,input_width,input_height,center_size=[]):
    bottom_crop = False
    if bottom_crop :
        size = np.array([img.shape[1],input_height*img.shape[1]/input_width], dtype=np.float32)
        center = np.array([img.shape[1]//2,img.shape[0]-size[1]+size[1]//2], dtype=np.float32)
    # shift = np.random.randint(low =0 ,high= (center[1]-size[1]//2)//2, size=1)

    # center[1] = center[1] - shift
    else:
        if center_size==[]:
            
            center = np.array([img.size[0]//2,img.size[1]//2], dtype=np.float32)
            center[1] = center[1] + 504
            size = np.array([3840,1152], dtype=np.float32)
        else:
            center = np.array([img.size[0]//2,img.size[1]//2], dtype=np.float32)
            size = np.array([2560,768], dtype=np.float32)

    """
    resize, horizontal flip, and affine augmentation are performed here.
    since it is complicated to compute heatmap w.r.t transform.
    """
    # print("center : ",center)
    # print("size :",size)
    center_size = [center, size]
    trans_affine,affine_opencv = get_transfrom_matrix(
        center_size,
        [input_width, input_height]
    )
    trans_affine_inv = np.linalg.inv(trans_affine)
    img = img.transform(
        (input_width, input_height),
        method=Image.AFFINE,
        data=trans_affine_inv.flatten()[:6],
        resample=Image.BILINEAR,
    )
    
    img = np.array(img)
    img_numpy = img.copy()
    img= normalize_img(img, mean,std)


    return img ,img_numpy,trans_affine_inv,center_size


def preprocess(img: Image.Image, calib, mean, std, input_size: tuple, img_size:tuple, crop_box:list):
    """
        img_size, input_size: (w, h)
        crop_box: [lx, ly, rx, ry]
    """
    input_width, input_height = input_size
    img_width, img_height = img_size
    roi_width, roi_height = crop_box[2]-crop_box[0], crop_box[3]-crop_box[1]
    
    
    # bottom_crop = False
    # if bottom_crop :
    #     size = np.array([img.size[1],input_height*img.shape[1]/input_width], dtype=np.float32)
    #     center = np.array([img.shape[1]//2,img.shape[0]-size[1]+size[1]//2], dtype=np.float32)
    # else:
        ## image size [1936, 1220]
    center = np.array([(crop_box[2]+crop_box[0]) / 2., (crop_box[3]+crop_box[1]) / 2.], dtype=np.float32)
    size = np.array([roi_width, roi_height], dtype=np.float32)

    """
    resize, horizontal flip, and affine augmentation are performed here.
    since it is complicated to compute heatmap w.r.t transform.
    """
    # print("center : ",center)
    # print("size :",size)
    center_size = [center, size]
    trans_affine, affine_opencv = get_transfrom_matrix(
        center_size,
        [input_width, input_height]
    )
    trans_affine_inv = np.linalg.inv(trans_affine)
    img = img.transform(
        (input_width, input_height),
        method=Image.Transform.AFFINE,
        data=trans_affine_inv.flatten()[:6],
        resample=Image.Resampling.BILINEAR,
    )
    calib.matAndUpdate(trans_affine)
    scale_z = calib.normalize_calib(290)
    # 
    # cv2.imwrite("/home/lipengcheng/MonoFlex-llt/test/output_file/save_affine.png",imgaffine)
    # np.save("/home/lipengcheng/MonoFlex-llt/test/output_file/save_affine.npy",imgaffine)
    # img_t = np.array(img)
    # img_t = cv2.resize(img_t,(int(input_width),int(img.shape[0]*input_width/img.shape[1])),cv2.INTER_LINEAR)
    # img = img_t[img_t.shape[0]//2-input_height//2 :img_t.shape[0]//2+input_height//2,:,:]
    
    #img =cv2.warpAffine(np.array(img), affine_opencv, (input_width, input_height), cv2.INTER_LINEAR)
    img = np.array(img)
    img_numpy = img.copy()
    img= normalize_img(img, mean,std)


    return img ,img_numpy,trans_affine_inv,center_size,calib,scale_z


def preprocess_hh(img,calib,mean,std,input_width,input_height,roi_id,camera_type):
    bottom_crop = False
    center = [0,0]
    if bottom_crop :
        size = np.array([img.shape[1],input_height*img.shape[1]/input_width], dtype=np.float32)
        center = np.array([img.shape[1]//2,img.shape[0]-size[1]+size[1]//2], dtype=np.float32)
    elif roi_id ==0:
        center = np.array([img.size[0] / 2., img.size[1] / 2.], dtype=np.float32)
        size = np.array([2880,1440], dtype=np.float32)
        center[1] = center[1] + 210
    elif roi_id ==1:
        if camera_type =='rl' or camera_type =='fr':
            size = np.array([1600, 800], dtype=np.float32)
            center[1] = 930
            center[0] = 2079

        if camera_type =='fl' or camera_type =='rr':
            size = np.array([1600, 800], dtype=np.float32)
            center[1] = 930
            center[0] = 799

    """
    resize, horizontal flip, and affine augmentation are performed here.
    since it is complicated to compute heatmap w.r.t transform.
    """
    # print("center : ",center)
    # print("size :",size)
    center_size = [center, size]
    trans_affine,affine_opencv = get_transfrom_matrix(
        center_size,
        [input_width, input_height]
    )
    trans_affine_inv = np.linalg.inv(trans_affine)
    img = img.transform(
        (input_width, input_height),
        method=Image.AFFINE,
        data=trans_affine_inv.flatten()[:6],
        resample=Image.BILINEAR,
    )
    calib.matAndUpdate(trans_affine)
    scale_z = calib.normalize_calib(290)
    # 
    # cv2.imwrite("/home/lipengcheng/MonoFlex-llt/test/output_file/save_affine.png",imgaffine)
    # np.save("/home/lipengcheng/MonoFlex-llt/test/output_file/save_affine.npy",imgaffine)
    # img_t = np.array(img)
    # img_t = cv2.resize(img_t,(int(input_width),int(img.shape[0]*input_width/img.shape[1])),cv2.INTER_LINEAR)
    # img = img_t[img_t.shape[0]//2-input_height//2 :img_t.shape[0]//2+input_height//2,:,:]
    
    #img =cv2.warpAffine(np.array(img), affine_opencv, (input_width, input_height), cv2.INTER_LINEAR)
    img = np.array(img)
    img_numpy = img.copy()
    img= normalize_img(img, mean,std)


    return img ,img_numpy,trans_affine_inv,center_size,calib,scale_z

# def getflops(model,input_w,input_h):
#     from torchstat import stat
#     from thop import profile
#     from thop import clever_format 
#     #stat(model, (3, 512, 512))
#     input_img = torch.randn(1, 3, input_h, input_w)
#     thop_flops, thop_params = profile(model, inputs=(input_img, ))
#     thop_flops, thop_params = clever_format([thop_flops, thop_params], "%.3f") 
#     print(thop_flops,thop_params)


def affine_transform(point, matrix):

    point = point.reshape(-1, 2)
    point_exd = np.concatenate((point, np.ones((point.shape[0], 1))), axis=1)

    new_point = np.matmul(point_exd, matrix.T)

    return new_point[:, :2].squeeze()

def update2Dbox2OriIm(boxes,trans_affine,input_width,input_height):
    for i, box2d in enumerate(boxes):
        boxes[i,:2] = affine_transform(box2d[:2], trans_affine)
        boxes[i,2:] = affine_transform(box2d[2:], trans_affine)
        boxes[i,[0, 2]] = boxes[i,[0, 2]].clip(0, input_width - 1)
        boxes[i,[1, 3]] = boxes[i,[1, 3]].clip(0, input_height - 1)
    return boxes 

def update2DKeyPoint2OriIm(boxes,trans_affine,input_width,input_height):
    for i, box2d in enumerate(boxes):
        boxes[i,:2] = affine_transform(box2d[:2], trans_affine)
        boxes[i,0] = boxes[i,0].clip(0, input_width - 1)
        boxes[i,1] = boxes[i,1].clip(0, input_height - 1)
    return boxes 

def setup_test_args(parser:argparse.ArgumentParser):
    """
        output_dir = "./output/test/" # 输出路径
        imageset_txt_path ="/home/utopilot/workspace/infer/eval_txt/grad.txt" # 图片文件路径
        model_path = "./checkpoints/model_checkpoint_100_small_obj.pth" # pth路径
        calib_folder = None # xml路径
        input_width, input_height = 640, 384 # 模型输入尺寸
        roi_width, roi_height = 1920, 1152 # 图片ROI尺寸
        img_width, img_height = 1936, 1220 # 图片原始尺寸
    """
    parser.add_argument("--vis_video", action="store_true", help="Generate video.")
    parser.add_argument("--vis_25d", action="store_true", help="Visualize 2.5d results.")
    parser.add_argument("--vis_3d", action="store_true", help="Visualize 3d results.")
    parser.add_argument("--output_dir", type=str, default="./output/", help="Output directory.")
    parser.add_argument("--image_txt", type=str, required=True, help="txt file including paths of all images to test.")
    parser.add_argument("--model_path", type=str, required=True, help="Model to test.")
    parser.add_argument("--calib_path", type=str, default=None, help="Path of Calibration files.")
    parser.add_argument("--input_size", type=int, default=[640, 384], nargs=2, help="Model input size, [w, h]")
    parser.add_argument("--image_size", type=int, default=[1936, 1220], nargs=2, help="Original Image size, [w, h]")
    # parser.add_argument("--roi_size", type=int, default=[1920, 1152], nargs='+')
    parser.add_argument("--alpha", type=float, default=0.3, help="truncation alpha threshold")
    parser.add_argument("--thres", type=float, default=0.29, help="det_threshold")
    parser.add_argument("--crop", type=int, default=[8, 28, 1928, 1220], nargs=4, help="Crop box diagonal coordinates [x1, y1, x2, y2]")
    parser.add_argument("--output_height", type=int, default=800, help="height of result visualization")
    return parser

def setup(args):
    test_config={}
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    cfg.DATALOADER.NUM_WORKERS = args.num_work
    cfg.TEST.EVAL_DIS_IOUS = args.eval_iou
    cfg.TEST.EVAL_DEPTH = args.eval_depth 
    
    if args.vis_thre > 0:
        cfg.TEST.VISUALIZE_THRESHOLD = args.vis_thre 
    
    if args.output is not None:
        cfg.OUTPUT_DIR = args.output

    if args.test:
        cfg.DATASETS.TEST_SPLIT = 'test'
        cfg.DATASETS.TEST = ("kitti_test",)

    cfg.START_TIME = datetime.datetime.strftime(datetime.datetime.now(), '%m-%d %H:%M:%S')
    default_setup(cfg, args)

    return cfg, test_config

class KeypointDetector_v2(nn.Module):
    '''
    Generalized structure for keypoint based object detector.
    main parts:
    - backbone
    - heads
    '''

    def __init__(self, cfg):
        super(KeypointDetector_v2, self).__init__()
        self.backbone = build_backbone(cfg)
        self.heads = make_predictor(cfg, self.backbone.out_channels)

    def forward(self, images):
#         img_comp = torch.load("/workspace/UserData/lpc/Mono_0718/Mono_simple_2d/output/add_cone_fall/onnx_infer/img_input.pt")
#         img_diff = img_comp - images
        features = self.backbone(images)
#         feature_comp = torch.load("/workspace/UserData/lpc/Mono_0718/Mono_simple_2d/output/add_cone_fall/onnx_infer/feature.pt")
#         feature_diff = feature_comp - features
        
        output_cls,  output_regs= self.heads(features)
#         output_cls_cmp = torch.load("/workspace/UserData/lpc/Mono_0718/Mono_simple_2d/output/add_cone_fall/onnx_infer/output_cls.pt")
#         output_cls_diff = output_cls_cmp -output_cls
#         output_regs_cmp = torch.load("/workspace/UserData/lpc/Mono_0718/Mono_simple_2d/output/add_cone_fall/onnx_infer/output_regs.pt")
#         output_regs_diff = output_regs_cmp -output_regs
#         pdb.set_trace()
        return output_cls,  output_regs

def get_imgs_path(src_dir):
    imgs_path_list = []
    for (root, dirs, files) in os.walk(src_dir):
        if (len(files) != 0):
            for cur_file in files:
                cur_file_path = os.path.join(root, cur_file)
                file_name_prefix, file_name_suffix = os.path.splitext(cur_file)
                if not file_name_suffix.lower() in (".jpg", ".jpeg", ".png"):
                    continue
                else:
                    imgs_path_list.append(cur_file_path)
    return imgs_path_list

# def nms3d(x, nms_th):
#     # x:[p, z, w, h, d]
#     if len(x) == 0:
#         return x
#     sizescore = np.zeros([len(x['name'])])
#     for i in np.arange(len(x['name'])):
#         sizescore[i] = x['dimensions'][i,0]*x['dimensions'][i,1]*x['dimensions'][i,2]
#     sortid = np.argsort(-sizescore)
#     target = np.ones([sortid.size])
    
#     for i in sortid:
        
#         ref_boxes = np.concatenate([x['location'][i], x['dimensions'][i], x['rotation_y'][i][..., np.newaxis]], axis=0)
#         ref_boxes = ref_boxes.reshape(1,-1)
       
#         for j in sortid:
#             if j>i and target[j]==1:
#                 comp_boxes = np.concatenate([x['location'][j], x['dimensions'][j], x['rotation_y'][j][..., np.newaxis]], axis=0)
#                 comp_boxes = comp_boxes.reshape(1,-1)
#                 overlap_part = d3_box_overlap(ref_boxes, comp_boxes).astype(
#                     np.float64)
#                 if overlap_part > nms_th:
#                     target[j] = 0
#                     print(overlap_part)
                    
                
#         #target[i]=0                 
#     saveid = np.where(target==1.0)
#     return saveid

# def read_picture():
#     path = '/home/lipengcheng/results/yizhuang/eval3/neolix_1/img/'
#     file_list = os.listdir(path)

#     fps = 2 # 视频每秒2帧
#     height = 1920
#     weight = 1280
#     size = (int(height), int(weight))  # 需要转为视频的图片的尺寸
#     return [path, fps, size, file_list]



# def write_video(savefolder, fps, size, file_list):
#     #path, fps, size, file_list = read_picture()
#     # AVI格式编码输出 XVID
#     four_cc = cv2.VideoWriter_fourcc(*'XVID')
#     save_path = savefolder + '/' + '%s.avi' % str(uuid.uuid1())
#     video_writer = cv2.VideoWriter(save_path, four_cc, float(fps), size)
#     # 视频保存在当前目录下
#     for item in file_list:
#         if item.endswith('.jpg') or item.endswith('.png'):
#             # 找到路径中所有后缀名为.png的文件，可以更换为.jpg或其它
#             img = cv2.imread(item)
#             re_pics = cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)  # 定尺寸
#             if len(re_pics):
#                 video_writer.write(re_pics)

#     video_writer.release()



def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

class Object3d(object):
    """ 2d object label """

    def __init__(self):
        data = np.zeros([15])
        
        # extract label, truncation, occlusion
        self.type = data[0]  # 'Car', 'Pedestrian', ...
        self.truncation = data[1]  # truncated pixel ratio [0..1]
        self.occlusion = int(
            data[2]
        )  # 0=visible, 1=partly occluded, 2=fully occluded, 3=unknown

        # extract 2d bounding box in 0-based coordinates
        self.xmin = data[4]  # left
        self.ymin = data[5]  # top
        self.xmax = data[6]  # right
        self.ymax = data[7]  # bottom
        self.box2d = np.array([self.xmin, self.ymin, self.xmax, self.ymax], dtype=np.float32)

        # extract 3d bounding box information
        self.h = data[8]  # box height
        self.w = data[9]  # box width
        self.l = data[10]  # box length (in meters)
        self.t = np.array((float(data[11]), float(data[12]), float(data[13])), dtype=np.float32)  # location (x,y,z) in camera coord.

        self.ry = data[14]  # yaw angle (around Y-axis in camera coordinates) [-pi..pi]
        self.real_alpha = data[3]
        self.ray = 0
        self.alpha = 0
        self.img = None
        self.center_type = 0
        self.keypoint_down= np.array([data[15],data[16],data[17],data[18],data[19],
                            data[20],data[21],data[22]]) if len(data) ==36 else np.array([0,0,0,0,0,0,0,0])
        self.keypoint_up= np.array([data[23],data[24],data[25],data[26],data[27],
                            data[28],data[29],data[30]]) if len(data) ==36 else np.array([0,0,0,0,0,0,0,0])

        self.ytop_vis =  data[31] if len(data) ==36 else 0
        self.ybuttom_vis = data[32] if len(data) ==36 else 0
        self.xlft_vis = data[33] if len(data) ==36 else 0
        self.xrght_vis = data[34] if len(data) ==36 else 0

        self.vid = data[35] if len(data) ==36 else 0
    
    def generate_corners3d(self):
        """
        generate corners3d representation for this object
        :return corners_3d: (8, 3) corners of box3d in camera coord
        """
        l, h, w = self.l, self.h, self.w
        x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
        y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
        z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

        R = np.array([[np.cos(self.ry), 0, np.sin(self.ry)],
                      [0, 1, 0],
                      [-np.sin(self.ry), 0, np.cos(self.ry)]])

        corners3d = np.vstack([x_corners, y_corners, z_corners])  # (3, 8)
        corners3d = np.dot(R, corners3d).T
        corners3d = corners3d + self.t

        return corners3d
## BEV MAP config
metric_width, metric_height, worldsize = 800, 800, 160 
def creat_bev_map():
    polar_step_size_meters = 10
    pixels_per_meter = metric_height//worldsize # 每米5个pixel 
    center_pixel = ( metric_width//2, metric_height)
    img_bev = np.ones((metric_width, metric_height, 3), dtype=np.uint8) * 230
        # # Draw metric polar grid
    for i in range(1, int(metric_width/(polar_step_size_meters*pixels_per_meter))):
        cv2.circle(img_bev, center_pixel, int(i * polar_step_size_meters*pixels_per_meter ),
                (50, 50, 50), 2)
        cv2.line(img_bev, (0, center_pixel[1]-int(i * polar_step_size_meters*pixels_per_meter )), (metric_width, center_pixel[1]-int(i * polar_step_size_meters*pixels_per_meter )), (200, 200, 200))
        cv2.line(img_bev, (abs(int(i * polar_step_size_meters*pixels_per_meter )),0), (abs(int(i * polar_step_size_meters*pixels_per_meter )),metric_height), (200, 200, 200))
    return img_bev

# def update_calib(calib):
#     if 'rosbag2_2022_08_16-10_51_33_trim' in line:
#         # print(calib.P)
#         calib.P = np.array([[1966.6187, 0, 1896.6259, 0, 0, 1965.1837, 1075.0922,0, 0, 0, 1, 0]]).reshape(3,4)
#         calib.c_u = calib.P[0, 2]
#         calib.c_v = calib.P[1, 2]
#         calib.f_u = calib.P[0, 0]
#         calib.f_v = calib.P[1, 1]
#         calib.b_x = calib.P[0, 3] / (-calib.f_u)  # relative
#         calib.b_y = calib.P[1, 3] / (-calib.f_v)
#         # print(calib.P)
#         # pdb.set_trace()
        
#     if 'rosbag2_2022_08_30-16_04_39-1' in line:
#         # print(calib.P)
#         calib.P = np.array([[1945.9661, 0, 1868.6702, 0, 0, 1946.149, 1104.4604,0, 0, 0, 1, 0]]).reshape(3,4)
#         calib.c_u = calib.P[0, 2]
#         calib.c_v = calib.P[1, 2]
#         calib.f_u = calib.P[0, 0]
#         calib.f_v = calib.P[1, 1]
#         calib.b_x = calib.P[0, 3] / (-calib.f_u)  # relative
#         calib.b_y = calib.P[1, 3] / (-calib.f_v)

#     if 'rosbag_2022_10_19_15_34_10' in line:
#         # print(calib.P)
#         calib.P = np.array([[1958.7679, 0, 1956.5509, 0, 0, 1959.7946, 1123.9296,0, 0, 0, 1, 0]]).reshape(3,4)
#         calib.c_u = calib.P[0, 2]
#         calib.c_v = calib.P[1, 2]
#         calib.f_u = calib.P[0, 0]
#         calib.f_v = calib.P[1, 1]
#         calib.b_x = calib.P[0, 3] / (-calib.f_u)  # relative
#         calib.b_y = calib.P[1, 3] / (-calib.f_v)
#     return calib




if __name__ == "__main__":
    ## Parse args
    parser = default_argument_parser()
    parser = setup_test_args(parser)
    args = parser.parse_args()
    
    ###################CONFIGS########################
    output_dir = args.output_dir # 输出路径
    imageset_txt_path =args.image_txt # 图片文件路径
    model_path = args.model_path # pth路径
    calib_folder = args.calib_path # xml路径
    input_width, input_height = args.input_size[0], args.input_size[1] # 模型输入尺寸
    # roi_width, roi_height = args.roi_size[0], args.roi_size[1] # 图片ROI尺寸
    crop_box = args.crop # cropbox 对角线顶点 [x1, y1, x2, y2]
    img_width, img_height = args.image_size[0], args.image_size[1] # 图片原始尺寸
    trunc_alpha = args.alpha
    det_thres = args.thres
    vis_25d =args.vis_25d
    vis_3d =args.vis_3d
    vis_video = args.vis_video
    output_height = args.output_height
    # output_dir = "./output/test/" # 输出路径
    # imageset_txt_path ="/home/utopilot/workspace/infer/eval_txt/grad.txt" # 图片文件路径
    # model_path = "    " # pth路径
    # calib_folder = None # xml路径
    # input_width, input_height = 640, 384 # 模型输入尺寸
    # roi_width, roi_height = 1920, 1152 # 图片ROI尺寸
    # img_width, img_height = 1936, 1220 # 图片原始尺寸
    
    # vis_25d =True
    # vis_3d =False
    # vis_video = True
    ################################################
    args.ckpt = model_path
    cfg, _ = setup(args)
    cfg.OUTPUT_DIR = output_dir
    # cfg.MODEL.BACKBONE.CONV_BODY = "Vggx2SmallNet"
    # cfg.MODEL.HEAD.NUM_CHANNEL = 128
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    ## Load model
    model = KeypointDetector_v2(cfg).cuda()
    checkpointer = DetectronCheckpointer(cfg, model, save_dir=cfg.OUTPUT_DIR)
    _ = checkpointer.load(args.ckpt, use_latest=False)
    model.eval()
    postproc = postprocess(input_width,input_height, det_thres).cuda()

    ID_TYPE_CONVERSION = {k : v for v, k in TYPE_ID_CONVERSION.items()}
    pred_color = (0, 0, 255)
    

    roi_box =crop_box # cropbox四个顶点
    
    save_dir_3d= os.path.join(cfg.OUTPUT_DIR,'3D')
    save_dir_25d= os.path.join(cfg.OUTPUT_DIR,'25D')
    save_dir_txt= os.path.join(cfg.OUTPUT_DIR,'det_txt')
    os.makedirs(save_dir_3d,exist_ok =True)
    os.makedirs(save_dir_25d,exist_ok =True)
    os.makedirs(save_dir_txt, exist_ok =True)
    # os.makedirs(os.path.join(save_dir,'2dbox'),exist_ok =True)
    
    
    ## Generate video
    if vis_video:
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        if vis_3d:
            out3D_size = (int(img_width*output_height/img_height) + output_height,output_height)
            out3D = cv2.VideoWriter(os.path.join(cfg.OUTPUT_DIR,'3D.avi'), fourcc, 10.0, out3D_size)
        if vis_25d:
            out25D_size = (int(img_width*output_height/img_height), output_height)
            out25D = cv2.VideoWriter(os.path.join(cfg.OUTPUT_DIR,'25D.avi'), fourcc, 10.0, out25D_size) 
            

    
    with open(imageset_txt_path, "r") as f:
        lines = f.readlines()
    lines.sort()
        
    print()
    print("[TEST INFO]")
    print(" {:<12}:".format("output_dir"), output_dir)
    print(" {:<12}:".format("image_txt"), imageset_txt_path)
    print(" {:<12}:".format("image_num"), len(lines))
    print(" {:<12}:".format("model"), model_path)
    print(" {:<12}:".format("input_size"), args.input_size)
    print(" {:<12}:".format("image_size"), args.image_size)
    print(" {:<12}:".format("crop_box"), roi_box)
    print(" {:<12}:".format("vis_25d"), args.vis_25d)
    print(" {:<12}:".format("vis_3d"), args.vis_3d)
    print(" {:<12}:".format("vis_video"), args.vis_video)
    print()
    
    ## Test phrase
    with torch.no_grad():
        idx= 0
        for line in tqdm(lines, unit='img'):
            # if idx % 1000 == 0:
            #     print(idx)
            idx = idx+1
            
            line =line.strip()
            # print("IMG:", line)
            basename = os.path.basename(line).split('.')[0]
            impath = line
            name = impath.split('/')[-1]
            savename = name
            
            ## TODO: 因为xml标注不一定有，需要自行定义对应Calib文件路径
            try:
                # dataset = "/home/utopilot/workspace/infer/data_1209/aiv_test/"
                # calib_folder = line.split('/')[-2]
                calib_file = os.path.join(calib_folder, basename + ".xml")
                calib = Calibration(calib_file)
            except:
                # 若无则随便传入一个
                calib_file = "/home/utopilot/workspace/infer/HH_3D_SIDE/Label0000004/truck46_rr_20220808_155831_850_108.xml"
                calib = Calibration(calib_file)
            
            
            img = Image.open(impath).convert('RGB')
            img_vis = np.array(img).copy()
            w,h  =img.size
            #print(w,h)
            pixel_mean = torch.from_numpy(np.array([0.485, 0.456, 0.406]))
            pixel_std = torch.from_numpy(np.array([0.229, 0.224, 0.225]))
            
            img, img_numpy, trans_affine_inv, center_size, calib, scale_z= preprocess(img,calib,pixel_mean,pixel_std, (input_width,input_height), (img_width, img_height), crop_box)
            img =img.unsqueeze(0).to('cuda')    
            
            output_cls,  output_regs =  model(img)
            clses,alphas,rotys, box2d, dimensions,scores,locations,keypoint, pred_keypoint_visible,center_proj, center_type = postproc(output_cls, output_regs,calib)
            
            if clses is None:
                # print("no results:",name)
                img_vis = draw_2d(img_vis, [8,1220-1152,1928,1220]) ## TODO: roi box 
                img_vis = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)
                img_vis = cv2.resize(img_vis,(int(img_vis.shape[1]*800/img_vis.shape[0]),800)) 
                cv2.imwrite(os.path.join(save_dir_25d,name) ,img_vis)
                continue
            
            clses,box2d,scores = clses.data.cpu().numpy(),box2d.data.cpu().numpy(),scores.data.cpu().numpy()
            alphas,rotys,dimensions,locations,keypoint,keypoint_visible,center_proj,center_type = alphas.data.cpu().numpy(),rotys.data.cpu().numpy(),dimensions.data.cpu().numpy(),locations.data.cpu().numpy(),keypoint.data.cpu().numpy(),\
                pred_keypoint_visible.data.cpu().numpy(),center_proj.data.cpu().numpy(), center_type.data.cpu().numpy()
            
            ## 映射回原图
            box2d_ori = update2Dbox2OriIm(box2d,trans_affine_inv,w,h)
            center_proj = center_proj.reshape(-1,2)
            center_proj = update2DKeyPoint2OriIm(center_proj,trans_affine_inv,w,h)
            center_proj = center_proj.astype(int)
            keypoint = keypoint.reshape(-1,2)
            keypoint = update2DKeyPoint2OriIm(keypoint,trans_affine_inv,w,h)
            keypoint = keypoint.reshape(-1,4,2)
            locations[:,2] = locations[:,2]/scale_z

            ## 筛除重叠BBOX
            dets = np.concatenate((box2d_ori,scores.reshape(-1,1)),axis=1)
            dets = np.concatenate((dets,clses.reshape(-1,1)),axis=1)
            keep = nms_eara(dets,0.85) 
            
            calib = Calibration(calib_file)
            # calib = update_calib(calib)
            
            img_vis = np.array(img_vis[...,::-1])
            if vis_25d:
                img_25d =  img_vis.copy()
                img_25d = draw_2d(img_25d, roi_box)
            if vis_3d:
                img_bev = creat_bev_map()
                img_3d = img_vis.copy()
                img_3d = draw_2d(img_3d, roi_box)
            
            ## 生成要保留的object
            objs_extra = [Object3d() for _ in np.arange(box2d_ori.shape[0])]
            save_obj = []
            for k,obj in enumerate(objs_extra):
                if k not in keep:
                    continue
                obj.t = locations[k]
                obj.w = dimensions[k][1]
                obj.l = dimensions[k][2]
                obj.h = dimensions[k][0]
                obj.ry = rotys[k][0]
                obj.box2d = box2d_ori[k]
                obj.xmin = box2d_ori[k][0]
                obj.ymin = box2d_ori[k][1]
                obj.xmax = box2d_ori[k][2]
                obj.ymax = box2d_ori[k][3]
                obj.type = TYPE_ID_INVERSE[int(clses[k])]
                obj.keypoint_down= np.array([[kp[0],kp[1]] if vis else [-1,-1]for kp,vis in zip(keypoint[k],keypoint_visible[k])])
                obj.alpha = scores[k][0]
                obj.center_type = center_type[k]
                margin = 75 ## 判定truncation的margin
                obj.truncation = obj.xmax >crop_box[2] - margin or obj.xmin < crop_box[0]
                
                
                # PD Rider 阈值不同
                if not obj.truncation:
                    if obj.type == "PD" or obj.type == "Rider":
                        alpha_thres = 0.28
                    else:
                        alpha_thres = trunc_alpha
                    # print(obj.type, obj.alpha, obj.truncation)
                    if obj.alpha < alpha_thres:
                        # print("OMIT", obj.type, obj.alpha, obj.truncation)
                        continue
                    
                # if obj.l >20 or obj.t[2]>160:
                #     continue
                
                # 2.2 根据长宽比筛选CAR TRUCK
                if obj.type == ["TRUCK", "trailerback"]:
                    length_x = abs(obj.xmax-obj.xmin)
                    length_y = abs(obj.ymax-obj.ymin)
                    if length_y / length_x > 4.0:
                        continue
                if obj.type in ["CAR", "VAN", "SPECIALCAR", "Three"]:
                    length_x = abs(obj.xmax-obj.xmin)
                    length_y = abs(obj.ymax-obj.ymin)
                    if length_y / length_x > 3.0:
                        continue
                
                # box大小筛选
                if obj.type in ["CAR","Three","BUS","TRUCK","trailerback","VAN"] and (obj.ymax-obj.ymin)*(obj.xmax-obj.xmin)<50:
                    continue
                if obj.type in ["PD","Rider"] and (obj.xmax-obj.xmin)<10:
                    continue
                
                save_obj.append(obj) 
                
                ## Visualization
                if vis_3d:
                    ## 3D box可视化
                    if obj.type in [ "CAR","PD","Rider","Three","BUS","TRUCK","TRUCKHEAD","VAN","SPECIALCAR", "STACKER"]:
                        # corners_3d_det = obj.generate_corners3d()
                        corners_3d_det = corner_to_3dboundingbox(obj.t,[obj.h, obj.w, obj.l], obj.ry,obj.center_type)
                        img_bev = draw_bev_box3d(img_bev, corners_3d_det[np.newaxis, :], 0, thickness=2, color=TYPE_ID_COLOR[obj.type], scores=None,world_size=worldsize,out_size=metric_width)
                        corners_2d_det, depth = calib.project_rect_to_image(corners_3d_det)
                        img_3d = draw_projected_box3d(img_3d, corners_2d_det, 0, cls=obj.type, color=TYPE_ID_COLOR[obj.type], draw_orientation=False,draw_corner=False)
                if vis_25d:
                    ## 2.5D可视化
                    if obj.type in [ "CAR","PD","Rider","Three","BUS","TRUCK","TRUCKHEAD","VAN","SPECIALCAR", "STACKER"]:
                        if obj.type in [ "CAR","Three","BUS","TRUCK","TRUCKHEAD","VAN","SPECIALCAR", "STACKER"]:
                            img_25d = show_result_keypoints(img_25d,keypoint_visible[k],keypoint[k],obj.box2d) # 绘制keypoints
                        
                        box_center = (obj.box2d[0:2] + obj.box2d[2:4]) / 2
                        box_center = box_center.astype(int)
                        cv2.circle(img_25d, tuple(center_proj[k]),2,(0,0,255),2) # box中心圆圈
                        cv2.putText(img_25d, '{}{}'.format(k, obj.type), tuple(obj.box2d[0:2].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 
                                            1, (255, 100, 0), 2, cv2.LINE_AA) # box左上角类别
                        cv2.putText(img_25d, '{:.3f}'.format(obj.alpha), tuple(center_proj[k]), cv2.FONT_HERSHEY_SIMPLEX, 
                                            1, (255, 0, 0), 2, cv2.LINE_AA) # 置信度
                        cv2.rectangle(img_25d, tuple(obj.box2d[0:2].astype(int)), tuple(obj.box2d[2:4].astype(int)), (0, 0, 255), thickness = 2) # bbox
            
            if vis_25d:
                img_25d = cv2.resize(img_25d,(int(img_25d.shape[1]*800/img_25d.shape[0]),800))
                cv2.imwrite(os.path.join(save_dir_25d,savename) ,img_25d)
                if vis_video:
                    out25D.write(img_25d)
            if vis_3d:
                
                img_3d = cv2.resize(img_3d,(int(img_3d.shape[1]*800/img_3d.shape[0]),800))          
                img_array = np.concatenate((img_bev, img_3d), axis=1)
                cv2.imwrite(os.path.join(save_dir_3d,savename) ,img_array) ## 3d box
                if vis_video:
                    out3D.write(img_array)

            ## 输出txt
            with open(os.path.join(save_dir_txt,name.replace('jpg','txt').replace('png','txt')),'w') as f:
                for i,obj in enumerate(save_obj):
                    # print(obj.alpha)
                    cube = np.zeros([8,2])-1
                    cube[0:4,0:2] = obj.keypoint_down
                    cube = cube.tolist()
                    visline =[-1,-1,-1,-1]
                    v_id = 0
                    output = box_to_string2(obj.type,[obj.w,obj.l,obj.h],[obj.t[0],obj.t[1],obj.t[2]],obj.box2d,float(obj.truncation+0),0,obj.alpha,obj.ry,cube,visline,v_id)
                    f.write(output+'\n')
            f.close()
    if vis_video:
        if vis_3d:
            out3D.release()
        if vis_25d:
            out25D.release()
                    


  
