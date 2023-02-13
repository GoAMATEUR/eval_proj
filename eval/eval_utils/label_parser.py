from PIL import Image
import sys
sys.path.append('/home/utopilot/workspace/infer/mono_infer_vgg/')
import numpy as np
import cv2
import os
from eval.eval_utils.eval_vis_two_box import vis_two_box
from eval.eval_utils.eval_kitti_utils import Object3d
from skimage import transform as trans
# from utils.visualize_infer import show_result_keypoints
from eval.eval_utils.eval_kitti_utils import Calibration,read_label
import copy
from typing import List, Dict, Set, Tuple

VERTICAL_COLORS = [(255, 255, 255), (0, 125, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]

def show_result_keypoints(img,
                vedges_flag,
                vedges,bbox,
                show=True,
                out_file=None,POST_PROCESS=False, vis_invis = False):
    """Visualize the detection results on the image.

    Args:
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        class_names (list[str] or tuple[str]): A list of class names.
        score_thr (float): The threshold to visualize the bboxes and masks.
        wait_time (int): Value of waitKey param.
        show (bool, optional): Whether to show the image with opencv or not.
        out_file (str, optional): If specified, the visualization result will
            be written to the out file instead of shown in a window.

    Returns:
        np.ndarray or None: If neither `show` nor `out_file` is specified, the
            visualized image is returned, otherwise None is returned.
    """
    thickness=1
    bbox_int = bbox.astype(np.int32)
    left_top = (bbox_int[0], bbox_int[1])
    right_bottom = (bbox_int[2], bbox_int[3])
    
    vedges = vedges.reshape(-1)
    direction_flag = 0
    lf_x, lf_y = vedges[:2]
    lb_x, lb_y = vedges[2:4]
    rf_x, rf_y = vedges[4:6]
    rb_x, rb_y = vedges[6:8]

    lf_x=int(lf_x)
    lf_y=int(lf_y)
    lb_x=int(lb_x)
    lb_y=int(lb_y)
    rf_x=int(rf_x)
    rf_y=int(rf_y)
    rb_x=int(rb_x)
    rb_y=int(rb_y)

    
    
    for idx, vflag in enumerate(vedges_flag):
        vflag = vflag+0
        direction_flag += 1 << (vflag * (idx + 1)) if vflag > 0 else 0

    if direction_flag == 2:
        # only left_font is detected
        if vis_invis:
            cv2.circle(img, (lf_x, lf_y),4,VERTICAL_COLORS[0],4)

    elif direction_flag == 4:
        # only left_behind is detected
        if vis_invis:
            
            cv2.circle(img, (lb_x, lb_y),4,VERTICAL_COLORS[1],4)

    elif direction_flag == 6:
        # left_front and left_behind is detected

        if POST_PROCESS:
            lf_x = left_top[0]
            lb_x = right_bottom[0]
        cv2.line(img, (lf_x, left_top[1]), (lf_x, lf_y), VERTICAL_COLORS[0], thickness=thickness)
        cv2.line(img, (lb_x, left_top[1]), (lb_x, lb_y), VERTICAL_COLORS[1], thickness=thickness)
        cv2.line(img, (lf_x, lf_y), (lb_x, lb_y), VERTICAL_COLORS[-1], thickness=thickness)
    elif direction_flag == 8:
        # only right front is detected
        if vis_invis:
            cv2.circle(img, (rf_x, rf_y),4,VERTICAL_COLORS[2],4)
        pass
    elif direction_flag == 10:
        # left_front and right front is detected
        if POST_PROCESS:
            lf_x = right_bottom[0]
            rf_x = left_top[0]
        cv2.line(img, (lf_x, left_top[1]), (lf_x, lf_y), VERTICAL_COLORS[0], thickness=thickness)
        cv2.line(img, (rf_x, left_top[1]), (rf_x, rf_y), VERTICAL_COLORS[2], thickness=thickness)
        cv2.line(img, (lf_x, lf_y), (rf_x, rf_y), VERTICAL_COLORS[-1], thickness=thickness)
    elif direction_flag == 12:
        # left_behind, right_front is detected
        if vis_invis:
            cv2.circle(img, (lb_x, lb_y),4, VERTICAL_COLORS[1],4)
            cv2.circle(img, (rf_x, rf_y),4, VERTICAL_COLORS[2],4)
            pass
    elif direction_flag == 14:
        # left_front, left_behind, right_front is detected
        if POST_PROCESS:
            lb_x = right_bottom[0]
            rf_x = left_top[0]

        cv2.line(img, (lf_x, left_top[1]), (lf_x, lf_y), VERTICAL_COLORS[0], thickness=thickness)
        cv2.line(img, (lb_x, left_top[1]), (lb_x, lb_y), VERTICAL_COLORS[1], thickness=thickness)
        cv2.line(img, (rf_x, left_top[1]), (rf_x, rf_y), VERTICAL_COLORS[2], thickness=thickness)
        cv2.line(img, (lf_x, lf_y), (lb_x, lb_y), VERTICAL_COLORS[-1], thickness=thickness)
    elif direction_flag == 16:
        if vis_invis:
            cv2.circle(img, (rb_x, rb_y),4, VERTICAL_COLORS[3],4)
        # right_behind is detected
        pass
    elif direction_flag == 18:
        if vis_invis:
            cv2.circle(img, (lf_x, lf_y),4,VERTICAL_COLORS[0],4)
            cv2.circle(img, (rb_x, rb_y),4, VERTICAL_COLORS[3],4)
        # left_front, right_behind
        pass
    elif direction_flag == 20:
        # left_behind, right_behind
        if POST_PROCESS:
            lb_x = left_top[0]
            rb_x = right_bottom[0]

        cv2.line(img, (lb_x, left_top[1]), (lb_x, lb_y), VERTICAL_COLORS[1], thickness=thickness)
        cv2.line(img, (rb_x, left_top[1]), (rb_x, rb_y), VERTICAL_COLORS[3], thickness=thickness)
        cv2.line(img, (lb_x, lb_y), (rb_x, rb_y), VERTICAL_COLORS[-1], thickness=thickness)
    elif direction_flag == 22:
        # left_front, left_behind, right_behind
        if POST_PROCESS:
            lf_x = left_top[0]
            rb_x = right_bottom[0]

        cv2.line(img, (lf_x, left_top[1]), (lf_x, lf_y), VERTICAL_COLORS[0], thickness=thickness)
        cv2.line(img, (lb_x, left_top[1]), (lb_x, lb_y), VERTICAL_COLORS[1], thickness=thickness)
        cv2.line(img, (rb_x, left_top[1]), (rb_x, rb_y), VERTICAL_COLORS[3], thickness=thickness)
        cv2.line(img, (lf_x, lf_y), (lb_x, lb_y), VERTICAL_COLORS[-1], thickness=thickness)

    elif direction_flag == 24:
        # right_front right_behind
        if POST_PROCESS:
            rf_x = right_bottom[0]
            rb_x = left_top[0]

        cv2.line(img, (rf_x, left_top[1]), (rf_x, rf_y), VERTICAL_COLORS[2], thickness=thickness)
        cv2.line(img, (rb_x, left_top[1]), (rb_x, rb_y), VERTICAL_COLORS[3], thickness=thickness)
        cv2.line(img, (rb_x, rb_y), (rf_x, rf_y), VERTICAL_COLORS[-1], thickness=thickness)
    elif direction_flag == 26:
        # left_front, right_front, right_behind
        if POST_PROCESS:
            lf_x = right_bottom[0]
            rb_x = left_top[0]

        cv2.line(img, (lf_x, left_top[1]), (lf_x, lf_y), VERTICAL_COLORS[0], thickness=thickness)
        cv2.line(img, (rf_x, left_top[1]), (rf_x, rf_y), VERTICAL_COLORS[2], thickness=thickness)
        cv2.line(img, (rb_x, left_top[1]), (rb_x, rb_y), VERTICAL_COLORS[3], thickness=thickness)
        cv2.line(img, (rf_x, rf_y), (rb_x, rb_y), VERTICAL_COLORS[-1], thickness=thickness)
    elif direction_flag == 28:
        # left_behind, right_front, right_behind
        if POST_PROCESS:
            lb_x = left_top[0]
            rf_x = right_bottom[0]

        cv2.line(img, (lb_x, left_top[1]), (lb_x, lb_y), VERTICAL_COLORS[1], thickness=thickness)
        cv2.line(img, (rf_x, left_top[1]), (rf_x, rf_y), VERTICAL_COLORS[2], thickness=thickness)
        cv2.line(img, (rb_x, left_top[1]), (rb_x, rb_y), VERTICAL_COLORS[3], thickness=thickness)
        cv2.line(img, (rf_x, rf_y), (rb_x, rb_y), VERTICAL_COLORS[-1], thickness=thickness)
    elif direction_flag == 30:
        # left_front, left_behind, right_front, right_behind
        if vis_invis:
            cv2.circle(img, (lf_x, lf_y),4,VERTICAL_COLORS[0],4)
            cv2.circle(img, (rb_x, rb_y),4, VERTICAL_COLORS[3],4)
            cv2.circle(img, (lb_x, lb_y),4, VERTICAL_COLORS[1],4)
            cv2.circle(img, (rf_x, rf_y),4, VERTICAL_COLORS[2],4)
        pass

    
    return img

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

    src_dir = np.array([0, src_w * -0.5])
    dst_dir = np.array([0, dst_w * -0.5])

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

def affine_transform(point, matrix):
	point = point.reshape(-1, 2)
	point_exd = np.concatenate((point, np.ones((point.shape[0], 1))), axis=1)
	new_point = np.matmul(point_exd, matrix.T)
	return new_point[:, :2].squeeze()


class LabelParser(object):
    def __init__(self, img_shape: tuple, crop_coor:list):
        self.crop_coor = crop_coor
        input_width, input_height = img_shape[0], img_shape[1]
        self.input_width, self.input_height = input_width,input_height
        
        # self.roi_width, self.roi_high = roi_width,roi_high
        self.original_size=None
        self.upper_bound = crop_coor[1]
        self.lower_bound = crop_coor[3]
        self.offset_l = crop_coor[0]
        self.offset_r = img_shape[0] - crop_coor[2]

    def _get_border(self, border, size):
        i = 1
        while size - border // i <= border // i:
            i *= 2
        return border // i

    def updatebox(self,keypoint_down,keypoint_up,box2d):
        # 0print(f"box2d: {box2d}")
        bottom = keypoint_down.reshape(-1, 2)[:,1]
        minbottom = [item for item in bottom if item>=0 ]
        # print(f"minbottom: {minbottom}")
        if len(minbottom)==1:
            box2d[3] = min(box2d[3],minbottom[0])
        elif len(minbottom)>1:
            box2d[3] = max(minbottom) #min(box2d[3],max(minbottom))

        top = keypoint_up.reshape(-1, 2)[:,1]
        mintop = [item for item in top if item>=0]
        if len(mintop)==1:
            box2d[1] = max(box2d[1],mintop[0])
        elif len(mintop)>1:
            box2d[1] = min(mintop)
        box2d[1] = max(box2d[1],self.upper_bound)
        box2d[[0, 2]] = box2d[[0, 2]].clip(self.offset_l, self.input_width-self.offset_r - 1)
        box2d[[1, 3]] = box2d[[1, 3]].clip(self.upper_bound, self.lower_bound - 1)
        return box2d

    def __call__(self, img, objs:List[Object3d], calib: Calibration=None):
        save_id = []

        # print("Original objs:", len(objs))
        for idx, obj in enumerate(objs):
            #print(obj.keypoint_up)
            if obj.type not in ["PD", "Rider", "CAR","VAN","Three","BUS","SPECIALCAR","TRUCK","TRUCKHEAD","AIV", "trailerback", "Crane"]:
                # print("Omit:", obj.type)
                continue 
            # print(upper_bound)
            obj.keypoint_to_insideimage((self.input_width,self.input_height), self.crop_coor) # has aggregtated
            # obj.keypointup_to_insideimage((self.input_width,self.input_height), upper_bound, self.offset)
    #             print(obj.keypoint_down)
    #             print(box2d," : ",box2d[2]-box2d[0])
            top_left_point = np.array([obj.xmin,obj.ymin])
            if top_left_point[1] < self.upper_bound:
                top_left_point[1] = self.upper_bound
            
            obj.xmin,obj.ymin = top_left_point[0], top_left_point[1]

            # if obj.type in ["CAR","VAN","Three","BUS","SPECIALCAR","TRUCK","TRUCKHEAD","AIV", "trailerback", "Crane"] and w>2 \
            #     and area_box2d_w != 0. and area_box2d_h != 0. and area_clip_w/area_box2d_w>0.2 and h/area_box2d_h>0.2:
            #     save_id.append(idx)
            #     #print("SAVE:", obj.type," : ",h," : ",w," : ",area_clip_w," : ",area_clip_h," : ", "RATIO W: " , area_clip_w/area_box2d_w,"RATIO H:", h/area_box2d_h)
            # else:
            #     #print("ABANDON:", obj.type," : ",h," : ",w," : ",area_clip_w," : ",area_clip_h," : ",w/h)
            #     pass
            # if obj.type in ["PD","Rider"] and w>2 and area_clip_w/area_box2d_w>0.4:
            #     save_id.append(idx)
            # update truncation coef 
            # obj.truncation = 0
            save_id.append(idx)
            box2d = obj.box2d
            obj.box2d = self.updatebox(obj.keypoint_down,obj.keypoint_up, box2d)
            # print(obj.keypoint_down.astype(int))
            obj.xmin = obj.box2d[0]
            obj.xmax = obj.box2d[2]
            obj.ymin = obj.box2d[1]
            obj.ymax = obj.box2d[3]
            obj.orientation_x = obj.keypoint_down[::2]
            obj.orientation_y = obj.keypoint_down[1::2]
            objs[idx] = obj 
   
        objs = [objs[id] for id in save_id]

        return img, objs, calib

#     def trans_obj(self, obj:Object3d):
#         # apply affine tramnsformation to objects
#         box2d = obj.box2d
# #         keypoint_down = obj.keypoint_down
# #         # print(keypoint_down)
# #         keypoint_down = keypoint_down.reshape(-1,2)
# #         x,y = np.where(keypoint_down ==[-1,-1])
# # #             print("affine before",keypoint_down)
# #         keypoint_down = affine_transform(keypoint_down, trans_affine)
# #         keypoint_down[x,y]=-1.0
# # #             print("affine after",keypoint_down)
# #         obj.keypoint_down = keypoint_down.reshape(-1)

# #         keypoint_up = obj.keypoint_up
# #         keypoint_up = keypoint_up.reshape(-1,2)
# #         x,y = np.where(keypoint_up ==[-1,-1])
# #         #print("affine before",keypoint_down)
# #         keypoint_up = affine_transform(keypoint_up, trans_affine)
# #         keypoint_up[x,y]=-1.0
# #         #print("affine after",keypoint_down)
# #         obj.keypoint_up = keypoint_up.reshape(-1)

# #             print(obj.keypoint_down)
        
#         upper_bound = self.input_height-self.ref_high
#         # print(upper_bound)
#         obj.keypoint_to_insideimage((self.input_width,self.input_height), upper_bound, self.offset)
#         obj.keypointup_to_insideimage((self.input_width,self.input_height), upper_bound, self.offset)
# #             print(obj.keypoint_down)
# #             print(box2d," : ",box2d[2]-box2d[0])
#         top_left_point = np.array([obj.xmin,obj.ymin])
#         if top_left_point[1] < self.upper_bound:
#             top_left_point[1] = self.upper_bound
        
#         obj.xmin,obj.ymin = top_left_point[0], top_left_point[1]

#         #print(img.size)
#         box2d_inside = box2d.copy() # 原始图像内的box 经过affine 后 box 大小 
#         # if self.original_size[1] == self.bsd_height_original:
#         #     box2d_inside[[0, 2]] = box2d_inside[[0, 2]].clip(0, self.ref_width_bsd - 1)
#         #     box2d_inside[[1, 3]] = box2d_inside[[1, 3]].clip(0, self.bsd_height_original - 1)
#         # elif self.original_size[1] == self.aiv_height_original:
#         #     # print("IMG AIV")
#         #     box2d_inside[[0, 2]] = box2d_inside[[0, 2]].clip(0, self.ref_width_aiv - 1)
#         #     box2d_inside[[1, 3]] = box2d_inside[[1, 3]].clip(0, self.aiv_height_original - 1)
#         # else:
#         #     box2d_inside[[0, 2]] = box2d_inside[[0, 2]].clip(0, self.ref_width_hh - 1)
#         #     box2d_inside[[1, 3]] = box2d_inside[[1, 3]].clip(0, self.hh_height_original - 1)
        
#         # box2d_inside[[0, 2]] = box2d_inside[[0, 2]].clip(0, self.ref_width_hh - 1)
#         # box2d_inside[[1, 3]] = box2d_inside[[1, 3]].clip(0, self.hh_height_original - 1)
        
#         # # box2d_inside[:2] = affine_transform(box2d_inside[:2], trans_affine)
#         # # box2d_inside[2:] = affine_transform(box2d_inside[2:], trans_affine)
        
#         # # 变换后在box的大小
#         # area_box2d_w = (box2d_inside[2]-box2d_inside[0])
#         # area_box2d_h = (box2d_inside[3]-box2d_inside[1])


#         # # box2d[:2] = affine_transform(box2d[:2], trans_affine) # 允许外 原始图像的box 经过affine 后 box 大小 
#         # # box2d[2:] = affine_transform(box2d[2:], trans_affine)

#         # box2d[[0, 2]] = box2d[[0, 2]].clip(0, self.input_width - 1)
#         # box2d[[1, 3]] = box2d[[1, 3]].clip(0, self.input_height - 1)
        
#         # # 变换后box在图像内的大小
#         # area_clip_w = abs(box2d[2]-box2d[0])
#         # area_clip_h = abs(box2d[3]-box2d[1])
#         # # cal iou : (intersection bewteen affined box2d and affined image corner )/ ( affined box2d )
#         # ious = box_iou(torch.Tensor(box2d).view(-1,4), roi)
        
#         ######## 取绝对值
#         h, w = abs(box2d[3] - box2d[1]), abs(box2d[2] - box2d[0])
#         return obj, h, w, area_clip_h, area_clip_w, area_box2d_h, area_box2d_w, box2d

    def show_after_augmentation(self,img, objs:List[Object3d], folder, basename, idx=-1):
        img3= np.copy(img)
        img3 = cv2.cvtColor(img3, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"./out/{folder}/original/"+basename+f"_{idx}.jpg", img3)
        
        img3 = cv2.rectangle(img3, (self.crop_coor[0], self.crop_coor[1]), (self.crop_coor[2], self.crop_coor[3]), (0, 255, 0))
        for oid, obj in enumerate(objs):
            cls = obj.type
            if cls == 'PD'or cls =='Rider':
                #print("人 /骑行着 ： ",obj.box2d)
                cv2.rectangle(img3, (int(obj.box2d[0]), int(obj.box2d[1])), (int(obj.box2d[2]), int(obj.box2d[3])), color=(0,0,255), thickness=1)
            if cls == 'CAR' or cls == 'BUS'or cls == 'Three' or cls =='TRUCK'or cls=='AIV'  or cls=='TRUCKHEAD' or cls=='VAN' or cls =='SPECIALCAR' or cls == "trailerback" or cls == "Crane":
                keypoints_down = obj.keypoint_down.copy()
                keypoint_up = obj.keypoint_up.copy()
                cv2.rectangle(img3, (int(obj.box2d[0]), int(obj.box2d[1])), (int(obj.box2d[2]), int(obj.box2d[3])), color=(0,0,255), thickness=2)
                for i in range(4):
                    if keypoints_down[2*i+1]>=0 and keypoints_down[2*i]>=0 :
    #                     print("plot on img",keypoints_down[2*i],keypoints_down[2*i+1])
                        img3=cv2.circle(img3, (round(float(keypoints_down[2*i])),round(float(keypoints_down[2*i+1]))), 5, color=(0, 255, 1), thickness=3)
                        img3=cv2.circle(img3, (round(float(keypoint_up[2*i])),round(float(keypoint_up[2*i+1]))), 5, color=(0, 0, 255), thickness=3)
            keypoint = np.zeros((4, 2), dtype=float)
            keypoint_vis = np.zeros(4, dtype=bool)
            for index in range(4):
                px = obj.keypoint_down[2*index]
                py = obj.keypoint_down[2*index+1]
                keypoint[index, 0] = px 
                keypoint[index, 1] = py
                if px == -1. and py == -1.:
                    keypoint_vis[index] = False
                else:
                    keypoint_vis[index] = True
            if obj.type in [ "CAR","PD","Rider","Three","BUS","TRUCK","TRUCKHEAD","VAN","SPECIALCAR", "trailerback", "Crane"]:
                cv2.putText(img3, '{}{}'.format(oid, obj.type), tuple(obj.box2d[0:2].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, \
                    0.6 , (255, 100, 0), 1, cv2.LINE_AA)
                if obj.type in [ "CAR","Three","BUS","TRUCK","TRUCKHEAD","VAN","SPECIALCAR","trailerback", "Crane"]:
                    img3 = show_result_keypoints(img3,keypoint_vis,keypoint,obj.box2d)
        if idx >=0:
            cv2.imwrite(os.path.join(f'./out/{folder}/{basename}_{idx}.jpg'), img3)
        else:
            cv2.imwrite(os.path.join(f'./out/{folder}/{basename}.jpg'), img3)
        
def approx_proj_center(proj_center, surface_centers, img_size):
    # surface_inside
    img_w, img_h = img_size
    surface_center_inside_img = (surface_centers[:, 0] >= 0) & (surface_centers[:, 1] >= 0) & \
                            (surface_centers[:, 0] <img_w) & (surface_centers[:, 1] <img_h)

    if surface_center_inside_img.sum() > 0:
        target_surface_center = surface_centers[surface_center_inside_img.argmax()]
        # y = ax + b
        a, b = np.polyfit([proj_center[0], target_surface_center[0]], [proj_center[1], target_surface_center[1]], 1)
        valid_intersects = []
        valid_edge = []

        left_y = b
        if (0 <= left_y <img_h):
            valid_intersects.append(np.array([0, left_y]))
            valid_edge.append(0)

        right_y = (img_w - 1) * a + b
        if (0 <= right_y <img_h):
            valid_intersects.append(np.array([img_w - 1, right_y]))
            valid_edge.append(1)

        top_x = -b / a
        if (0 <= top_x <img_w):
            valid_intersects.append(np.array([top_x, 0]))
            valid_edge.append(2)

        bottom_x = (img_h - 1 - b) / a
        if (0 <= bottom_x <img_w):
            valid_intersects.append(np.array([bottom_x, img_h - 1]))
            valid_edge.append(3)
            
        if valid_intersects==[]:
            return None,None
        valid_intersects = np.stack(valid_intersects)
        min_idx = np.argmin(np.linalg.norm(valid_intersects - proj_center.reshape(1, 2), axis=1))
        
        return valid_intersects[min_idx], valid_edge[min_idx]
    else:

        return None,None
    
def box_to_string2( name,
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

if __name__ == "__main__":
    randomAffineCrop = LabelParser((1936, 1220), [8,68,1910,1220]) 
    # imageset_txt_path ="/home/utopilot/workspace/infer/HH_3D_SIDE/test_aug.txt"
    # dataset = "/home/utopilot/workspace/infer/HH_3D_SIDE/"
    
    folder = "croptest_0129"
    os.makedirs(f"./out/{folder}/", exist_ok=True)
    os.makedirs(f"./out/{folder}/savetxt/", exist_ok=True)
    os.makedirs(f"./out/{folder}/original/", exist_ok=True)
    
    dataroot = "/home/utopilot/workspace/infer/eval_txt/img/"
    
    lines = os.listdir(dataroot)
    lines = ["truck38_fl_20221107_165003_650_081.jpg"]
    for line in lines:
        print()
        if not line.endswith(".jpg"):
            continue
        label_file = os.path.join("/home/utopilot/workspace/infer/eval_txt/gt/", line.replace("jpg", "txt"))
        line = os.path.join(dataroot, line)
        
        # img = cv2.imread(img_path)
        
        #objs = read_label(label_path)
        line = line.strip()
        # line = "/home/utopilot/workspace/infer/HH_3D_SIDE/Image0000002/" + line + ".jpg"
        print(line)

        basename = os.path.basename(line).split(".")[0]
        calib_file = None # no calib file

        for i in range(1):
            img = Image.open(line)
            # print("Original Size:", img.size)
            objs = read_label(label_file)
            print(objs[0].type)
            print("index", i, "objs:", len(objs))
            if calib_file != None:
                calib = Calibration(calib_file)
            else:
                calib = None
            
            # objs = vis_two_box(img, objs, "test")
            img, objs, calib = randomAffineCrop(img, objs, calib)
            print("new:", len(objs))
            
            randomAffineCrop.show_after_augmentation(img, objs, folder, basename, i)

            
            # with open(os.path.join(f"./out/{folder}/savetxt/", basename+f"_{i}.txt"),'w') as f:
            #     for i,obj in enumerate(objs):
            #         keypoint = np.zeros((4, 2), dtype=float)
            #         keypoint_vis = np.zeros(4, dtype=bool)
            #         for index in range(4):
            #             px = obj.keypoint_down[2*index]
            #             py = obj.keypoint_down[2*index+1]
            #             keypoint[index, 0] = px
            #             keypoint[index, 1] = py
            #             if px == -1. and py == -1.:
            #                 keypoint_vis[index] = False
            #             else:
            #                 keypoint_vis[index] = True
            #         # print(obj.alpha)
            #         cube = np.zeros([8,2])-1
            #         cube[0:4,0:2] = keypoint
            #         cube = cube.tolist()
            #         visline =[-1,-1,-1,-1] # ?
            #         v_id = int(obj.vid)
            #         output = box_to_string2(obj.type,[obj.w,obj.l,obj.h],[obj.t[0],obj.t[1],obj.t[2]],obj.box2d,float(obj.truncation+0),0,obj.alpha,obj.ry,cube,visline,v_id)
            #         f.write(output + '\n')