import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2
cv2.setNumThreads(0)
import os
import pdb

from PIL import Image
import time 
from scipy.optimize import minimize
from config import TYPE_ID_CONVERSION
from shapely.geometry import Polygon
from config import cfg
from utils.visualizer import Visualizer
from utils.kitti_utils import draw_projected_box3d, draw_box3d_on_top, init_bev_image, draw_bev_box3d

keypoint_colors = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], 
                [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0], 
                [107, 142, 35], [255, 0, 0], [0, 0, 142], [0, 0, 70],
                [152, 251, 152], [0, 130, 180], [220, 20, 60], [0, 60, 100]]

def box_iou(box1, box2):
    intersection = max((min(box1[2], box2[2]) - max(box1[0], box2[0])), 0) * max((min(box1[3], box2[3]) - max(box1[1], box2[1])), 0)
    union = (box1[2] - box1[0]) * (box1[3] - box1[1]) + (box2[2] - box2[0]) * (box2[3] - box2[1]) - intersection

    return intersection / union

def box_iou_3d(corner1, corner2):
    # for height overlap, since y face down, use the negative y
    min_h_a = -corner1[0:4, 1].sum() / 4.0
    max_h_a = -corner1[4:8, 1].sum() / 4.0
    min_h_b = -corner2[0:4, 1].sum() / 4.0
    max_h_b = -corner2[4:8, 1].sum() / 4.0

    # overlap in height
    h_max_of_min = max(min_h_a, min_h_b)
    h_min_of_max = min(max_h_a, max_h_b)
    h_overlap = max(0, h_min_of_max - h_max_of_min)

    if h_overlap == 0:
        return 0

    # x-z plane overlap
    box1, box2 = corner1[0:4, [0, 2]], corner2[0:4, [0, 2]]
    bottom_a, bottom_b = Polygon(box1), Polygon(box2)
    if bottom_a.is_valid and bottom_b.is_valid:
        # check is valid, A valid Polygon may not possess any overlapping exterior or interior rings.
        bottom_overlap = bottom_a.intersection(bottom_b).area

    overlap_3d = bottom_overlap * h_overlap        
    union3d = bottom_a.area * (max_h_a - min_h_a) + bottom_b.area * (max_h_b - min_h_b) - overlap_3d

    return overlap_3d / union3d

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


def rotate_dims(roty ,dim):
    """ Takes an object and a projection matrix (P) and projects the 3d
    bounding box into the image plane.
    Returns:
        corners_2d: (8,2) array in left image coord.
        corners_3d: (8,3) array in in rect camera coord.
    """
    # compute rotational matrix around yaw axis
    R = np.array([[np.cos(roty), 0, np.sin(roty)],
                  [0, 1, 0],
                  [-np.sin(roty), 0, np.cos(roty)]])
    # 3d bounding box dimensions
    h, w, l = dim
    # pdb.set_trace()
    # 3d bounding box corners
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

    # rotate and translate 3d bounding box
    corners_3d_offset = np.dot(R, np.vstack([x_corners, y_corners, z_corners])).T
 
    return corners_3d_offset

def corner_to_3dboundingbox(pred_locations_3D,dim,roty,center_type):
    corner_offset = rotate_dims(roty ,dim)
    
    if (center_type ==0):
#         print("0roty: ", roty)
#         print("0loc shape: ",pred_locations_3D)
#         print("0dim shape: ",dim)
#         print("0corner_offset: ",corner_offset)
        corners_3d = pred_locations_3D+corner_offset
#         print("0corners_3d: ",corners_3d)


    if (center_type ==1):
#         print("roty: ", roty)
#         print("loc shape: ",pred_locations_3D)
#         print("dim shape: ",dim)
#         print("corner_offset: ",corner_offset)
        
        predict_center=pred_locations_3D
        predict_center[0]-=corner_offset[1,0]
        predict_center[2]-=corner_offset[1,2]
        predict_center[1]-=dim[0]/2
        corners_3d= predict_center+corner_offset
#         print("corners_3d: ",corners_3d)
        
    if (center_type ==2):
        predict_center=pred_locations_3D
        predict_center[0]-=corner_offset[0,0]
        predict_center[2]-=corner_offset[0,2]
        predict_center[1]-=dim[0]/2
        corners_3d= predict_center+corner_offset

        
    if (center_type ==3):
        predict_center=pred_locations_3D
        predict_center[0]-=corner_offset[3,0]
        predict_center[2]-=corner_offset[3,2]
        predict_center[1]-=dim[0]/2
        corners_3d= predict_center+corner_offset

        
    if (center_type ==4):
        predict_center=pred_locations_3D
        predict_center[0]-=corner_offset[2,0]
        predict_center[2]-=corner_offset[2,2]
        predict_center[1]-=dim[0]/2
        corners_3d= predict_center+corner_offset

        
    return corners_3d

# visualize for test-set
def show_image_with_boxes_test(image, output, target, visualize_preds):
    # output Tensor:
    # clses, pred_alphas, pred_box2d, pred_dimensions, pred_locations, pred_rotys, scores
    image = image.numpy().astype(np.uint8)
    output = output.cpu().float().numpy()
    
    # filter results with visualization threshold
    vis_thresh = cfg.TEST.VISUALIZE_THRESHOLD
    output = output[output[:, -1] > vis_thresh]

    ID_TYPE_CONVERSION = {k : v for v, k in TYPE_ID_CONVERSION.items()}
    clses = output[:, 0]
    box2d = output[:, 2:6]
    dims = output[:, 6:9]
    locs = output[:, 9:12]
    rotys = output[:, 12]
    score = output[:, 13]
    keypoints = visualize_preds['keypoints'].cpu()
    proj_center = visualize_preds['proj_center'].cpu()
    
    calib = target.get_field('calib')
    pad_size = target.get_field('pad_size')

    # B x C x H x W  ----> H x W x C
    pred_heatmap = visualize_preds['heat_map']
    all_heatmap = np.asarray(pred_heatmap[0, :, ...].cpu().sum(dim=0))
    all_heatmap = cv2.resize(all_heatmap, (1280, 384))
    all_heatmap = all_heatmap[pad_size[1] : pad_size[1] + image.shape[0], pad_size[0] : pad_size[0] + image.shape[1]]

    img2 = Visualizer(image.copy()) # for 2d bbox
    img3 = image.copy() # for 3d bbox
    img4 = init_bev_image() # for bev
    img_keypoint = image.copy() 
    font = cv2.FONT_HERSHEY_SIMPLEX

    pred_color = (0, 255, 0)
    # plot prediction 
    for i in range(box2d.shape[0]):
        img2.draw_box(box_coord=box2d[i], edge_color='g')
        img2.draw_text(text='{}, {:.3f}'.format(ID_TYPE_CONVERSION[clses[i]], score[i]), position=(int(box2d[i, 0]), int(box2d[i, 1])))

        corners3d = box3d_to_corners(locs[i], dims[i], rotys[i])
        corners_2d, depth = calib.project_rect_to_image(corners3d)
        img3 = draw_projected_box3d(img3, corners_2d, color=pred_color)

        corners3d_lidar = calib.project_rect_to_velo(corners3d)
        img4 = draw_box3d_on_top(img4, corners3d_lidar[np.newaxis, :], thickness=2, color=pred_color, scores=None)
        # 10 x 2
        keypoint_i = (keypoints[i].view(-1, 2) + proj_center[i].view(-1, 2)) * 4 - pad_size.view(1, 2)
        # depth from keypoint
        center_height = keypoint_i[-2, -1] - keypoint_i[-1, -1]
        edge_height = keypoint_i[:4, -1] - keypoint_i[4:8, -1]
        # depth of four edges
        edge_depth = calib.f_u * dims[i, 0] / edge_height
        center_depth = calib.f_u * dims[i, 0] / center_height
        edge_depth = [edge_depth[[0, 3]].mean(), edge_depth[[1, 2]].mean()]
        # print(locs[i, -1], center_depth, edge_depth)

        for i in range(keypoint_i.shape[0]):
            cv2.circle(img_keypoint, tuple(keypoint_i[i]), 4, keypoint_colors[i], -1)

    img2 = img2.output.get_image()
    heat_mixed = img2.astype(np.float32) / 255 + all_heatmap[..., np.newaxis] * np.array([1, 0, 0]).reshape(1, 1, 3)
    img3 = img3.astype(np.float32) / 255
    stacked_img = np.vstack((heat_mixed, img3))

    plt.figure()
    plt.imshow(stacked_img)
    plt.title('2D and 3D results')
    plt.show()

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
    thickness=4
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
    
# # heatmap and 3D detections
def show_image_with_3dboxes(image, output, calib,visualize_preds,idx,save_folder):
    # output Tensor:
    # clses, pred_alphas, pred_box2d, pred_dimensions, pred_locations, pred_rotys, scores
    image = image.numpy().astype(np.uint8)
    output = output.cpu().float().numpy()
    
    # filter results with visualization threshold
    vis_thresh = cfg.TEST.VISUALIZE_THRESHOLD
    output = output[output[:, -1] > vis_thresh]
    ID_TYPE_CONVERSION = {k : v for v, k in TYPE_ID_CONVERSION.items()}

    # predictions
    clses = output[:, 0]
    box2d = output[:, 2:6]
    dims = output[:, 6:9]
    locs = output[:, 9:12]
    rotys = output[:, 12]
    score = output[:, 13]
    
    #box_center = (box2d[:,:2] + box2d[:,2:]) / 2
    proj_center = visualize_preds['proj_center'].cpu()
    keypoints = visualize_preds['keypoints'].cpu()
    center_type = visualize_preds['center_type']
    center_type = torch.softmax(center_type, dim=1)
    center_type = center_type.argmax(dim=1)
    center_type = center_type.cpu().data.numpy()
    
    keypoints_vis = visualize_preds['keypoints_visible'].reshape(-1,4,2)
    keypoints_vis = torch.softmax(keypoints_vis, dim=2)
    keypoints_vis = keypoints_vis[:,:, 0] < keypoints_vis[:,:, 1]
    keypoints_vis =keypoints_vis.cpu().data.numpy()
    
    pred_heatmap = visualize_preds['heat_map']
    all_heatmap = np.asarray(pred_heatmap[0, 0, ...].cpu())
    all_heatmap = cv2.resize(all_heatmap, (image.shape[1], image.shape[0]))

    img2 = Visualizer(image.copy()) # for 2d bbox
    img3 = image.copy() # for 3d bbox
    img_keypoint = image.copy() # for keypoint
    img_centerpoint = image.copy() # for keypoint
    img4 = init_bev_image() # for bev

    # plot prediction 
    for i in range(box2d.shape[0]):
        img2.draw_box(box_coord=box2d[i], edge_color='g')
        # corners3d = box3d_to_corners(locs[i], dims[i], rotys[i])
        corners3d = corner_to_3dboundingbox(locs[i], dims[i], rotys[i],center_type[i])
        corners_2d, depth = calib.project_rect_to_image(corners3d)
        img3 = draw_projected_box3d(img3, corners_2d, color=(0, 255, 0))
        keypoint_i = (keypoints[i].view(-1, 2) + proj_center[i].view(-1, 2)) * 4
        keypoint_i = keypoint_i.data.numpy().astype(np.int)
#         keypoint_i = keypoints[i].data.numpy()
#         keypoint_i = keypoint_i.reshape(-1, 2)*4 + box_center[i].reshape(-1, 2)
#         keypoint_i = keypoint_i.astype(np.int)
        
#         print(i,"box_center : ",box_center[i])
#         print(i," offset: ",keypoints[i])
#         print(i,"keypoint_i : ",keypoint_i)
        if ID_TYPE_CONVERSION[int(clses[i])] in ['CAR','BUS','TRUCK','Three','VAN','trailerback']:
            img_centerpoint = show_result_keypoints(img_centerpoint,keypoints_vis[i],keypoint_i,box2d[i])
#             print(proj_center[i])
#             import pdb
#             pdb.set_trace()
            proj_center_ = proj_center[i].data.numpy().astype(np.int)*4
            cv2.circle(img_keypoint, tuple(proj_center_),1,(0,255),2)
            cv2.putText(img_keypoint, '{}'.format(center_type[i]), tuple(proj_center_), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.5, (255, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(img_keypoint, '{}'.format(locs[i]), tuple(proj_center_-10), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.5, (255, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(img_keypoint, 'h {}/ w {}/l {}'.format(dims[i][0],dims[i][1],dims[i][2]), tuple(proj_center_-20), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.5, (255, 0, 0), 1, cv2.LINE_AA)

#             for k in range(0, 4):
#                 i, j = k, (k + 1) % 4
#                 cv2.line(img_keypoint, tuple(keypoint_i[i]), tuple(keypoint_i[j]), (0,0,255), 1, cv2.LINE_AA)
            
# 		corners3d_lidar = calib.project_rect_to_velo(corners3d)
# 		img4 = draw_bev_box3d(img4, corners3d[np.newaxis, :], thickness=2, color=pred_color, scores=None)
    img2 = img2.output.get_image()
    img4 = cv2.resize(img2, (img3.shape[0], img3.shape[0]))
    
    img_keypoint = img_keypoint.copy()
    
    
    stack_img = np.concatenate([img3, img4], axis=1)

    result = os.path.join(save_folder, str(idx) + "_3d.jpg")
    cv2.imwrite(result, stack_img)
    result = os.path.join(save_folder, str(idx) + "_keypoint.jpg")
    cv2.imwrite(result, img_centerpoint)
    result = os.path.join(save_folder, str(idx) + "_heatmap.jpg")
    cv2.imwrite(result, img_keypoint)
    
# heatmap and 2D detections
def show_image_with_2dboxes(image, output, visualize_preds,idx):
    # output Tensor:
    # clses, pred_alphas, pred_box2d, pred_dimensions, pred_locations, pred_rotys, scores
    image = image.numpy().astype(np.uint8)
    output = output.cpu().float().numpy()
    
    # filter results with visualization threshold
    vis_thresh = cfg.TEST.VISUALIZE_THRESHOLD
    output = output[output[:, -1] > vis_thresh]
    ID_TYPE_CONVERSION = {k : v for v, k in TYPE_ID_CONVERSION.items()}

    # predictions
    clses = output[:, 0]
    box2d = output[:, 2:6]
    dims = output[:, 6:9]
    locs = output[:, 9:12]
    rotys = output[:, 12]
    score = output[:, 13]

    proj_center = visualize_preds['proj_center'].cpu()
    keypoints = visualize_preds['keypoints'].cpu()

    pred_heatmap = visualize_preds['heat_map']
    all_heatmap = np.asarray(pred_heatmap[0, 0, ...].cpu())
    all_heatmap = cv2.resize(all_heatmap, (image.shape[1], image.shape[0]))

    img2 = Visualizer(image.copy()) # for 2d bbox

    # plot prediction 
    for i in range(box2d.shape[0]):
        img2.draw_box(box_coord=box2d[i], edge_color='g')
# 		img2.draw_text(text='{}, {:.3f}'.format(ID_TYPE_CONVERSION[clses[i]], score[i]), position=(int(box2d[i, 0]), int(box2d[i, 1])))

    img2 = img2.output.get_image()
    new_save_path = "./tmp"
    os.makedirs(new_save_path,exist_ok=True)

    result = os.path.join(new_save_path, str(idx) + ".jpg")
    cv2.imwrite(result, img2)

# # heatmap and 3D detections
# def show_image_with_3dboxes(image, output, visualize_preds,idx,save_folder):
# 	# output Tensor:
# 	# clses, pred_alphas, pred_box2d, pred_dimensions, pred_locations, pred_rotys, scores
# 	image = image.numpy().astype(np.uint8)
# 	output = output.cpu().float().numpy()
    
# 	# filter results with visualization threshold
# 	vis_thresh = cfg.TEST.VISUALIZE_THRESHOLD
# 	output = output[output[:, -1] > vis_thresh]
# 	ID_TYPE_CONVERSION = {k : v for v, k in TYPE_ID_CONVERSION.items()}

# 	# predictions
# 	clses = output[:, 0]
# 	box2d = output[:, 2:6]
# 	dims = output[:, 6:9]
# 	locs = output[:, 9:12]
# 	rotys = output[:, 12]
# 	score = output[:, 13]

# 	proj_center = visualize_preds['proj_center'].cpu()
# 	keypoints = visualize_preds['keypoints'].cpu()

# 	pred_heatmap = visualize_preds['heat_map']
# 	all_heatmap = np.asarray(pred_heatmap[0, 0, ...].cpu())
# 	all_heatmap = cv2.resize(all_heatmap, (image.shape[1], image.shape[0]))

# 	img2 = Visualizer(image.copy()) # for 2d bbox
# 	img3 = image.copy() # for 3d bbox
# 	img4 = init_bev_image() # for bev
    
    
# 	# plot prediction 
# 	for i in range(box2d.shape[0]):
# 		img2.draw_box(box_coord=box2d[i], edge_color='g')
# # 		img2.draw_text(text='{}, {:.3f}'.format(ID_TYPE_CONVERSION[clses[i]], score[i]), position=(int(box2d[i, 0]), int(box2d[i, 1])))
#         corners3d = box3d_to_corners(locs[i], dims[i], rotys[i])
# 		corners_2d, depth = calib.project_rect_to_image(corners3d)
# 		img3 = draw_projected_box3d(img3, corners_2d, cls=ID_TYPE_CONVERSION[clses[i]], color=pred_color, draw_corner=False)

# 		corners3d_lidar = calib.project_rect_to_velo(corners3d)
# 		img4 = draw_bev_box3d(img4, corners3d[np.newaxis, :], thickness=2, color=pred_color, scores=None)
        
# 	img2 = img2.output.get_image()
# 	img4 = cv2.resize(img4, (img3.shape[0], img3.shape[0]))
# 	stack_img = np.concatenate([img3, img4], axis=1)
    
# 	result = os.path.join(save_folder, str(idx) + "_2d.jpg")
# 	cv2.imwrite(result, img2)
#     result = os.path.join(save_folder, str(idx) + "_3d.jpg")
# 	cv2.imwrite(result, stack_img)


# heatmap and 3D detections
def show_image_with_boxes(image, output, target, visualize_preds, vis_scores=None):
    # output Tensor:
    # clses, pred_alphas, pred_box2d, pred_dimensions, pred_locations, pred_rotys, scores
    image = image.numpy().astype(np.uint8)
    output = output.cpu().float().numpy()

    if vis_scores is not None:
        output[:, -1] = vis_scores.squeeze().cpu().float().numpy()
    
    # filter results with visualization threshold
    vis_thresh = cfg.TEST.VISUALIZE_THRESHOLD
    output = output[output[:, -1] > vis_thresh]
    ID_TYPE_CONVERSION = {k : v for v, k in TYPE_ID_CONVERSION.items()}

    # predictions
    clses = output[:, 0]
    box2d = output[:, 2:6]
    dims = output[:, 6:9]
    locs = output[:, 9:12]
    rotys = output[:, 12]
    score = output[:, 13]

    proj_center = visualize_preds['proj_center'].cpu()
    keypoints = visualize_preds['keypoints'].cpu()

    # ground-truth
    calib = target.get_field('calib')
    pad_size = target.get_field('pad_size')
    valid_mask = target.get_field('reg_mask').bool()
    trunc_mask = target.get_field('trunc_mask').bool()
    num_gt = valid_mask.sum()
    gt_clses = target.get_field('cls_ids')[valid_mask]
    gt_boxes = target.get_field('gt_bboxes')[valid_mask]
    gt_locs = target.get_field('locations')[valid_mask]
    gt_dims = target.get_field('dimensions')[valid_mask]
    gt_rotys = target.get_field('rotys')[valid_mask]

    print('detections / gt objs: {} / {}'.format(box2d.shape[0], num_gt))

    pred_heatmap = visualize_preds['heat_map']
    all_heatmap = np.asarray(pred_heatmap[0, 0, ...].cpu())
    all_heatmap = cv2.resize(all_heatmap, (image.shape[1], image.shape[0]))

    img2 = Visualizer(image.copy()) # for 2d bbox
    img3 = image.copy() # for 3d bbox
    img4 = init_bev_image() # for bev

    font = cv2.FONT_HERSHEY_SIMPLEX
    pred_color = (0, 255, 0)
    gt_color = (255, 0, 0)

    # plot prediction 
    for i in range(box2d.shape[0]):
        img2.draw_box(box_coord=box2d[i], edge_color='g')
        img2.draw_text(text='{}, {:.3f}'.format(ID_TYPE_CONVERSION[clses[i]], score[i]), position=(int(box2d[i, 0]), int(box2d[i, 1])))

        corners3d = box3d_to_corners(locs[i], dims[i], rotys[i])
        corners_2d, depth = calib.project_rect_to_image(corners3d)
        img3 = draw_projected_box3d(img3, corners_2d, cls=ID_TYPE_CONVERSION[clses[i]], color=pred_color, draw_corner=False)

        corners3d_lidar = calib.project_rect_to_velo(corners3d)
        img4 = draw_bev_box3d(img4, corners3d[np.newaxis, :], thickness=2, color=pred_color, scores=None)

    # plot ground-truth
    for i in range(num_gt):
        img2.draw_box(box_coord=gt_boxes[i], edge_color='r')

        # 3d bbox template
        l, h, w = gt_dims[i]
        x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
        y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
        z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

        # rotation matirx
        roty = gt_rotys[i]
        R = np.array([[np.cos(roty), 0, np.sin(roty)],
                      [0, 1, 0],
                      [-np.sin(roty), 0, np.cos(roty)]])

        corners3d = np.vstack([x_corners, y_corners, z_corners])  # (3, 8)
        corners3d = np.dot(R, corners3d).T
        corners3d = corners3d + gt_locs[i].numpy() + np.array([0, h / 2, 0]).reshape(1, 3)

        corners_2d, depth = calib.project_rect_to_image(corners3d)
        img3 = draw_projected_box3d(img3, corners_2d, color=gt_color, draw_corner=False)

        corners3d_lidar = calib.project_rect_to_velo(corners3d)
        img4 = draw_bev_box3d(img4, corners3d[np.newaxis, :], thickness=2, color=gt_color, scores=None)

    img2 = img2.output.get_image()
    new_save_path = "/workspace/code/monodetect/tmp/viz2d/"
    os.makedirs(new_save_path,exist_ok=True)

    result = os.path.join(new_save_path, time.ctime(time.time()-86640) + ".jpg")
    cv2.imwrite(result, img2)

    # heat_mixed = img2.astype(np.float32) / 255 + all_heatmap[..., np.newaxis] * np.array([1, 0, 0]).reshape(1, 1, 3)
    # img4 = cv2.resize(img4, (img3.shape[0], img3.shape[0]))
    # stack_img = np.concatenate([img3, img4], axis=1)

    # plt.figure(figsize=(12, 8))
    # plt.subplot(211)
    # plt.imshow(all_heatmap); plt.title('heatmap'); plt.axis('off')
    # plt.subplot(212)
    # plt.imshow(stack_img); plt.title('2D/3D boxes'); plt.axis('off')
    # plt.suptitle('Detections')
    # plt.show()