import sys
sys.path.append('/home/utopilot/workspace/infer/infer_test/')
import os
import cv2
import numpy as np   
import operator
from PIL import Image
import json
import utils.kitti_common as kitti
from utils.visualize_infer import show_result_keypoints
from eval.eval_utils.eval_kitti_utils import draw_boxcube, calculate_cube_error_onlyrear, draw_projected_box3d, \
    draw_bev_box3d, Calibration, read_label,\
    calculate_depth_error, sum_list, mean_list, seperate_POS_NEG
from eval.eval_utils.eval_vis_two_box import vis_two_box
from eval.eval_utils.label_parser import LabelParser
from eval.eval_utils.parse_results import parse_metrics, toxlxs_3d, toxlxs_cube, merge_video
from tqdm import tqdm
import argparse

colors = {"VAN" : (0, 0, 255),
          "CAR": (0, 255, 0),
          "PD": (255, 122, 0),
          "Rider": (0, 122, 255),
          "Three": (0, 122, 122),
          "TRUCK":(255, 0, 0),
          "BUS": (122, 122, 0),
          "SPECIALCAR":(122, 0, 122),
          "trailback":(122,122,122),
          "trailerback":(122,122,122)} #

def draw_2d(image, bbox_2d):
    ## 绘制2d box
    cv2.rectangle(image, (int(bbox_2d[0]), int(bbox_2d[1])),
                  (int(bbox_2d[2]), int(bbox_2d[3])), (0, 255, 0), 10)
    return image

def draw_point(image, center):
    cv2.circle(image, (int(center[0]), int(center[1])), 10,(0, 0, 255), 2)
    return image

def align_center_corners3d(whl,ry=0):
    """
    generate corners3d representation for this object
    :return corners_3d: (8, 3) corners of box3d in camera coord
    """
    l, h, w = whl[2],whl[1],whl[0]
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

    R = np.array([[np.cos(ry), 0, np.sin(ry)],
                    [0, 1, 0],
                    [-np.sin(ry), 0, np.cos(ry)]])

    corners3d = np.vstack([x_corners, y_corners, z_corners])  # (3, 8)
    corners3d = np.dot(R, corners3d).T
    corners3d = corners3d

    return corners3d

class Evaluator:
    def __init__(self, 
                 test_file:str, 
                 save_dir:str, 
                 gt_path:str, 
                 det_path:str, 
                 img_shape:tuple=(1936, 1220), 
                 crop_coor:tuple=[8, 28, 1928, 1220], 
                 calib_path:str=None):
        self.test_file = test_file
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.gt_path = gt_path
        self.det_path = det_path
        self.calib_path = calib_path
        _, _, self.imgpaths, self.labels, self.labels_det = self._init_label_det_object()
        self.cropper = LabelParser(img_shape, crop_coor)
        
    # def _init_cropper(self, img_shape:tuple, crop_coor:tuple):
    #     img_w, img_h = img_shape
    #     roi_w, roi_h = crop_coor
    #     return 
    
    def _init_label_det_object(self):      
        annos_det = []
        annos_gt = []
        # label_filenames = os.listdir(self.gt_path)
        # label_filenames.sort()
        image_files = []
        label_files = []
        dt_files = []
        with open(self.test_file, 'r') as fsets:
            test_list = list(map(lambda x: x.strip(), fsets.readlines()))
        test_list.sort()
        # print("图片数：",len(test_list))
        if self.test_file != []:
            label_filenames =[]
            for line in test_list:
                img_name = os.path.basename(line)
                image_files.append(line)
                a = line.split("/")
                label_file = self.gt_path + "{}".format(a[-1].replace("jpg", "txt"))
                label_files.append(label_file)
                dt_file = self.det_path + "{}".format(a[-1].replace("jpg", "txt"))
                dt_files.append(dt_file)
        labels =[]
        labels_det =[]
        
        for i, (label_filename_det, label_filename_gt) in enumerate(zip(dt_files, label_files)):
            # print(label_filename_det, label_filename_gt)
            if os.path.exists(label_filename_det) and os.path.exists(label_filename_gt):
                # print(label_filename_gt)
                label = read_label(label_filename_gt)
                label_det = read_label(label_filename_det)
                annos_gt.append(kitti.get_label_anno(label_filename_gt))
                annos_det.append(kitti.get_label_anno(label_filename_det))
                # filename.append(idx)
                labels.append(label)
                labels_det.append(label_det)
            elif os.path.exists(label_filename_gt):
                label = read_label(label_filename_gt)
                annos_gt.append(kitti.get_label_anno(label_filename_gt))
                annos_det.append(None)
                labels.append(label) 
                labels_det.append(None)
            else:
                annos_gt.append(None)
                annos_det.append(None)
                labels.append(None) 
                labels_det.append(None)
        return annos_gt,annos_det,image_files,labels,labels_det
    
    def evaluate(self, eval2D, eval25D, eval3D, vizGT=True, video=True, \
        box_size_range=[32, 96], channel=1, crop_box=[], eval_cls=None, lane=1):
        
        viz25D=eval2D
        viz3D=False
        
        x_dim = lane # 分车道
        classes = ['VAN', 'CAR', 'Three', 'TRUCK', 'BUS', 'SPECIALCAR', 'trailerback']
        
        if eval25D:
            rear_abs,rear_rlt,raw_data=[],[],[]
        if eval2D or eval3D:
            gt_depth= [[[] for i in range(channel) ] for j in range(x_dim )]
            absolute_error= [[[] for i in range(channel) ] for j in range(x_dim )]
            relative_error= [[[] for i in range(channel) ] for j in range(x_dim )]
            shape_error=[[[] for i in range(channel) ] for j in range(x_dim )]
            yaw_error=[[[] for i in range(channel) ] for j in range(x_dim )]
            depth_abs_list = [[[] for i in range(channel) ] for j in range(x_dim )]
            depth_list = [[[] for i in range(channel) ] for j in range(x_dim )]
            gt_num_list = [[[] for i in range(channel) ] for j in range(x_dim )]
            pre_num_list = [[[] for i in range(channel) ] for j in range(x_dim )]
            pre_iou_list = [[[] for i in range(channel) ] for j in range(x_dim )]
            tp_list = [[[] for i in range(channel) ] for j in range(x_dim )]
            fn_list = [[[] for i in range(channel) ] for j in range(x_dim )]
            ## 每一类计算误检漏检测
            cls_tp_list = {}
            cls_fn_list = {}
            cls_pre_iou_list = {}
            cls_pre_num_list = {}
            cls_gt_num_list = {}
            
            ## 根据2d box大小统计
            box_tp_list = []
            box_fn_list = []
            box_pre_iou_list = []
            box_pre_num_list = []
            box_gt_num_list = []
            box_area_list = [] # [[box_size, tp, fn, fn, img_path]]
            box_error_list = [] # [[box_size, error_x, error_y, w, h, imgpath]]
        miss_items = {} # key: imgname, value: [miss_item_index in txt]
        false_items = {}
        
        
        print("Evaluating")
        for j,(objs, objs_det, imgpath) in tqdm(enumerate(zip(self.labels, self.labels_det, self.imgpaths)), total=len(self.imgpaths), unit="imgs"):
            # print("="*20)
            idx =j 
            filename = os.path.basename(imgpath)
            if not os.path.exists(imgpath):
                continue
            # print(f"img {idx}:", filename)
            img = Image.open(imgpath)
            
            # Find calib file
            if self.calib_path is not None:
                calibrationPath = os.path.join(self.calib_path, filename.replace(".jpg", ".xml").replace(".png", ".xml"))
                # print(filename)
                # print(calibrationPath)
                try:
                    calib = Calibration(calibrationPath)
                # print(calib.c_u,calib.c_v)
                except:
                    calib = None
                    # print("No Caliberation File.")
            else:
                calib = None
            # print('objs',objs)
            # print('objs_det',objs_det)  
            # print("gt目标数{}：，检测目标数：{}.".format(len(objs), len(objs_det)))
            
            # 读取GT，若无则赋值None
            if (objs_det == None or objs_det == []) and (objs == None or objs == []):
                # print("无GT与DET")
                continue
            
            gt_all_box = []
            pre_all_box = []
            miss_flags = []
            false_flags = []
            
            if objs != None and len(objs) >0:
                
                # project to inside image
                # print("GT数量:{}".format(len(objs)))
                objs = vis_two_box(img, objs)
                _, objs, _ = self.cropper(img, objs, calib)
                # print("GT数量:{}".format(len(objs)))
                for i, obj in enumerate(objs):
                    # if obj.type != "Vehicles":
                    # 合并类
                    if obj.type in ["trailerback", "AIV", "TRUCKHEAD"]:
                        obj.type = "TRUCK"
                    if obj.type == "VAN":
                        obj.type = "CAR"
                    # if obj.type == "Rider":
                    #     obj.type = "PD"
                    if obj.xmin > obj.xmax:
                        obj.xmin, obj.xmax = obj.xmax, obj.xmin
                    if obj.ymin > obj.ymax:
                        obj.ymin, obj.ymax = obj.ymax, obj.ymin
                    gt_box = [obj.xmin, obj.ymin, obj.xmax, obj.ymax, obj.t[0], obj.t[2], obj.type,  obj.occlusion]
                    gt_all_box.append(gt_box)
                    # print(obj.occlusion)

            else:
                # print("无GT")
                gt_box = None
            
            # 读取det， 若无检测结果则赋值None
            if objs_det != None and len(objs_det) >0:
                # print("DET数量:{}".format(len(objs_det)))
                for i, (obj_det) in enumerate(objs_det):
                    
                    # 合并类
                    if obj_det.type == "trailerback" or obj_det.type == "AIV" or  obj_det.type == "TRUCKHEAD":
                        obj_det.type = "TRUCK"
                    if obj_det.type == "VAN":
                        obj_det.type = "CAR"
                    # if obj_det.type == "Rider":
                    #     obj_det.type = "PD"
                    if obj_det.xmin > obj_det.xmax:
                        obj_det.xmin, obj_det.xmax = obj_det.xmax, obj_det.xmin
                    if obj_det.ymin > obj_det.ymax:
                        obj_det.ymin, obj_det.ymax = obj_det.ymax, obj_det.ymin
                    pre_box = [obj_det.xmin, obj_det.ymin, obj_det.xmax, obj_det.ymax, obj_det.t[0], obj_det.t[2], obj_det.type]
                    pre_XX0 = round(float(pre_box[4]), 3)
                    pre_ZZ0 = round(float(pre_box[5]), 2)
                    # if -5.7 <= pre_XX0 < -1.9:
                    # if pre_XX0 < -1.9:    
                    #     index_XX = 0
                    # elif -1.9 <= pre_XX0 <= 1.9:
                    #     index_XX = 1
                    # # elif 1.9 < pre_XX0 <= 5.7:
                    # else:
                    #     index_XX = 2
            
                    # 不分车道
                    index_XX = 0
                    # 分深度
                    index_ZZ = int((pre_ZZ0 - pre_ZZ0 % 10) / 10)
                    if index_ZZ >= channel:
                        index_ZZ = channel-1
                    # print("xx:", index_XX)
                    # print("zz:", index_ZZ)
                    try:
                        pre_num_list[index_XX][index_ZZ].append(1)
                    except:
                        # 深度过深则加入最后一个channel
                        pre_num_list[index_XX][-1].append(1)
                        
                    if obj_det.type not in cls_pre_num_list.keys():
                        cls_pre_num_list[obj_det.type] = 1
                    else:
                        cls_pre_num_list[obj_det.type] += 1
                    box_size = min(abs((pre_box[2]-pre_box[0])), abs((pre_box[3]-pre_box[1])))
                    box_pre_num_list.append([box_size, obj_det.type, imgpath])
                    pre_all_box.append(pre_box)
            else:
                # print("无DET")
                pre_box = None
            
            if pre_box == None:
                ## 无预测结果， 计算漏检
                # 带深度信息depth误差和漏检率计算函数，计算漏检
                if eval2D or eval3D:
                    tp_list, fn_list, depth_abs_list, depth_list, gt_num_list, miss_flags = self.cal_3DIOU_Matrix_M(gt_all_box, pre_all_box,depth_list,depth_abs_list,tp_list,fn_list,gt_num_list, \
                                                                                                    cls_tp_list, cls_fn_list, cls_gt_num_list, box_tp_list, box_fn_list, box_gt_num_list, imgpath)
            elif gt_box == None:
                ## 无gt但有预测结果，计算误检
                # 带深度信息误检率计算函数, 计算误检率
                if eval2D or eval3D:
                    pre_iou_list, false_flags = self.cal_3DIOU_Matrix_F(gt_all_box, pre_all_box,pre_iou_list, cls_pre_iou_list, box_pre_iou_list, box_error_list, imgpath)
                
            else:
                ## 有预测结果及gt
                # 带深度信息depth误差和漏检率计算函数，计算漏检
                if eval2D or eval3D:
                    tp_list, fn_list, depth_abs_list, depth_list, gt_num_list, miss_flags = self.cal_3DIOU_Matrix_M(gt_all_box, pre_all_box,depth_list,depth_abs_list,tp_list,fn_list,gt_num_list,\
                                                                                                    cls_tp_list, cls_fn_list, cls_gt_num_list, box_tp_list, box_fn_list, box_gt_num_list, imgpath)
                # 带深度信息误检率计算函数
                    pre_iou_list, false_flags = self.cal_3DIOU_Matrix_F(gt_all_box, pre_all_box,pre_iou_list, cls_pre_iou_list, box_pre_iou_list, box_error_list, imgpath) # false detection # 
                # 匹配GT及检测结果, 深度误差最小匹配上
                # obj.id 记录匹配对应，从1开始记录，0表示未匹配
                if eval25D or eval3D:
                    matched_id=1
                    count = 0
                    for j, obj in enumerate(objs):
                        #正样本没标3d信息的不做匹配
                        # if obj.h == 0:
                        #     print("NO3D")
                        #     continue
                        #iou赋值0，距离赋值9999
                        list_iou = [0]*len(objs_det)
                        abs_list =[9999]*len(objs_det)
                        for i, obj_det in enumerate(objs_det):
                            #print("obj_det.id : ",obj_det.id)
                            if obj_det.id == 0: # 取未匹配过的
                                #print("obj.type :",obj.type )
                                #如果真值是车，就将检测到的车跟他一一计算IOU
                                classes = ['VAN', 'CAR', 'Three', 'TRUCK', 'BUS', 'SPECIALCAR', 'trailerback'] ## 检测类别
                                if obj.type in classes and obj_det.type in classes:
                                    area_b = (obj_det.xmax - obj_det.xmin) * (obj_det.ymax - obj_det.ymin)
                                    area_a = (obj.xmax - obj.xmin) * (obj.ymax - obj.ymin)
                                    w = min(obj.xmax, obj_det.xmax) - max(obj.xmin, obj_det.xmin)
                                    h = min(obj.ymax, obj_det.ymax) - max(obj.ymin, obj_det.ymin)
                                    if w<0 or h <0:
                                        pass
                                    else:
                                        area = w*h
                                        IOU = round(area / (area_b+area_a-area), 3)
                                        list_iou[i]=IOU
                                        abs_list[i]=round(abs(obj_det.t[2]- obj.t[2]), 3)
                                #如果真值是人，就将检测到的人跟他一一计算IOU
                                if (obj.type in ['PD', 'Rider']) and (obj_det.type in ['PD', 'Rider']):
                                    area_b = (obj_det.xmax - obj_det.xmin) * (obj_det.ymax - obj_det.ymin)
                                    area_a = (obj.xmax - obj.xmin) * (obj.ymax - obj.ymin)
                                    w = min(obj.xmax, obj_det.xmax) - max(obj.xmin, obj_det.xmin)
                                    h = min(obj.ymax, obj_det.ymax) - max(obj.ymin, obj_det.ymin)
                                    if w<0 or h <0:
                                        pass
                                    else:
                                        area = w*h
                                        IOU = round(area / (area_b+area_a-area), 3)
                                        list_iou[i]=IOU
                                        abs_list[i]=round(abs(obj_det.t[2]- obj.t[2]), 3)
                        #print(list_iou,abs_list)
                        #算法选择匹配上的那一对
                        if list_iou!=[] and max(list_iou) >= 0.5:
                            min_index, min_number = min(enumerate(abs_list), key=operator.itemgetter(1))#取深度最小为匹配上
                            # max_index, max_number = max(enumerate(list_iou), key=operator.itemgetter(1))#取iou最大为匹配上
                            
                            # print(f"obj:{j}, obj_det:{min_index}")
                            obj.id=matched_id
                            objs_det[min_index].id=matched_id
                            matched_id += 1
                            count += 1  
                    # print("最大匹配数： ", matched_id)
                    # print("det-gt匹配数:", count)
                    ## 计算 error
                    if eval25D:
                        rear_abs_pic,rear_rlt_pic,raw_data_pic = calculate_cube_error_onlyrear(objs,objs_det) # need match_id
                        
                        # print("right first channel: ", yaw_error_pic[2][0])
                        # if yaw_error_pic[2][0] !=[] and yaw_error_pic[2][0]>100:
                        #     print("==========================================")
                        # add to all pic
                        rear_abs.extend(rear_abs_pic)
                        rear_rlt.extend(rear_rlt_pic)
                        raw_data.extend(raw_data_pic) 
                    if eval3D:
                        gt_depth_pic, absolute_error_pic, relative_error_pic, shape_error_pic, yaw_error_pic=calculate_depth_error(objs,objs_det,channel)
                        gt_depth = [[gt_depth[j][i] + gt_depth_pic[j][i] for i in range(channel)] for j in range(x_dim )]
                        absolute_error = [[absolute_error[j][i] + absolute_error_pic[j][i] for i in range(channel)] for j in range(x_dim )]
                        relative_error = [[relative_error[j][i] + relative_error_pic[j][i] for i in range(channel)] for j in range(x_dim )]
                        shape_error = [[shape_error[j][i] + shape_error_pic[j][i] for i in range(channel)] for j in range(x_dim )]
                        yaw_error= [[yaw_error[j][i] + yaw_error_pic[j][i] for i in range(channel)] for j in range(x_dim )]
            
            if 1 in miss_flags:
                miss_items[filename] = [index for (index,value) in enumerate(miss_flags) if value == 1]
            if 1 in false_flags:
                false_items[filename] = [index for (index,value) in enumerate(false_flags) if value == 1]
                
            if (eval2D or eval3D) and (viz25D or vizGT or viz3D):
                img = cv2.imread(imgpath)
                
                if viz25D:
                    # print("VIZ25D")
                    os.makedirs(os.path.join(self.save_dir, "25D"),exist_ok=True)
                    img_2_3 = img.copy()
                    if vizGT:
                        img_2_2 = img.copy()
                
                if viz3D:
                    os.makedirs(os.path.join(self.save_dir, "3D"),exist_ok=True)
                    ## bev图
                    img2 = img.copy() # 3d box graph
                    metric_width, metric_height = 800, 800
                    worldsize = 160
                    polar_step_size_meters = 10
                    pixels_per_meter = metric_height//worldsize # 每米5个pixel 
                    center_pixel = ( metric_width//2, metric_height)
                    img4 = np.ones((metric_width, metric_height, 3), dtype=np.uint8) * 230
                    # # Draw metric polar grid
                    for i in range(1, int(metric_width/(polar_step_size_meters*pixels_per_meter))):
                        cv2.circle(
                            img4, center_pixel, int(i * polar_step_size_meters*pixels_per_meter ),
                            (50, 50, 50), 2)
                        cv2.line(img4, (0, center_pixel[1]-int(i * polar_step_size_meters*pixels_per_meter )), (metric_width, center_pixel[1]-int(i * polar_step_size_meters*pixels_per_meter )), (200, 200, 200))
                        cv2.line(img4, (abs(int(i * polar_step_size_meters*pixels_per_meter )),0), (abs(int(i * polar_step_size_meters*pixels_per_meter )),metric_height), (200, 200, 200))
                
                
                
                if objs != None:
                    for i,(obj, miss_flag) in enumerate(zip(objs, miss_flags)):
                        # print(1)
                        # print(obj.box2d[0]-obj.box2d[2])
                        # print(obj.box2d[1]-obj.box2d[3])
                        box_size = round(min(abs(obj.box2d[0]-obj.box2d[2]), abs(obj.box2d[1]-obj.box2d[3])), 1)
                        if viz3D:
                            # 3d box and bev
                            if obj.type in ['VAN', 'CAR', 'PD', 'Rider', 'Three', 'TRUCK', 'BUS', 'SPECIALCAR', 'trailerback']:
                                corners_3d = obj.generate_corners3d()
                                # if obj.w ==0.0: # show that obj only has 2d box 
                                    # img2 = cv2.rectangle(img2, (int(obj.box2d[0]), int(obj.box2d[1])), (int(obj.box2d[2]), int(obj.box2d[3])), color=(0, 255, 255), thickness=5)
                                img4 = draw_bev_box3d(img4, corners_3d[np.newaxis, :], obj.id, thickness=2, color=(255, 0, 0), scores=None, world_size=worldsize,out_size=metric_width) # circ pos
                                corners_2d, depth = calib.project_rect_to_image(corners_3d)
                                img2 = draw_projected_box3d(img2, corners_2d, obj.id, cls=obj.type, color=colors[obj.type], draw_orientation=True,draw_corner=False) # 3d box
                            if crop_box != []:
                                img = draw_2d(img, [calib.c_u-crop_box[1]//2,calib.c_v-crop_box[0]//2,calib.c_u+crop_box[1]//2,calib.c_v+crop_box[0]//2])
                                img = draw_point(img,[calib.c_u,calib.c_v])
                        if viz25D and vizGT:
                            if miss_flag == 1:
                                color_gt = (0,255,255)
                            elif miss_flag == -1:
                                color_gt = (0,215,0)
                            else:
                                color_gt = (0, 0, 255)
                            # print(2)
                            img_2_2 = cv2.rectangle(img_2_2, (int(obj.box2d[0]), int(obj.box2d[1])),
                                                (int(obj.box2d[2]), int(obj.box2d[3])), color=color_gt, thickness=3)
                            # print(obj.keypoint_down)
                            img_2_2 = draw_boxcube(img_2_2, obj)
                            cv2.putText(img_2_2, '{}'.format(obj.type), tuple(obj.box2d[0:2].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 
                                                   1, (255, 100, 0), 2, cv2.LINE_AA)
                            cv2.putText(img_2_2, '{}'.format(int(box_size)), tuple(obj.box2d[[0,3]].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 
                                                   1, (255, 100, 0), 2, cv2.LINE_AA)
                            cv2.putText(img_2_2, 'GT', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
                            
                    
                if objs_det != None:
                    for i, (obj_det,false_flag) in enumerate(zip(objs_det, false_flags)):   
                        box_size = round(min(abs(obj_det.box2d[0]-obj_det.box2d[2]), abs(obj_det.box2d[1]-obj_det.box2d[3])), 1)
                        if obj_det.type in  ['VAN', 'CAR', 'PD', 'Rider', 'Three', 'TRUCK', 'BUS', 'SPECIALCAR', 'trailerback']:
                            if viz3D:
                                corners_3d_det = obj_det.generate_corners3d()
                                # print("l, h, w, t, ry: ", obj_det.l, obj_det.h, obj_det.w,obj_det.t,obj_det.ry)
                                img4 = draw_bev_box3d(img4, corners_3d_det[np.newaxis, :], obj_det.id, thickness=2, color=colors[obj_det.type], scores=None,world_size=worldsize,out_size=metric_width)
                                # print(obj_det.box2d)
                                # img = cv2.rectangle(img, (int(obj_det.box2d[0]), int(obj_det.box2d[1])), (int(obj_det.box2d[2]), int(obj_det.box2d[3])), color=(0, 255, 255), thickness=3)
                                corners_2d_det, depth = calib.project_rect_to_image(corners_3d_det)
                                img = draw_projected_box3d(img, corners_2d_det, obj_det.id, cls=obj_det.type, color=colors[obj_det.type], draw_orientation=False,draw_corner=False)
                        
                        if viz25D:
                            if false_flag == 1:
                                color_det = (0,255,255)
                            else:
                                color_det = (0,0,255)
                            img_2_3 = cv2.rectangle(img_2_3, (int(obj_det.box2d[0]), int(obj_det.box2d[1])),
                                                (int(obj_det.box2d[2]), int(obj_det.box2d[3])), color=color_det, thickness=3)
                            img_2_3 = draw_boxcube(img_2_3, obj_det)
                            cv2.putText(img_2_3, '{}'.format(obj_det.type), tuple(obj_det.box2d[0:2].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 
                                                   1, (255, 100, 0), 2, cv2.LINE_AA)
                            # print(box_size)
                            cv2.putText(img_2_3, '{}'.format(int(box_size)), tuple(obj_det.box2d[[0,3]].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 
                                                   1, (255, 100, 0), 2, cv2.LINE_AA)  
                # INFER-GT img
                if  viz25D:
                    if vizGT:
                        cv2.putText(img_2_3, 'INFER', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 2)
                        img_2_3 = cv2.resize(img_2_3, (1200, int(img_2_3.shape[0] * 1200 / img_2_3.shape[1])))
                        img_2_2 = cv2.resize(img_2_2, (1200, int(img_2_2.shape[0] * 1200 / img_2_2.shape[1])))
                        img_array_2 = np.concatenate((img_2_2, img_2_3), axis=1)
                        cv2.imwrite(os.path.join(self.save_dir, "25D", filename.split('/')[-1]), img_array_2)
                    else:
                        img_2_3 = cv2.resize(img_2_3, (1200, int(img_2_3.shape[0] * 1200 / img_2_3.shape[1])))
                        cv2.imwrite(os.path.join(self.save_dir, "25D", filename.split('/')[-1]), img_2_3)
                if viz3D:
                    img3 = cv2.resize(img,(800,int(img.shape[0]*800/img.shape[1])))
                    img2 = cv2.resize(img2,(800,int(img2.shape[0]*800/img2.shape[1])))
                    img_array = np.concatenate((img3, img2), axis=0) 
                    img4 = cv2.resize(img4,(int(img4.shape[1]/img4.shape[0]*img_array.shape[0]),img_array.shape[0]))
                    img_array = np.concatenate((img4, img_array), axis=1) 
                    cv2.imwrite(cv2.imwrite(os.path.join(self.save_dir, "3D", filename.split('/')[-1]),img_array))
                    
        cal_metrics = True
        if cal_metrics:
            print("="*20)
            print("RESULTS:")
            if eval3D:
                tp_list_new =sum_list(tp_list, channel, x_dim)
                fn_list_new = sum_list(fn_list,channel, x_dim)
                
                depth_abs_list_new = np.round(sum_list(depth_abs_list, channel, x_dim),1)
                depth_list_new = np.round(sum_list(depth_list, channel, x_dim),1)
                gt_num_list_new = sum_list(gt_num_list, channel, x_dim)
                pre_num_list_new = sum_list(pre_num_list, channel, x_dim)
                pre_iou_list_new = sum_list(pre_iou_list, channel, x_dim)
                # print("gt: ", gt_num_list_new)
                
                # avoid divison by 0
                depth_list_new_ = np.array(depth_list_new)
                gt_num_list_new_= np.array(gt_num_list_new)
                pre_num_list_new_ = np.array(pre_num_list_new)
                
                # avoid divided by zero
                depth_list_new_[depth_list_new_ == 0] += 1
                gt_num_list_new_[gt_num_list_new_ == 0] += 1
                pre_num_list_new_[pre_num_list_new_ == 0] += 1
                
                # calculate error rate
                depth_error_list = np.round(np.array(depth_abs_list_new)/(depth_list_new_), 3) # 
                miss_rate = np.round(np.array(fn_list_new)/(gt_num_list_new_), 3) #
                false_rate = np.round(np.array(pre_iou_list_new) / (pre_num_list_new_), 3)
                
                gt_depth = np.round(sum_list(gt_depth,channel, x_dim),3)
                # print(gt_depth)
                absolute_error=np.round(sum_list(absolute_error,channel, x_dim),3)
                # print(absolute_error)
                gt_depth[gt_depth==0]= 1000000
                
                ### calculate error
                depth_error=np.round(np.array(absolute_error)/np.array(gt_depth),3) 
                # print('shape_error',mean_list(shape_error,channel))
                shape_error = mean_list(shape_error, channel, x_dim)
                yaw_error = mean_list(yaw_error, channel, x_dim)
                # print('gt_depth',gt_depth,'absolute_error',absolute_error)
                # print('error',error)
                # toxlxs_3d(savepath_3D,depth_error,shape_error,yaw_error,channel)
                
                
                # print("pre_iou_list_new: ", pre_iou_list_new)
                savepath_3D = os.path.join(self.save_dir, "eval_3D_results.xls")
                toxlxs_3d(savepath_3D,
                    depth_error_list, depth_abs_list_new, depth_list_new,
                    tp_list_new,fn_list_new,gt_num_list_new,miss_rate,
                    pre_iou_list_new,pre_num_list_new, false_rate, 
                    depth_error, shape_error, yaw_error, channel)
                
            if eval2D:    
                # 处理按类
                clses = list(cls_gt_num_list.keys())
                for key in clses:
                    if key not in list(cls_tp_list.keys()):
                        cls_tp_list[key] = 0
                    if key not in list(cls_fn_list.keys()):
                        cls_fn_list[key] = 0
                    if key not in list(cls_pre_iou_list.keys()):
                        cls_pre_iou_list[key] = 0
                    if key not in list(cls_pre_num_list.keys()):
                        cls_pre_num_list[key] = 0
                        
                print("类漏检无检测")
                print("类检出数量", cls_tp_list)
                print("类漏检总数",cls_fn_list) # 漏
                print("类误检总数",cls_pre_iou_list) # 误
                print("类真值总数",cls_gt_num_list)
                print("类预测数量",cls_pre_num_list)
                # toxlsx_cls(savepath_cls, cls_tp_list, cls_fn_list, cls_pre_iou_list, cls_pre_num_list, cls_gt_num_list)
                
                metrics_dir = os.path.join(self.save_dir, "metrics")
                os.makedirs(metrics_dir, exist_ok=True)
                with open(os.path.join(metrics_dir, "box_tp_list.txt"), 'w') as f:
                    f.write(json.dumps(box_tp_list))
                with open(os.path.join(metrics_dir, "box_fn_list.txt"), 'w') as f:
                    f.write(json.dumps(box_fn_list))
                with open(os.path.join(metrics_dir, "box_pre_iou_list.txt"), 'w') as f:
                    f.write(json.dumps(box_pre_iou_list))
                with open(os.path.join(metrics_dir, "box_gt_num_list.txt"), 'w') as f:
                    f.write(json.dumps(box_gt_num_list))
                with open(os.path.join(metrics_dir, "box_pre_num_list.txt"), 'w') as f:
                    f.write(json.dumps(box_pre_num_list))
                with open(os.path.join(metrics_dir, "box_error_list.txt"), 'w') as f:
                    f.write(json.dumps(box_error_list))
                parse_metrics(box_tp_list,box_fn_list, box_pre_iou_list, box_pre_num_list, box_gt_num_list, box_error_list, 
                            box_size_range, self.save_dir, name="eval_2D_results.xls", eval_cls=eval_cls)
            
            err_items = {"miss": miss_items, "false": false_items}
            with open(os.path.join(metrics_dir, "err_items.txt"), 'w', encoding='utf8') as f:
                f.write(json.dumps(err_items))
            
            if eval25D:
                savepath_cube = os.path.join(self.save_dir, "eval_cube_results.xls")
                pos,pos_mean,neg,neg_mean=seperate_POS_NEG(rear_abs)
                _,pos_std,_,neg_std=seperate_POS_NEG(rear_rlt)
                # np.save("./rear_abs.npy", rear_abs)
                # np.save("./rear_rlt.npy", rear_rlt)
                # print("rear_abs: ", rear_abs)
                print('目标总数为：',len(rear_abs))
                print('偏移量为正的个数',pos,'绝对误差',pos_mean,'相对误差',pos_std)
                print('偏移量为负的个数',neg,'绝对误差',neg_mean,'相对误差',neg_std)
                toxlxs_cube(savepath_cube,raw_data)
            
            if eval2D and video:
                merge_video(self.save_dir, os.path.join(self.save_dir, "25D"))
    # print(get_average(relative_error))
    # print(sum(absolute_error)/sum(gt_depth))
    def evaluate2D(self):
        pass
    def evaluate25D(self):
        pass
    def evaluate3D(self):
        pass
    
    def cal_3DIOU_Matrix_F(self, gt_box, pre_box,pre_iou_list, cls_pre_iou_list:dict=None, box_pre_iou_list:list=None, box_error_list:list=None, img_path=None):  
        ## 带深度信息误检率计算函数
        # 对于每个检测结果，匹配gt
        fn = 0
        false_flags = []
        cls_record = True if cls_pre_iou_list is not None else False
        box_record = True if box_pre_iou_list is not None else False
        for i in range(len(pre_box)):
            pre_xmin, pre_ymin, pre_xmax, pre_ymax = float(pre_box[i][0]), float(pre_box[i][1]), float(
                pre_box[i][2]), float(pre_box[i][3])
            pre_x0 = round(float(pre_box[i][4]), 3)
            pre_z0 = round(float(pre_box[i][5]), 3)
            obj_type = pre_box[i][-1]
            ## 区分车道
            # if -5.7 <= pre_x0 < -1.9:
            #     index_x = 0
            # elif -1.9 <= pre_x0 <= 1.9:
            #     index_x = 1
            # elif 1.9 < pre_x0 <= 5.7:
            #     index_x = 2
            if box_record:
                box_size = min(abs(pre_xmax-pre_xmin), abs(pre_ymax-pre_ymin))
                det_area = abs(pre_xmax-pre_xmin) * abs(pre_ymax-pre_ymin)
                
            ## 不分车道
            index_x = 0
            index_z = int((pre_z0 - pre_z0 % 10) / 10)
            # if index_z >= channel:
            #     index_z = channel-1
            # index_z = 0
            list_iou = []
            # list_gt = []
            # list_tp = []
            

            
            ## 若没有gt，代表全部误检测
            if gt_box == None:
                fn += 1
                false_flags.append(1)
                try:
                    pre_iou_list[index_x][index_z].append(1)
                except:
                    pre_iou_list[index_x][-1].append(1)
                
                if cls_record:
                    if obj_type not in cls_pre_iou_list.keys():
                        cls_pre_iou_list[obj_type] = 1
                    else:
                        cls_pre_iou_list[obj_type] += 1
                if box_record:
                    box_pre_iou_list.append([box_size, obj_type,  img_path]) # [length, obj_type, img_path, tp, det_area, gt_area]
                    # box_area_list.append([box_size, obj_type,  0, det_area, 0,img_path])
                continue
            
            # 匹配GT
            max_iou = -100
            
            for j in range(len(gt_box)):
                # for all labels, calculate iou
                gt_xmin, gt_ymin, gt_xmax, gt_ymax = float(gt_box[j][0]), float(gt_box[j][1]), float(gt_box[j][2]), float(gt_box[j][3])
                
                area_b = (pre_xmax - pre_xmin) * (pre_ymax - pre_ymin)
                gt_area = (gt_xmax - gt_xmin) * (gt_ymax - gt_ymin)
                w = min(gt_xmax, pre_xmax) - max(gt_xmin, pre_xmin)
                h = min(gt_ymax, pre_ymax) - max(gt_ymin, pre_ymin)
                if w < 0 or h < 0:
                    list_iou.append(0)
                else:
                    area = w * h
                    IOU = round(area / (area_b), 3)
                    list_iou.append(IOU)
                    if IOU > max_iou:
                        info = (gt_xmin, gt_ymin, gt_xmax, gt_ymax)
                        max_iou = IOU
            
            # calculate # of false detection.
            if list_iou == []:
                fn += 1
                false_flags.append(1)
                try:
                    pre_iou_list[index_x][index_z].append(1)
                except:
                    pre_iou_list[index_x][-1].append(1)
                if cls_record:
                    if obj_type not in cls_pre_iou_list.keys():
                        cls_pre_iou_list[obj_type] = 1
                    else:
                        cls_pre_iou_list[obj_type] += 1
                if box_record:
                    box_pre_iou_list.append([box_size, obj_type, img_path])
                    # box_area_list.append([box_size, obj_type,  0, det_area, 0,img_path])
            else:
                if max(list_iou) < 0.5:
                    fn += 1
                    false_flags.append(1)
                    try:
                        pre_iou_list[index_x][index_z].append(1)
                    except:
                        pre_iou_list[index_x][-1].append(1)
                    if cls_record:
                        if obj_type not in cls_pre_iou_list.keys():
                            cls_pre_iou_list[obj_type] = 1
                        else:
                            cls_pre_iou_list[obj_type] += 1
                    if box_record:
                        box_pre_iou_list.append([box_size, obj_type, img_path])
                else:
                    false_flags.append(0)
                    # max_index = np.argmax(list_iou)
                    # box_area_list.append([box_size, obj_type,  list_tp[max_index], det_area, list_gt[max_index],img_path])
                    # 像素计算误差
                    gt_xmin, gt_ymin, gt_xmax, gt_ymax = info
                    gt_x = (gt_xmax+gt_xmin) / 2
                    det_x = (pre_xmax+pre_xmin) / 2
                    gt_y = (gt_ymax + gt_ymin) / 2
                    det_y = (pre_ymax+pre_ymin) / 2
                    error_x = gt_x - det_x
                    error_y = gt_y - det_y
                    gt_w = abs(gt_xmax-gt_xmin)
                    gt_h = abs(gt_ymax-gt_ymin)
                    det_w = abs(pre_xmax-pre_xmin)
                    det_h = abs(pre_ymax-pre_ymin)
                    # print("err: ", gt_x, det_x, error_x, gt_w)
                    box_error_list.append([box_size, obj_type, error_x, error_y, gt_w, gt_h, det_w, det_h, img_path])
                
        # print("误检数量: {}".format(fn))
        return pre_iou_list, false_flags
    
    def cal_3DIOU_Matrix_M(self, gt_box,pre_box,depth_list,depth_abs_list,tp_list,fn_list,gt_num_list, \
                        cls_tp_list:dict=None, cls_fn_list:dict=None, cls_gt_num_list:dict=None,\
                            box_tp_list:list=None, box_fn_list:list=None, box_gt_num_list:list=None, img_path:str = None):  
        ## 带深度信息depth误差和漏检率计算函数
        # 对于每个gt 匹配检测结果
        fn = 0
        miss_flags = []
        cls_record = True if cls_gt_num_list is not None and cls_fn_list is not None and cls_tp_list is not None else False
        box_record = True if box_gt_num_list is not None and box_fn_list is not None and box_tp_list is not None else False
        for j in range(len(gt_box)):
            occ = gt_box[j][7]
            if occ < 0:
                miss_flags.append(-1)
                continue
            ## 区分车道
            list_iou = []
            abs_list =[]
            gt_xmin, gt_ymin, gt_xmax, gt_ymax = float(gt_box[j][0]), float(gt_box[j][1]), float(gt_box[j][2]), float(gt_box[j][3])
            
            gt_x0 =round(float(gt_box[j][4]),3)
            gt_z0 =round(float(gt_box[j][5]),2)
            obj_type = gt_box[j][6]
            
            # if -5.7 <= gt_x0 < -1.9:
            #     index_x = 0
            # elif -1.9 <= gt_x0 <= 1.9:
            #     index_x = 1
            # elif 1.9 < gt_x0 <= 5.7:
            #     index_x = 2
            
            ## 不分车道
            index_x = 0
            ## 区分深度
            index_z=int((gt_z0 - gt_z0 % 10) / 10)
            # index_z = 0
            
            ## 统计GT数量
            try:
                gt_num_list[index_x][index_z].append(1) # num of gt
            except:
                index_z = -1
                gt_num_list[index_x][index_z].append(1)
                  
            if cls_record:
                if obj_type not in cls_gt_num_list.keys():
                    cls_gt_num_list[obj_type] = 1
                else:
                    cls_gt_num_list[obj_type] += 1
            if box_record:
                box_size = min(abs(gt_xmax-gt_xmin), abs(gt_ymax-gt_ymin))
                gt_area = abs(gt_xmax-gt_xmin) * abs(gt_ymax-gt_ymin)
                box_gt_num_list.append([box_size, obj_type, img_path])
                        
            
            
            ## 若无预测结果，全部统计漏检
            if pre_box == None:
                fn += 1
                miss_flags.append(1)
                fn_list[index_x][index_z].append(1)
                if cls_record:
                    if obj_type not in cls_fn_list.keys():
                        cls_fn_list[obj_type] = 1
                    else:
                        cls_fn_list[obj_type] += 1
                if box_record:
                    box_fn_list.append([box_size,obj_type, gt_area, img_path])
                continue 
            
            ## 对每个检测结果计算IOU
            for i in range(len(pre_box)):
                pre_xmin, pre_ymin, pre_xmax, pre_ymax = float(pre_box[i][0]), float(pre_box[i][1]), float(pre_box[i][2]), float(pre_box[i][3])
                
                pre_z0 =round( float(pre_box[i][5]),3)
                area_b = (pre_xmax - pre_xmin) * (pre_ymax - pre_ymin)
                area_a = (gt_xmax - gt_xmin) * (gt_ymax - gt_ymin)
                w = min(gt_xmax, pre_xmax) - max(gt_xmin, pre_xmin)
                h = min(gt_ymax, pre_ymax) - max(gt_ymin, pre_ymin)
                if w<0 or h <0:
                    list_iou.append(0)
                else:
                    area = w*h
                    IOU = round(area / (area_b), 3)
                    list_iou.append(IOU)
                    abs_list.append(round(abs(pre_z0 - gt_z0), 3)) # depth error
            
            if list_iou == []:
                fn += 1
                fn_list[index_x][index_z].append(1)
                miss_flags.append(1)
                if obj_type not in cls_fn_list.keys():
                        cls_fn_list[obj_type] = 1
                else:
                        cls_fn_list[obj_type] += 1
                if box_record:
                    box_fn_list.append([box_size, obj_type, img_path])
            else:
                if max(list_iou) >= 0.5:

                    depth_abs_list[index_x][index_z].append(min(abs_list)) # depth error list
                    depth_list[index_x][index_z].append(gt_z0)
                    tp_list[index_x][index_z].append(1)
                    miss_flags.append(0)
                    if cls_record:
                        if obj_type not in cls_tp_list.keys():
                            cls_tp_list[obj_type] = 1
                        else:
                            cls_tp_list[obj_type] += 1
                    if box_record:
                        box_tp_list.append([box_size, obj_type, img_path])
                else:
                    if cls_record:
                        if obj_type not in cls_fn_list.keys():
                            cls_fn_list[obj_type] = 1
                        else:
                            cls_fn_list[obj_type] += 1
                    fn += 1
                    miss_flags.append(1)
                    fn_list[index_x][index_z].append(1)
                    if box_record:
                        box_fn_list.append([box_size, obj_type, img_path])
        # print("漏检数量: {}".format(fn))
        return tp_list,fn_list ,depth_abs_list,depth_list,gt_num_list, miss_flags

def setup_eval_args():
    parser = argparse.ArgumentParser(description="Evaluation params")
    parser.add_argument("--det", required=True, type=str, help="Path of inference results txt.")
    parser.add_argument("--gt", required=True, type=str, help="Path of gt labels txt.")
    parser.add_argument("--output", type=str, default="./eval_result/", help="Path of gt labels txt.")
    parser.add_argument("--image_txt", type=str, required=True, help="Path of txt including all images to evaluate.")
    parser.add_argument("--calib_path", type=str, default=None, help="Path of calib files")
    parser.add_argument("--vis_gt", help="Visualize 2.5d results.")
    parser.add_argument("--vis_video", help="Generate Video.")
    parser.add_argument("--eval_3d", help="Evaluate 3d metrics.")
    parser.add_argument("--eval_25d", help="Evaluate 2.5d results.")
    parser.add_argument("--eval_2d", help="Evaluate 2d results.")
    parser.add_argument("--channel", type=int, default=1, help="number of depth channel")
    parser.add_argument("--box_size_range", type=int, default=[32, 96], nargs="+", help="")
    parser.add_argument("--image_size", type=int, default=[1936, 1220], nargs=2, help="Original image size")
    parser.add_argument("--crop", type=int, default=[8, 28, 1928, 1220], nargs=4, help="Diagonal coordinates of crop box")
    parser.add_argument("--cls", type=str, default=None, nargs='+', help="Classes to evaluate")
    return parser
    
    

if __name__ == "__main__":
    
    args = setup_eval_args().parse_args()
    
    test_file = args.image_txt
    save_dir = args.output
    gt_path = args.gt
    det_path = args.det
    img_shape = tuple(args.image_size)
    crop_coor = list(args.crop)
    calib_path = args.calib_path
    
    eval2D = args.eval_2d
    eval3D = args.eval_3d
    eval25D = args.eval_25d
    vizGT = args.vis_gt
    video = args.vis_video
    box_size_range = args.box_size_range
    channel = args.channel
    eval_cls = args.cls
    
    
    # folder = "aiv_aggregate_3"
    # test_file = "/home/utopilot/workspace/infer/eval_txt/test.txt"
    # save_dir = "./eval_results/test/{}/".format(folder)
    
    # det_path = "/home/utopilot/workspace/infer/mono_infer_vgg/output/aiv_night/test_pd28/test_pd28_fc120_roi0/savetxt/"
    # gt_path = "/home/utopilot/workspace/infer/eval_txt/gt/"

    evaluator = Evaluator(test_file, save_dir, gt_path, det_path, 
                          img_shape, crop_coor, calib_path)
    evaluator.evaluate(eval2D, eval25D, eval3D, vizGT, video, box_size_range, channel, eval_cls)