import kitti_common as kitti
import os ,cv2 
from kitti_utils import draw_projected_box3d, draw_boxcube, draw_bev_box3d, Calibration, read_label, \
    calculate_depth_error, get_average, sum_list, toxlxs_3d, mean_list, mean_list_cube, calculate_cube_error, \
    calculate_cube_error_onlyrear, seperate_POS_NEG, toxlxs_cube
import numpy as np   
from shapely.geometry import Polygon
import sys
import operator

colors = {"VAN" : (0, 0, 255),"CAR": (0, 255, 0),"PD": (255, 0, 0),"Rider": (0, 122, 255),
          "Three": (0, 122, 122),"TRUCK":(122, 122, 255),"BUS": (122, 122, 0),"SPECIALCAR":(122, 0, 122),"trailerback":(0, 97, 255)}
def get_label_annos_gen(label_folder,det_folder,xml_path= []):
    only_genGT =False
    annos_det = []
    annos_gt = []
    label_filenames = os.listdir(label_folder)
    label_filenames.sort()
    image_files = []
    if xml_path != []:
        label_filenames =[]
        for line in xml_path:
            base_name = os.path.basename(line)
            image_files.append(line.replace('Label','Image').replace('xml','jpg'))
            label_filenames.append(os.path.join(base_name.replace('.xml','.txt')))
    filename = []
    labels =[]
    labels_det =[]
    
    for idx in label_filenames:
        label_filename_det = os.path.join(det_folder, idx.split('/')[-1])
        label_filename_gt = os.path.join(label_folder, idx.split('/')[-1])
        if only_genGT:
            if os.path.exists(label_filename_gt):
                label = read_label(label_filename_gt)
                annos_gt.append(kitti.get_label_anno(label_filename_gt))
                filename.append(idx)
                labels.append(label)
        else:
            if os.path.exists(label_filename_det) and os.path.exists(label_filename_gt):
                label = read_label(label_filename_gt)
                label_det = read_label(label_filename_det)
                annos_gt.append(kitti.get_label_anno(label_filename_gt))
                annos_det.append(kitti.get_label_anno(label_filename_det))
                filename.append(idx)
                labels.append(label)
                labels_det.append(label_det)

    return annos_gt,annos_det,image_files,labels,labels_det

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

def vis_bev_project3d(labels,label_det,imgpaths,calibPaths,save_dir,savepath,channel=30,vis_maxus=True,cal_maxus=True):
    # print(imgpaths)
    # gt_depth= [[[] for i in range(channel) ] for j in range(3)]
    # gt_x,gt_y,error_x,error_y=[],[],[],[]
    gt_x = [[] for i in range(4)]
    gt_y = [[] for i in range(4)]
    error_x = [[] for i in range(4)]
    error_y = [[] for i in range(4)]
    rear_abs,rear_rlt,raw_data=[],[],[]
    
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter(save_dir+'../0802_bev.avi', fourcc, 2.0, (1700,900))

    
    for j,(objs,objs_det,imgpath,calibrationPath) in enumerate(zip(labels,label_det,imgpaths,calibPaths)):
        # imgpath='/workspace/honghu-1-day/Image0000007/truck22_1643082169944827.jpg'
        print('id: ',j, 'img:',imgpath)
        # print(objs[0].box2d)
        filename = os.path.basename(imgpath)
        if not os.path.exists(imgpath):
            continue
        img_2 = cv2.imread(imgpath)
        img_2_2 = img_2.copy()
        img_2_3 = img_2.copy()

        #match GT and det
        matched_id=1
        for j, obj in enumerate(objs):
            #正样本没标3d信息的不做匹配
            # if obj.h == 0:
            #     continue
            #iou赋值0，距离赋值9999
            list_iou = [0]*len(objs_det)
            abs_list =[9999]*len(objs_det)
            for i, obj_det in enumerate(objs_det):
                if obj_det.id == 0:#取未匹配过的
                    #如果真值是车，就将检测到的车跟他一一计算IOU
                    if (obj.type == 'VAN' or obj.type == 'CAR' or obj.type == 'Three'or obj.type =='TRUCK' or obj.type =='BUS'or obj.type =='SPECIALCAR' or obj.type == 'trailerback') and obj_det.type =='CAR':
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
                    # #如果真值是人，就将检测到的人跟他一一计算IOU
                    # if (obj.type == 'PD' or obj.type =='Rider') and obj_det.type =='PD':
                    #     area_b = (obj_det.xmax - obj_det.xmin) * (obj_det.ymax - obj_det.ymin)
                    #     area_a = (obj.xmax - obj.xmin) * (obj.ymax - obj.ymin)
                    #     w = min(obj.xmax, obj_det.xmax) - max(obj.xmin, obj_det.xmin)
                    #     h = min(obj.ymax, obj_det.ymax) - max(obj.ymin, obj_det.ymin)
                    #     if w<0 or h <0:
                    #         pass
                    #     else:
                    #         area = w*h
                    #         IOU = round(area / (area_b+area_a-area), 3)
                    #         list_iou[i]=IOU
                    #         abs_list[i]=round(abs(obj_det.t[2]- obj.t[2]), 3)
            # print(list_iou,abs_list)
            #算法选择匹配上的那一对
            if list_iou!=[] and max(list_iou) >= 0.5:
                # min_index, min_number = min(enumerate(abs_list), key=operator.itemgetter(1))#取深度最小为匹配上
                max_index, max_number = max(enumerate(list_iou), key=operator.itemgetter(1))#取iou最大为匹配上
                obj.id=matched_id
                objs_det[max_index].id=matched_id
                matched_id+=1

        #calculate single pic
        # gt_x_pic,gt_y_pic,error_x_pic,error_y_pic = calculate_cube_error(objs,objs_det)
        rear_abs_pic,rear_rlt_pic,raw_data_pic=calculate_cube_error_onlyrear(objs,objs_det)
        #add to all pic
        rear_abs.extend(rear_abs_pic)
        rear_rlt.extend(rear_rlt_pic)
        raw_data.extend(raw_data_pic)
        # print(rear_abs,rear_rlt,raw_data)
        # for direction in range(4):
        #     error_x[direction].extend(error_x_pic[direction])
        #     error_y[direction].extend(error_y_pic[direction])
        # print(error_x,error_y)


        if vis_maxus:
            for i, obj in enumerate(objs):
                if obj.cal:
                    img_2_2 = cv2.rectangle(img_2_2, (int(obj.box2d[0]), int(obj.box2d[1])),
                                         (int(obj.box2d[2]), int(obj.box2d[3])), color=(0, 0, 255), thickness=5)
                    img_2_2 = draw_boxcube(img_2_2, obj)
                    cv2.putText(img_2_2, 'GT', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

            for i, obj_det in enumerate(objs_det):
                if obj_det.cal:
                    img_2_3 = cv2.rectangle(img_2_3, (int(obj_det.box2d[0]), int(obj_det.box2d[1])),
                                         (int(obj_det.box2d[2]), int(obj_det.box2d[3])), color=(0, 255, 255),
                                         thickness=5)
                    img_2_3 = draw_boxcube(img_2_3, obj_det)
                    cv2.putText(img_2_3, 'INFER', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)

            img_2_3 = cv2.resize(img_2_3, (1200, int(img_2_3.shape[0] * 1200 / img_2_3.shape[1])))
            img_2_2 = cv2.resize(img_2_2, (1200, int(img_2_2.shape[0] * 1200 / img_2_2.shape[1])))
            img_array_2 = np.concatenate((img_2_2, img_2_3), axis=1)

            cv2.imwrite(save_dir + filename.split('/')[-1].replace('.jpg', '_cubebox.jpg'), img_array_2)

    if cal_maxus:
        # x = mean_list_cube(error_x)
        # y = mean_list_cube(error_y)
        pos,pos_mean,neg,neg_mean=seperate_POS_NEG(rear_abs)
        np.save("./rear_abs/rear_abs.npy", rear_abs)
        _,pos_std,_,neg_std=seperate_POS_NEG(rear_rlt)
        np.save("./rear_abs/rear_rlt.npy", rear_rlt)
        print('目标总数为：',len(rear_abs))
        print('偏移量为正的个数',pos,'绝对误差',pos_mean,'相对误差',pos_std)
        print('偏移量为负的个数',neg,'绝对误差',neg_mean,'相对误差',neg_std)

        toxlxs_cube(savepath,raw_data)
  

if __name__ == "__main__":
    test_file = "/home/uto/Documents/cube-evaluate/test30_list.txt"
    with open(test_file, 'r') as fsets:
        xml_files_list = list(map(lambda x: x.strip(), fsets.readlines()))
    print(xml_files_list)

    vis_maxus =True
    cal_maxus=True

    det_path = "/home/uto/Documents/cube-evaluate/infer_results-1024-2/infer_results/"
    gt_path = '/home/uto/Documents/cube-evaluate/test30/'
    gt_annos, dt_annos ,filename,labels,label_det = get_label_annos_gen(gt_path,det_path,xml_files_list)
    print(len(gt_annos)," : ", len(dt_annos))
    save_dir = "/home/uto/Documents/cube-evaluate/pic/"
    xls_path= '/home/uto/Documents/cube-evaluate/' + 'cube' + '.xls'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    vis_bev_project3d(labels,label_det,filename,xml_files_list,save_dir,xls_path,vis_maxus=vis_maxus,cal_maxus=cal_maxus,channel=22)
   
    








