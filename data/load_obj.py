import utils.kitti_common as kitti
import os ,cv2 
from eval.kitti_utils import draw_boxcube, calculate_cube_error_onlyrear, draw_projected_box3d, draw_bev_box3d,Calibration,read_label,\
    calculate_depth_error,get_average,sum_list,toxlxs_3d,mean_list, toxlxs_cube, seperate_POS_NEG

def get_label_annos_gen(label_folder,det_folder,test_path, only_genGT =True):
    annos_det = []
    annos_gt = []
    # label_filenames = os.listdir(label_folder)
    # label_filenames.sort()
    image_files = []
    label_files = []
    dt_files = []
    if test_path != []:
        label_filenames =[]
        for line in test_path:
            img_name = os.path.basename(line)
            image_files.append(line)
            #/home/utopilot/workspace/infer/demo_zhanting/donghai_1/rl/image_002230.jpg
            a = line.split("/")
            label_file = label_folder + "{}/{}/0/{}".format(a[-3], a[-2], a[-1].replace("jpg", "txt"))
            # print(line)
            # print(label_file)
            
            label_files.append(label_file)
            dt_file = det_folder + "{}/{}/0/{}".format(a[-3], a[-2], a[-1].replace("jpg", "txt"))
            # print(dt_file)
            # print("==")
            dt_files.append(dt_file)

    filename = []
    labels =[]
    labels_det =[]
    
    
    
    for i, (label_filename_det, label_filename_gt) in enumerate(zip(dt_files, label_files)):
        if only_genGT:
            if os.path.exists(label_filename_gt):
                label = read_label(label_filename_gt)
                annos_gt.append(kitti.get_label_anno(label_filename_gt))
                # filename.append(idx)
                labels.append(label)
            else:
                labels.append(None)
        else:
            if os.path.exists(label_filename_det) and os.path.exists(label_filename_gt):
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
    return annos_gt,annos_det,image_files,labels,labels_det

