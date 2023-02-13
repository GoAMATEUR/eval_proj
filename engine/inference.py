import logging
import pdb
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from utils import comm
from utils.timer import Timer, get_time_str
from collections import defaultdict
from data.datasets.evaluation import evaluate_python
from data.datasets.evaluation import generate_kitti_3d_detection

from .visualize_infer import show_image_with_boxes, show_image_with_boxes_test,show_image_with_2dboxes,show_image_with_3dboxes


def compute_2bbox_on_original_imgae(output,pad_size,calib,reisze_scale,img_w,img_h):
    pred_2dbox = output[:,2:6]
    pred_2dbox = pred_2dbox*reisze_scale
#     print(pred_2dbox)
#     pdb.set_trace()
    pred_2dbox[:,[0,2]] = torch.clamp(pred_2dbox[:,[0,2]], 0, img_w-1)
    pred_2dbox[:,[1,3]] = torch.clamp(pred_2dbox[:,[1,3]], 0, img_h-1)
#     print(pred_2dbox)
#     pdb.set_trace()
    output[:,2:6] = pred_2dbox
    output[:,11] = output[:,11]/calib.rescale_z
    return output

def compute_on_dataset(cfg,model, data_loader, device, predict_folder, timer=None, vis=False, 
                        eval_score_iou=False, eval_depth=False, eval_trunc_recall=False):
    
    model.eval()
    cpu_device = torch.device("cpu")
    dis_ious = defaultdict(list)
    depth_errors = defaultdict(list)

    differ_ious = []
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(data_loader)):
            images, targets, image_ids = batch["images"], batch["targets"], batch["img_ids"]
            images = images.to(device)

            # extract label data for visualize
            vis_target = targets[0]
            targets = [target.to(device) for target in targets]

            if timer:
                timer.tic()

            output, eval_utils, visualize_preds = model(images, targets)
            output = output.to(cpu_device)

            if timer:
                torch.cuda.synchronize()
                timer.toc()

            dis_iou = eval_utils['dis_ious']

            if dis_iou is not None:
                for key in dis_iou: dis_ious[key] += dis_iou[key].tolist()
            if cfg.MODEL.ONLY_TRAIN_2D :
                show_image_with_2dboxes(vis_target.get_field('ori_img'), output, visualize_preds,idx)
            else:
                show_image_with_3dboxes(vis_target.get_field('ori_img'), output,vis_target.get_field('calib'), visualize_preds,idx,predict_folder)
            # generate txt files for predicted objects
            output = compute_2bbox_on_original_imgae(output,vis_target.get_field('pad_size'),vis_target.get_field('calib'),3,3840,2160)
            predict_txt = image_ids[0] + '.txt'
            predict_txt = os.path.join(predict_folder, predict_txt)
            generate_kitti_3d_detection(output, predict_txt)


    
    # disentangling IoU
    for key, value in dis_ious.items():
        mean_iou = sum(value) / len(value)
        dis_ious[key] = mean_iou

    return dis_ious

def inference(cfg,
        model,
        data_loader,
        dataset_name,
        eval_types=("detections",),
        device="cuda",
        output_folder=None,
        metrics=['R40'],
        vis=False,
        eval_score_iou=False,
):
    device = torch.device(device)
    num_devices = comm.get_world_size()
    logger = logging.getLogger("monodetect.inference")
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
    predict_folder = os.path.join(output_folder, 'data')
    os.makedirs(predict_folder, exist_ok=True)

    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()

    dis_ious = compute_on_dataset(cfg,model, data_loader, device, predict_folder, 
                                inference_timer, vis, eval_score_iou)
    print("compute on dataset")
    # # comm.synchronize()
    # print("compute on dataset later")
    # for key, value in dis_ious.items():
    #     logger.info("{}, MEAN IOU = {:.4f}".format(key, value))

    # total_time = total_timer.toc()
    # total_time_str = get_time_str(total_time)
    # logger.info(
    #     "Total run time: {} ({} s / img per device, on {} devices)".format(
    #         total_time_str, total_time * num_devices / len(dataset), num_devices
    #     )
    # )
    # total_infer_time = get_time_str(inference_timer.total_time)
    # logger.info(
    #     "Model inference time: {} ({} s / img per device, on {} devices)".format(
    #         total_infer_time,
    #         inference_timer.total_time * num_devices / len(dataset),
    #         num_devices,
    #     )
    # )
    # if not comm.is_main_process():
    #     return None, None, None

    # logger.info('Finishing generating predictions, start evaluating ...')
    # ret_dicts = []
    
    # for metric in metrics:
    #     result, ret_dict = evaluate_python(label_path=dataset.label_dir, 
    #                                     result_path=predict_folder,
    #                                     label_split_file=dataset.imageset_txt,
    #                                     current_class=dataset.classes,
    #                                     metric=metric)

    #     logger.info('metric = {}'.format(metric))
    #     logger.info('\n' + result)

    #     ret_dicts.append(ret_dict)
    result = None
    ret_dicts = {}
    return ret_dicts, result, dis_ious

def inference_all_depths(cfg,
        model,
        data_loader,
        dataset_name,
        eval_types=("detections",),
        device="cuda",
        output_folder=None,
        vis=False,
        eval_score_iou=False,
        metrics=['R40', 'R11'],
):
    metrics = ['R40']
    inference_timer = None
    device = torch.device(device)
    logger = logging.getLogger("monodetect.inference")
    dataset = data_loader.dataset
    predict_folder = os.path.join(output_folder, 'eval_all_depths')
    os.makedirs(predict_folder, exist_ok=True)
    
    # all methods for solving depths
    class_threshs = [[0.7], [0.5], [0.5]]
    important_key = '{}_3d_{:.2f}/moderate'.format('Car', 0.7)
    important_classes = ['Car']

    eval_depth_methods = ['oracle', 'hard', 'soft', 'mean', 'direct', 'keypoints_center', 'keypoints_02', 'keypoints_13']

    eval_depth_dicts = []
    for depth_method in eval_depth_methods:
        logger.info("evaluation with depth method: {}".format(depth_method))
        method_predict_folder = os.path.join(predict_folder, depth_method)
        os.makedirs(method_predict_folder, exist_ok=True)

        # remove all previous predictions
        for file in os.listdir(method_predict_folder):
            os.remove(os.path.join(method_predict_folder, file))

        model.heads.post_processor.output_depth = depth_method
        dis_ious = compute_on_dataset(model, data_loader, device, method_predict_folder, 
                                    inference_timer, vis, eval_score_iou)
        result, ret_dict = evaluate_python(label_path=dataset.label_dir, 
                                        result_path=method_predict_folder,
                                        label_split_file=dataset.imageset_txt,
                                        current_class=dataset.classes,
                                        metric='R40')
        
        eval_depth_dicts.append(ret_dict)

    for cls_idx, cls in enumerate(important_classes):
        cls_thresh = class_threshs[cls_idx]
        for thresh in cls_thresh:
            logger.info('{} AP@{:.2f}, {:.2f}:'.format(cls, thresh, thresh))
            sort_metric = []
            for depth_method, eval_depth_dict in zip(eval_depth_methods, eval_depth_dicts):
                logger.info('bev/3d AP, method {}:'.format(depth_method))
                logger.info('{:.4f}/{:.4f}, {:.4f}/{:.4f}, {:.4f}/{:.4f}'.format(eval_depth_dict['{}_bev_{:.2f}/easy'.format(cls, thresh)], 
                        eval_depth_dict['{}_3d_{:.2f}/easy'.format(cls, thresh)], eval_depth_dict['{}_bev_{:.2f}/moderate'.format(cls, thresh)], 
                        eval_depth_dict['{}_3d_{:.2f}/moderate'.format(cls, thresh)], eval_depth_dict['{}_bev_{:.2f}/hard'.format(cls, thresh)],
                        eval_depth_dict['{}_3d_{:.2f}/hard'.format(cls, thresh)]))

                sort_metric.append(eval_depth_dict['{}_3d_{:.2f}/moderate'.format(cls, thresh)])

            sort_metric = np.array(sort_metric)
            sort_idxs = np.argsort(-sort_metric)
            join_str = ' > '
            sort_str = join_str.join([eval_depth_methods[idx] for idx in sort_idxs])
            logger.info('Cls {}, Thresh {}, Sort: '.format(cls, thresh) + sort_str)

    return None, None, None
