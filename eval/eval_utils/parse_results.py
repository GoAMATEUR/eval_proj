import json
import os
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append("/home/utopilot/workspace/infer/mono_infer_vgg/")
import xlwt
import cv2
from tqdm import tqdm
class TypeStats:
    def __init__(self, stats_type, tp_list, fn_list, pre_iou_list, pre_num_list, gt_num_list, box_error_list):
        self.type = stats_type
        self.tp_list = tp_list
        self.fn_list = fn_list
        self.pre_iou_list = pre_iou_list
        self.pre_num_list = pre_num_list
        self.gt_num_list = gt_num_list
        self.box_error_list = box_error_list
        
class EvalStats(object):
    def __init__(self, result_dict, dist_splits=["12", "25", "50", "100", "200","500"]):
        self.result_dict = result_dict
        self.dist_splits = dist_splits
        self._init_splits()
        self.book = None
        self.result_cls = dict()
        self.err_cls = dict()
    
    def _init_book(self):
        self.book = xlwt.Workbook(encoding='utf-8',style_compression=0)
    
    def _init_splits(self):
        # self.dist_splits = [f"{str(i)}-" for i in self.dist_splits]
        # self.dist_splits.append(">{}".format(self.dist_splits[-1]))
        for i in range(len(self.dist_splits)-1):
            self.dist_splits[i] = f"{self.dist_splits[i]}-{self.dist_splits[i+1]}"
        self.dist_splits[-1] = f">{self.dist_splits[-1]}"
    
    def to_xls(self, savepath, eval_cls:list=None):
        """
            save all stats
            eval_cls: List of classes to evaluate
        """
        if self.book is None:
            self._init_book()
        if eval_cls == None:
            eval_cls = list(self.result_dict.keys())
        
        self.to_xls_dist(eval_cls)
        for cls in (eval_cls):
            self.to_xls_cls_dist(cls)
        self.to_xls_cls(eval_cls)
        self.save(savepath)
    
    def to_xls_cls_dist(self, cls):
        ## 单独类按大小分类
        assert cls in list(self.result_dict.keys()),\
            "KEY ERROR."
        data = self.result_dict[cls]
        if self.book is None:
            self.init_book()
            
        sheet = self.book.add_sheet(str(data.type),cell_overwrite_ok=True)
        sheet.write(0, 0, "{}测评数据".format(cls))
        sheet.write(2, 0, "检出数量")
        sheet.write(3, 0, "漏检总数")
        sheet.write(4, 0, "真值总数")
        sheet.write(5, 0, "漏检率")
        sheet.write(6, 0, "误检总数")
        sheet.write(7, 0, "预测总数")
        sheet.write(8, 0, "误检率")
        sheet.write(9, 0, "Precision")
        sheet.write(10, 0, "Recall")
        sheet.write(11, 0, "mean_x")
        sheet.write(12, 0, "mean_y")
        sheet.write(13, 0, "mean_w")
        sheet.write(14, 0, "mean_h")
       
        tp_sum = 0
        fn_sum = 0
        pre_iou_sum = 0
        pre_sum = 0
        gt_sum = 0
        error_list = []
        for i in data.box_error_list:

            if i != []:
                error_list.append(np.mean(np.abs(i), axis=0).tolist())
            else:
                error_list.append([])
            
        
        # print(error_list)
        for i in range(len(self.dist_splits)):
            sheet.write(1, i+1, str(self.dist_splits[i]))
            sheet.write(2, i+1, str(data.tp_list[i]))
            sheet.write(3, i+1, str(data.fn_list[i]))
            sheet.write(4, i+1, str(data.gt_num_list[i]))
            if data.gt_num_list[i] > 0:
                missrate = round(data.fn_list[i] / data.gt_num_list[i], 5)
            else:
                missrate = 1. if data.fn_list[i] > 0. else 0.
            sheet.write(5, i+1, str(missrate))
            sheet.write(6, i+1, str(data.pre_iou_list[i]))
            sheet.write(7, i+1, str(data.pre_num_list[i]))
            if data.pre_num_list[i] > 0:
                falserate = round(data.pre_iou_list[i] / data.pre_num_list[i], 5)
            else:
                falserate = 1. if data.pre_iou_list[i] > 0. else 0.
            
            sheet.write(8, i+1, str(falserate))
            
            fp = data.pre_iou_list[i]
            fn = data.fn_list[i]
            tp = data.tp_list[i]
            if tp+fp > 0:
                precision  = round(tp / (tp+fp), 5)
            else:
                precision = 1. if tp > 0. else 0.
            sheet.write(9, i+1, str(precision))
            if tp+fn > 0:
                recall  = round(tp / (tp+fn), 5)
            else:
                recall = 1. if tp > 0. else 0.
            sheet.write(10, i+1, str(recall))
             # error_x, error_y, gt_w, gt_h, det_w, det_h, img_path
            err_dat = error_list[i]
            if err_dat != []:
                err_x = err_dat[0]
                err_y = err_dat[1]
                
                gt_w = err_dat[2]
                gt_h = err_dat[3]
                det_w = err_dat[4]
                det_h = err_dat[5]
                err_w = abs(gt_w - det_w)
                err_h = abs(gt_h - det_h)
                sheet.write(11, i+1, "{}/{}".format(round(err_x, 3), round(err_x/gt_w, 3)))
                sheet.write(12, i+1, "{}/{}".format(round(err_y, 3), round(err_y/gt_h, 3)))
                sheet.write(13, i+1, "{}/{}".format(round(err_w,3 ), round(err_w/gt_w, 3)))
                sheet.write(14, i+1, "{}/{}".format(round(err_h,3), round(err_h/gt_w, 3)))
            else:
                sheet.write(11, i+1, "{}/{}".format(0, 0))
                sheet.write(12, i+1, "{}/{}".format(0, 0))
                sheet.write(13, i+1, "{}/{}".format(0, 0))
                sheet.write(14, i+1, "{}/{}".format(0, 0))
            
            
            tp_sum += data.tp_list[i]
            fn_sum += data.fn_list[i]
            pre_iou_sum += data.pre_iou_list[i]
            pre_sum += data.pre_num_list[i]
            gt_sum += data.gt_num_list[i]
            
        self.result_cls[cls] = [tp_sum, fn_sum, pre_iou_sum, pre_sum, gt_sum]
        sheet.write(1, i+2,"合计")
        sheet.write(2, i+2, str(tp_sum))
        sheet.write(3, i+2, str(fn_sum))
        sheet.write(4, i+2, str(gt_sum))
        if gt_sum > 0:
            missrate = round(fn_sum / gt_sum, 5)
        else:
            missrate = 1. if data.fn_list[i] > 0. else 0.
        sheet.write(5, i+2, str(missrate))
        sheet.write(6, i+2, str(pre_iou_sum))
        sheet.write(7, i+2, str(pre_sum))
        if pre_sum > 0:
                falserate = round(pre_iou_sum / pre_sum, 5)
        else:
            falserate = 1. if pre_iou_sum > 0. else 0.
        sheet.write(8, i+2, str(falserate))
        if tp_sum+pre_iou_sum > 0:
            precision  = round(tp_sum / (tp_sum+pre_iou_sum), 5)
        else:
            precision = 1. if tp_sum > 0. else 0.
        sheet.write(9, i+2, str(precision))
        if tp_sum+fn_sum > 0:
            recall  = round(tp_sum / (tp_sum+fn_sum), 5)
        else:
            recall = 1. if tp_sum > 0. else 0.
        sheet.write(10, i+2, str(recall))
        
    def to_xls_cls(self, eval_cls):
        ## 所有评测数据按类分类

        if self.book is None:
            self.init_book()
        sheet = self.book.add_sheet("ALL",cell_overwrite_ok=True)
        sheet.write(0, 0, "类别测评数据")
        sheet.write(2, 0, "检出数量")
        sheet.write(3, 0, "漏检总数")
        sheet.write(4, 0, "真值总数")
        sheet.write(5, 0, "漏检率")
        sheet.write(6, 0, "误检总数")
        sheet.write(7, 0, "预测总数")
        sheet.write(8, 0, "误检率")
        sheet.write(9, 0, "Precision")
        sheet.write(10, 0, "Recall")
        sheet.write(11, 0, "mean_x")
        sheet.write(12, 0, "mean_y")
        sheet.write(13, 0, "mean_w")
        sheet.write(14, 0, "mean_h")
        
        tp_sum = 0
        fn_sum = 0
        pre_iou_sum = 0
        pre_sum = 0
        gt_sum = 0
        err_all = []
        for i in range(len(eval_cls)):
            cls = eval_cls[i]
            data = self.result_cls[cls]
            # print(data)
            obj = self.result_dict[cls]
            sheet.write(1, i+1, cls)
            tp = data[0]
            fn = data[1]
            pre_iou = data[2]
            fp = pre_iou
            pre = data[3]
            gt = data[4]
            sheet.write(2, i+1, str(tp))
            sheet.write(3, i+1, str(fn))
            sheet.write(4, i+1, str(gt))
            if gt > 0:
                missrate = round(fn / gt, 5)
            else:
                missrate = 1. if fn > 0. else 0.
            sheet.write(5, i+1, str(missrate))
            sheet.write(6, i+1, str(pre_iou))
            sheet.write(7, i+1, str(pre))
            if pre > 0:
                    falserate = round(pre_iou / pre, 5)
            else:
                falserate = 1. if pre_iou > 0. else 0.
            sheet.write(8, i+1, str(falserate))
            
            if tp+fp > 0:
                precision  = round(tp / (tp+fp), 5)
            else:
                precision = 1. if tp > 0. else 0.
            sheet.write(9, i+1, str(precision))
            if tp+fn > 0:
                recall  = round(tp / (tp+fn), 5)
            else:
                recall = 1. if tp > 0. else 0.
            sheet.write(10, i+1, str(recall))
            
            
            err_cls_all = []
            
            for err in obj.box_error_list:
                err_cls_all.extend(err)
            err_all.extend(err_cls_all)
            
            
            cls_sheet = self.book.get_sheet(cls)
            if err_cls_all != []:
                err_dat = np.mean(np.abs(err_cls_all), axis=0).tolist()
                err_x = err_dat[0]
                err_y = err_dat[1]
                
                gt_w = err_dat[2]
                gt_h = err_dat[3]
                det_w = err_dat[4]
                det_h = err_dat[5]
                err_w = abs(gt_w - det_w)
                err_h = abs(gt_h - det_h)
                sheet.write(11, i+1, "{}/{}".format(round(err_x, 3), round(err_x/gt_w, 3)))
                sheet.write(12, i+1, "{}/{}".format(round(err_y, 3), round(err_y/gt_h, 3)))
                sheet.write(13, i+1, "{}/{}".format(round(err_w,3 ), round(err_w/gt_w, 3)))
                sheet.write(14, i+1, "{}/{}".format(round(err_h,3), round(err_h/gt_w, 3)))
                
                cls_sheet.write(11, 3, "{}/{}".format(round(err_x, 3), round(err_x/gt_w, 3)))
                cls_sheet.write(12, 3, "{}/{}".format(round(err_y, 3), round(err_y/gt_h, 3)))
                cls_sheet.write(13, 3, "{}/{}".format(round(err_w,3 ), round(err_w/gt_w, 3)))
                cls_sheet.write(14, 3, "{}/{}".format(round(err_h,3), round(err_h/gt_w, 3)))
            else:
                sheet.write(11, i+1, "{}/{}".format(0, 0))
                sheet.write(12, i+1, "{}/{}".format(0, 0))
                sheet.write(13, i+1, "{}/{}".format(0, 0))
                sheet.write(14, i+1, "{}/{}".format(0, 0))
            
            tp_sum += tp
            fn_sum += fn
            pre_iou_sum += pre_iou
            pre_sum += pre
            gt_sum += gt
        sheet.write(1, i+2,"合计")
        sheet.write(2, i+2, str(tp_sum))
        sheet.write(3, i+2, str(fn_sum))
        sheet.write(4, i+2, str(gt_sum))
        if gt_sum > 0:
            missrate = round(fn_sum / gt_sum, 5)
        else:
            missrate = 1. if data.fn_list[i] > 0. else 0.
        sheet.write(5, i+2, str(missrate))
        sheet.write(6, i+2, str(pre_iou_sum))
        sheet.write(7, i+2, str(pre_sum))
        if pre_sum > 0:
                falserate = round(pre_iou_sum / pre_sum, 5)
        else:
            falserate = 1. if pre_iou_sum > 0. else 0.
        sheet.write(8, i+2, str(falserate))
        if tp_sum+pre_iou_sum > 0:
            precision  = round(tp_sum / (tp_sum+pre_iou_sum), 5)
        else:
            precision = 1. if tp_sum > 0. else 0.
        sheet.write(9, i+2, str(precision))
        if tp_sum+fn_sum > 0:
            recall  = round(tp_sum / (tp_sum+fn_sum), 5)
        else:
            recall = 1. if tp_sum > 0. else 0.
        sheet.write(10, i+2, str(recall))
        

        err_dat = np.mean(np.abs(err_all), axis=0).tolist()
        if err_dat != []:
            err_x = err_dat[0]
            err_y = err_dat[1]
            
            gt_w = err_dat[2]
            gt_h = err_dat[3]
            det_w = err_dat[4]
            det_h = err_dat[5]
            err_w = abs(gt_w - det_w)
            err_h = abs(gt_h - det_h)
            sheet.write(11, i+2, "{}/{}".format(round(err_x, 3), round(err_x/gt_w, 3)))
            sheet.write(12, i+2, "{}/{}".format(round(err_y, 3), round(err_y/gt_h, 3)))
            sheet.write(13, i+2, "{}/{}".format(round(err_w,3 ), round(err_w/gt_w, 3)))
            sheet.write(14, i+2, "{}/{}".format(round(err_h,3), round(err_h/gt_w, 3)))
        else:
            sheet.write(11, i+2, "{}/{}".format(0, 0))
            sheet.write(12, i+2, "{}/{}".format(0, 0))
            sheet.write(13, i+2, "{}/{}".format(0, 0))
            sheet.write(14, i+2, "{}/{}".format(0, 0))
            
    def to_xls_dist(self, eval_cls):
        ## 所有数据根据box大小统计
        # if eval_cls is None:
        #     eval_cls = self.result_dict.keys()
        if self.book is None:
            self.init_book()
        sheet = self.book.add_sheet("BOX_SIZE",cell_overwrite_ok=True)
        sheet.write(0, 0, "box大小测评数据")
        sheet.write(2, 0, "检出数量")
        sheet.write(3, 0, "漏检总数")
        sheet.write(4, 0, "真值总数")
        sheet.write(5, 0, "漏检率")
        sheet.write(6, 0, "误检总数")
        sheet.write(7, 0, "预测总数")
        sheet.write(8, 0, "误检率")
        sheet.write(9, 0, "Precision")
        sheet.write(10, 0, "Recall")
        sheet.write(11, 0, "mean_x")
        sheet.write(12, 0, "mean_y")
        sheet.write(13, 0, "mean_w")
        sheet.write(14, 0, "mean_h")
        
        tp_list = [0 for i in self.dist_splits]
        fn_list = [0 for i in self.dist_splits]
        pre_iou_list = [0 for i in self.dist_splits]
        pre_num_list = [0 for i in self.dist_splits]
        gt_num_list = [0 for i in self.dist_splits]
        err_list = [[] for i in self.dist_splits] # error_x, error_y, gt_w, gt_h, det_w, det_h
        print(err_list)
        # 根据大小统计
        for key in eval_cls:
            print(key)
            data = self.result_dict[key]
            tp_list = [i+j for i,j in zip(tp_list, data.tp_list)]
            fn_list = [i+j for i,j in zip(fn_list, data.fn_list)]
            pre_iou_list = [i+j for i,j in zip(pre_iou_list, data.pre_iou_list)]
            pre_num_list = [i+j for i,j in zip(pre_num_list, data.pre_num_list)]
            gt_num_list = [i+j for i,j in zip(gt_num_list, data.gt_num_list)]
            
            for j in range(len(self.dist_splits)):
                err_list[j].extend(data.box_error_list[j])

        for i in range(len(self.dist_splits)):
            sheet.write(1, i+1, str(self.dist_splits[i]))
            sheet.write(2, i+1, str(tp_list[i]))
            sheet.write(3, i+1, str(fn_list[i]))
            sheet.write(4, i+1, str(gt_num_list[i]))
            if gt_num_list[i] > 0:
                missrate = round(fn_list[i] / gt_num_list[i], 5)
            else:
                missrate = 1. if fn_list[i] > 0. else 0.
            sheet.write(5, i+1, str(missrate))
            sheet.write(6, i+1, str(pre_iou_list[i]))
            sheet.write(7, i+1, str(pre_num_list[i]))
            if pre_num_list[i] > 0:
                falserate = round(pre_iou_list[i] / pre_num_list[i], 5)
            else:
                falserate = 1. if pre_iou_list[i] > 0. else 0.
            sheet.write(8, i+1, str(falserate))
            fp = pre_iou_list[i]
            fn = fn_list[i]
            tp = tp_list[i]
            if tp+fp > 0:
                precision  = round(tp / (tp+fp), 5)
            else:
                precision = 1. if tp > 0. else 0.
            sheet.write(9, i+1, str(precision))
            if tp+fn > 0:
                recall  = round(tp / (tp+fn), 5)
            else:
                recall = 1. if tp > 0. else 0.
            sheet.write(10, i+1, str(recall))
            err_dat = err_list[i]
            
            if err_dat != []:
                err_dat = np.mean(np.abs(err_dat), axis=0).tolist()
                err_x = err_dat[0]
                err_y = err_dat[1]
                
                gt_w = err_dat[2]
                gt_h = err_dat[3]
                det_w = err_dat[4]
                det_h = err_dat[5]
                err_w = abs(gt_w - det_w)
                err_h = abs(gt_h - det_h)
                sheet.write(11, i+1, "{}/{}".format(round(err_x, 3), round(err_x/gt_w, 3)))
                sheet.write(12, i+1, "{}/{}".format(round(err_y, 3), round(err_y/gt_h, 3)))
                sheet.write(13, i+1, "{}/{}".format(round(err_w,3 ), round(err_w/gt_w, 3)))
                sheet.write(14, i+1, "{}/{}".format(round(err_h,3), round(err_h/gt_w, 3)))
            else:
                sheet.write(11, i+1, "{}/{}".format(0, 0))
                sheet.write(12, i+1, "{}/{}".format(0, 0))
                sheet.write(13, i+1, "{}/{}".format(0, 0))
                sheet.write(14, i+1, "{}/{}".format(0, 0))
    
    def save(self, savepath):
        if self.book != None:
            self.book.save(savepath)
        else:
            print("No book.")
    
def fill_gap(data, gaps=[i*2000 for i  in range(1, 6)]):
    for i in range(len(gaps)):
        if data < gaps[i]:
            return i-1
    return len(gaps) - 1

def classify(ls, cls, gaps):
    out = [0 for i in range(len(gaps))]
    if cls == "all":
        for i in ls:
            group = fill_gap(i[0], gaps)
            if group >= 0:
                out[group] += 1
    else:
        for i in ls:
            cur_cls = i[1]
            if cur_cls == cls:
                group = fill_gap(i[0], gaps)
                if group >=0:
                    out[group] += 1
    return out
    
def classify_error(ls, cls, gaps):
    # box_size, obj_type, error_x, error_y, gt_w, gt_h, det_w, det_h, img_path
    
    out = [[] for i in range(len(gaps))]
    if ls == None:
        return out
    if cls == "all":
        for i in ls:
            group = fill_gap(i[0], gaps)
            if group >= 0:
                out[group].append(i[2:6])
    else:
        for i in ls:
            cur_cls = i[1]
            if cur_cls == cls:
                group = fill_gap(i[0], gaps)
                if group >=0:
                    out[group].append(i[2:8])
    return out

def load_results(save_dir):
    with open(os.path.join(save_dir, "box_tp_list.txt"), 'r') as f:
        content = f.read()
        box_tp_list = json.loads(content)
    with open(os.path.join(save_dir, "box_fn_list.txt"), 'r') as f:
        content = f.read()
        box_fn_list = json.loads(content)
    with open(os.path.join(save_dir, "box_pre_iou_list.txt"), 'r') as f:
        content = f.read()
        box_pre_iou_list = json.loads(content)
    with open(os.path.join(save_dir, "box_gt_num_list.txt"), 'r') as f:
        content = f.read()
        box_gt_num_list  = json.loads(content)
        
    with open(os.path.join(save_dir, "box_pre_num_list.txt"), 'r') as f:
        content = f.read()
        box_pre_num_list  = json.loads(content)
    try:
        with open(os.path.join(save_dir, "box_error_list.txt"), 'r') as f:
            content = f.read()
            box_error_list  = json.loads(content)
    except:
        box_error_list = None
        print("NO BOX ERROR")
    return box_tp_list,box_fn_list, box_pre_iou_list, box_pre_num_list, box_gt_num_list, box_error_list

def load_type(box_tp_list,box_fn_list, box_pre_iou_list, box_pre_num_list, box_gt_num_list, box_error_list, cls, gaps):
    pre_num_list = classify(box_pre_num_list, cls, gaps)
    pre_iou_list = classify(box_pre_iou_list, cls, gaps)
    tp_list = classify(box_tp_list, cls, gaps)
    fn_list = classify(box_fn_list, cls, gaps)
    gt_num_list = classify(box_gt_num_list, cls, gaps)
    box_error_list = classify_error(box_error_list, cls, gaps)
    # if cls == "CAR":
    #     print(box_error_list[-1])
    # print(pre_num_list)
    # print(pre_iou_list)
    # print(tp_list)
    # print(fn_list)
    # print(gt_num_list)
    
    type_stats = TypeStats(cls, tp_list, fn_list, pre_iou_list, pre_num_list, gt_num_list, box_error_list)
    return type_stats

def parse_metrics(box_tp_list,box_fn_list, box_pre_iou_list, box_pre_num_list, box_gt_num_list, box_error_list, gaps, save_dir, name="eval_2D_results.xls", eval_cls = None):
    types = []
    for i in box_gt_num_list:
        if i[1] not in types:
            types.append(i[1])
    stats = {}
    for cls in types:
        print(cls)
        stats[cls] = load_type(box_tp_list,box_fn_list, box_pre_iou_list, box_pre_num_list, box_gt_num_list, box_error_list, cls, gaps)
    result_stats = EvalStats(stats, gaps)
    result_stats.to_xls(os.path.join(save_dir, name), eval_cls)


def toxlsx_cls(savepath, cls_tp_list, cls_fn_list, cls_pre_iou_list, cls_pre_num_list, cls_gt_num_list):
    clses = list(cls_gt_num_list.keys())
    book = xlwt.Workbook(encoding='utf-8',style_compression=0)
    listname = "classwise_all"
    sheet = book.add_sheet(listname,cell_overwrite_ok=True)
    
    sheet.write(1, 0, "检出数量")
    sheet.write(2, 0, "漏检总数")
    sheet.write(3, 0, "误检总数")
    sheet.write(4, 0, "预测总数")
    sheet.write(5, 0, "真值总数")
    
    i = 1
    for cls in clses:
        sheet.write(0, i, str(cls))
        

        # 检出数量
        sheet.write(1, i, str(cls_tp_list[cls]))
        sheet.write(2, i, str(cls_fn_list[cls]))
        sheet.write(3, i, str(cls_pre_iou_list[cls]))
        sheet.write(4, i, str(cls_pre_num_list[cls]))
        sheet.write(5, i, str(cls_gt_num_list[cls]))
        i += 1
    book.save(savepath)
    

def toxlxs_3d(savepath,
           depth_error_list, depth_abs_list_new, depth_list_new,
           tp_list_new,fn_list_new,gt_num_list_new,miss_rate,
           pre_iou_list_new,pre_num_list_new,false_rate, 
           depth_error, shape_error, yaw_error, channel):
    book = xlwt.Workbook(encoding='utf-8',style_compression=0)
    listname = '3D-depth'
    sheet = book.add_sheet(listname,cell_overwrite_ok=True)
    x = 0
    
    for j in range(channel):
        sheet.write(0 , j, str(x))
        x+=10
    sheet.write(1, 0, str('3D目标深度误差百分比depth_error_list_min'))
    for i in range(1):
        for j in range(channel):
            sheet.write(i+2,j,str(depth_error_list[i][j][0]))
    sheet.write(3, 0, str('3D目标深度误差和depth_abs_list_new_min'))
    for i in range(1):
        for j in range(channel):
            sheet.write(i+4,j,str(depth_abs_list_new[i][j][0]))
    sheet.write(5, 0, str('3D目标深度和depth_list_new'))
    for i in range(1):
        for j in range(channel):
            sheet.write(i+6,j,str(depth_list_new[i][j][0]))

    sheet.write(7, 0, str('3D目标漏检率miss_rate'))
    for i in range(1):
        for j in range(channel):
            sheet.write(i+8,j,str(miss_rate[i][j][0]))
    sheet.write(9, 0, str('3D目标真值总和gt_num_list_new'))
    for i in range(1):
        for j in range(channel):
            sheet.write(i+10,j,str(gt_num_list_new[i][j][0]))
    sheet.write(11, 0, str('3D目标漏检个数fn_list_new'))
    for i in range(1):
        for j in range(channel):
            sheet.write(i+12,j,str(fn_list_new[i][j][0]))
    sheet.write(13, 0, str('3D目标检出个数tp_list_new'))
    for i in range(1):
        for j in range(channel):
            sheet.write(i+14, j, str(tp_list_new[i][j][0]))

    sheet.write(15, 0, str('3D目标误检率false_rate'))
    for i in range(1):
        for j in range(channel):
            sheet.write(i + 16, j, str(false_rate[i][j][0]))
    sheet.write(17, 0, str('3D目标误检个数pre_iou_list_new'))
    for i in range(1):
        for j in range(channel):
            sheet.write(i + 18, j, str(pre_iou_list_new[i][j][0]))
    sheet.write(19, 0, str('3D目标预测个数总和pre_num_list_new'))
    for i in range(1):
        for j in range(channel):
            sheet.write(i + 20, j, str(pre_num_list_new[i][j][0]))
    sheet.write(21, 0, str('3D目标深度误差百分比'))
    for j in range(1):
        for i in range(channel):
            sheet.write(j+22,i,str(depth_error[j][i]))
    sheet.write(23, 0, str('3D目标尺寸误差百分比'))
    for i in range(1):
        for j in range(channel):
            sheet.write(i+24,j,str(shape_error[i][j]))
    sheet.write(25, 0, str('3D目标yaw误差'))
    for i in range(1):
        for j in range(channel):
            sheet.write(i+26,j,str(yaw_error[i][j]))
    book.save(savepath)

# def toxlxs_3d(savepath,depth_error,shape_error,yaw_error,channel):
#     """
#     3D深度误差，大小误差
#     """
#     book = xlwt.Workbook(encoding='utf-8',style_compression=0)
#     listname = '3D-depth-honghu'
#     sheet = book.add_sheet(listname,cell_overwrite_ok=True)
#     x = 0
#     for i in range(channel):
#         sheet.write(0 , i, str(x))
#         x+=10
#     sheet.write(1, 0, str('3D目标深度误差百分比'))
#     for j in range(3):
#         for i in range(channel):
#             sheet.write(j+2,i,str(depth_error[j][i]))
#     sheet.write(5, 0, str('3D目标尺寸误差百分比'))
#     for i in range(3):
#         for j in range(channel):
#             sheet.write(i+6,j,str(shape_error[i][j]))
#     sheet.write(9, 0, str('3D目标yaw误差'))
#     for i in range(3):
#         for j in range(channel):
#             sheet.write(i+10,j,str(yaw_error[i][j]))
#     book.save(savepath)

def toxlxs_cube(savepath,raw_data):
    book = xlwt.Workbook(encoding='utf-8',style_compression=0)
    listname = 'cube-honghu'
    sheet = book.add_sheet(listname,cell_overwrite_ok=True)
    sheet.write(0, 0,str('目标编号'))
    sheet.write(0, 1, str('真值距离'))
    sheet.write(0, 2, str('推理值底边两点距离'))
    sheet.write(0, 3, str('绝对误差'))
    sheet.write(0, 4, str('相对误差'))
    for i in range(len(raw_data)):
        sheet.write(i+1,0,str(i))
        for j in range(4):
            sheet.write(i+1,j+1,str(raw_data[i][j]))
    book.save(savepath)


def merge_video(save_dir, cube_dir ):
    print("Generating video...")
    imgs = os.listdir(cube_dir)
    imgs.sort()
    img_sample = None
    for root, dirs, files in os.walk(cube_dir):
        for file in files:
            if file.endswith(".jpg"):
                img_sample = os.path.join(root, file)
                break
        if img_sample is not None:
            break
    img_test = cv2.imread(img_sample)
    h, w= img_test.shape[0:2]
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(os.path.join(save_dir,'./25D.avi'), fourcc, 10.0, (w, h))
    for imgpath in tqdm(imgs):
        img = cv2.imread(os.path.join(cube_dir, imgpath))
        out.write(img)
    out.release()
  
if __name__ == "__main__":
    save_dir = "/home/uto/workspace/infer/mono_infer_vgg/eval_results/aiv_night/night_city_pd28_debug/"
    box_tp_list,box_fn_list, box_pre_iou_list, box_pre_num_list, box_gt_num_list, box_error_list = load_results(save_dir)
    ## aggregate sub class
    # save_dir = "/home/uto/workspace/infer/mono_infer_vgg/eval_results/aiv_night/avi_night_port/"
    # tp_list,fn_list, pre_iou_list, pre_num_list, gt_num_list = load_results(save_dir)
    # box_tp_list.extend(tp_list)
    # box_fn_list.extend(fn_list)
    # box_pre_iou_list.extend(pre_iou_list)
    # box_pre_num_list.extend(pre_num_list)
    # box_gt_num_list.extend(gt_num_list)
    # save_dir = "/home/uto/workspace/infer/mono_infer_vgg/eval_results/aiv_night/avi_day/"
    # tp_list,fn_list, pre_iou_list, pre_num_list, gt_num_list = load_results(save_dir)
    # box_tp_list.extend(tp_list)
    # box_fn_list.extend(fn_list)
    # box_pre_iou_list.extend(pre_iou_list)
    # box_pre_num_list.extend(pre_num_list)
    # box_gt_num_list.extend(gt_num_list)
    
    ## plot hist 
    # print(box_fn_list)
    # data = [i[0] for i in box_fn_list]
    # print(max(data))
    # bins_interval =10
    # bins = list(range(int(min(data)), 150, bins_interval))
    # # print(bins)
    # # print(len(box_gt_num_list))
    # # print(max(box_gt_num_list), min(box_gt_num_list))
    # plt.hist(data, bins=bins)
    # plt.show()
    
   
    types = []
    for i in box_gt_num_list:
        if i[1] not in types:
            types.append(i[1])
    gaps = [32, 96]
    stats = {}
    for cls in types:
        print(cls)
        stats[cls] = load_type(box_tp_list,box_fn_list, box_pre_iou_list, box_pre_num_list, box_gt_num_list, box_error_list, cls, gaps)
    result_stats = EvalStats(stats, gaps)
    result_stats.to_xls("./stats/eval_avi_night_test_0129.xls", ["PD", "TRUCK"])