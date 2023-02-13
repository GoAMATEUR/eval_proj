            
import sys 
sys.path.append('/home/utopilot/workspace/infer/mono_infer_vgg/')
import numpy as np
from eval.eval_utils.eval_twobox_utils import  *
import cv2 ,os 
from eval.eval_utils.eval_kitti_utils import approx_proj_center, convertRot2Alpha, read_label, Object3d
import math
import copy

def show_original_label(img3,box2d,keypoints_down,classname,):
    cv2.putText(img3, '{}'.format(classname), tuple((int(box2d[0]), int(box2d[1]))), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, (255, 0, 0), 1, cv2.LINE_AA)
    if classname == "Dontcare":
        color = (0, 0, 255)
        thinkness = 3
        cv2.rectangle(img3, (int(box2d[0]), int(box2d[1])), (int(box2d[2]), int(box2d[3])), color=(0,0,255), thickness=3)
    else:
        
        cv2.rectangle(img3, (int(box2d[0]), int(box2d[1])), (int(box2d[2]), int(box2d[3])), color=(0,255,255), thickness=2)
    for i in range(4):
        if keypoints_down[2*i+1]>=0 and keypoints_down[2*i]>=0 :       
            img3=cv2.circle(img3, (round(float(keypoints_down[2*i])),round(float(keypoints_down[2*i+1]))), 4, color=(0, 255, 1))
    return img3




def vis_two_box(img, objs: Object3d):
    # add 2 attributes recording truck head and trailer.
    Object3d.back = None
    Object3d.head = None
#            print("obj num: ",len(objs))
    if not isinstance(img, np.ndarray):
        img = np.asarray(img)
    img_h, img_w = img.shape[:2]
    # img3= np.copy(img)
    for obj in objs:
        box2d = obj.box2d.copy()
        box2d[...,[0,2]] = np.clip(box2d[...,[0,2]], 0, img_w-1)
        box2d[...,[1,3]] = np.clip(box2d[...,[1,3]], 0, img_h-1)
        # pdb.set_trace()
        if obj.ytop_vis != -1 :
            box2d[...,[1,3]] = np.clip(box2d[...,[1,3]], obj.ytop_vis, img_h-1)
            obj.ymin = obj.ytop_vis
        obj.box2d = box2d

    has_trailerback = False
    vids_trailerback=[]
    vids_truck=[]
    for i,obj in enumerate(objs):
        if obj.type =='trailerback':
            has_trailerback = True
            vids_trailerback.append([obj.vid,i])

        if obj.type =='TRUCK':
            vids_truck.append([obj.vid,i])
    is_merged = False
    if has_trailerback:
        vids_trailerback = np.array(vids_trailerback).reshape(-1,2).astype(np.int)
        vids_truck = np.array(vids_truck).reshape(-1,2).astype(np.int)
        for k, trailbackid in enumerate(vids_trailerback):
            
            if trailbackid[0] in vids_truck[:,0]:
                # 挂车有拖头
                for j, truckid in enumerate(vids_truck):
                    if trailbackid[0] == truckid[0]:
                        # 挂车匹配拖头
                        
                        # pdb.set_trace()
                        has_val,angle= twoBoxIsParallel(objs[truckid[1]],objs[trailbackid[1]])
                        # print("trailheadid.ry : ,trailbackid.ry : ",objs[truckid[1]].ry*180/np.pi," : ",objs[trailbackid[1]].ry*180/np.pi)
                        if has_val ==1: # 车头 和车尾都有侧边
                            # print(has_val, objs[truckid[1]].l, angle)
                            if objs[truckid[1]].l>0 and  objs[trailbackid[1]].l>0 and abs(objs[truckid[1]].ry*180/np.pi - objs[trailbackid[1]].ry*180/np.pi) < 10: # 有3D 角度标注
                                # print("Case1")
                                old_head = copy.deepcopy(objs[truckid[1]])
                                old_back = copy.deepcopy(objs[trailbackid[1]])
                                
                                obj_head,obj_back = merge3Dbox(objs[truckid[1]],objs[trailbackid[1]])
                                # print(obj_head.keypoint_down,obj_back.keypoint_down)
                                obj_head.keypoint_down = mergeKeypoint(obj_head.keypoint_down,obj_back.keypoint_down)
                                obj_head.keypoint_up = mergeKeypoint(obj_head.keypoint_up,obj_back.keypoint_up)
                                
                                obj_head.orientation_x = obj_head.keypoint_down[[0, 2, 4 ,6]]
                                obj_head.orientation_y = obj_head.keypoint_down[[1, 3, 5 ,7]]
                                
                                obj_head.box2d=merge2DBox(obj_head.box2d,obj_back.box2d)
                                obj_head.xmin,obj_head.ymin,obj_head.xmax,obj_head.ymax = obj_head.box2d[0],obj_head.box2d[1],obj_head.box2d[2],obj_head.box2d[3]
                                obj_head.head = old_head
                                obj_head.back = old_back
                                
                                objs[truckid[1]] = obj_head
                                # objs[truckid[1]] = old_head
                                # objs[truckid[1]] = old_back
                                
                                objs[trailbackid[1]] = obj_back
                                objs[trailbackid[1]].type = "Dontcare"
                            
                            elif objs[truckid[1]].l==0 and angle<6: # 只有2.5D 标注的目标 
                                # print("25D ANNO")
                                old_head = copy.deepcopy(objs[truckid[1]])
                                old_back = copy.deepcopy(objs[trailbackid[1]])
                                obj_head,obj_back =objs[truckid[1]],objs[trailbackid[1]]
                                obj_head.keypoint_down = mergeKeypoint(obj_head.keypoint_down,obj_back.keypoint_down)
                                obj_head.keypoint_up = mergeKeypoint(obj_head.keypoint_up,obj_back.keypoint_up)
                                obj_head.box2d=merge2DBox(obj_head.box2d,obj_back.box2d)
                                obj_head.xmin,obj_head.ymin,obj_head.xmax,obj_head.ymax = obj_head.box2d[0],obj_head.box2d[1],obj_head.box2d[2],obj_head.box2d[3]
                                obj_head.orientation_x = obj_head.keypoint_down[[0, 2, 4 ,6]]
                                obj_head.orientation_y = obj_head.keypoint_down[[1, 3, 5 ,7]]
                                obj_head.head = old_head
                                obj_head.back = old_back
                                objs[truckid[1]] = obj_head
                                
                                objs[truckid[1]] = obj_head
                                objs[trailbackid[1]].type = "Dontcare"
                                is_merged = True
                            else: # 夹角大 不合并 车尾即为truck，车头为truck head  
                                # print("trailheadid.ry : ,trailbackid.ry : ",objs[truckid[1]].ry*180/np.pi," : ",objs[trailbackid[1]].ry*180/np.pi)
                                # print("angle", angle)
                                # print("# 夹角大 不合并 车尾即为truck，车头为truck head  ")
                                objs[trailbackid[1]].type = "TRUCK"
                                objs[truckid[1]].type = "TRUCKHEAD"
                                # pdb.set_trace()
                            # print("two box ",self.image_files[idx].split('/')[-1].split('.')[0], "angle: ", angle)
                            
                        elif has_val ==-1: # 车头单面可见，车头无侧面
                            # print("# 车头单面可见，车头无侧面")
                            truckheight = objs[truckid[1]].t[1]
                            objs[truckid[1]]= copy.deepcopy(objs[trailbackid[1]])
                            objs[truckid[1]].box2d= merge2DBox(objs[truckid[1]].box2d,objs[trailbackid[1]].box2d)
                            objs[truckid[1]].xmin,objs[truckid[1]].ymin,objs[truckid[1]].xmax,objs[truckid[1]].ymax = objs[truckid[1]].box2d[0],objs[truckid[1]].box2d[1],objs[truckid[1]].box2d[2],objs[truckid[1]].box2d[3]
                            objs[truckid[1]].t[1] = truckheight
                            objs[truckid[1]].type = "TRUCK"
                            objs[trailbackid[1]].type = "Dontcare"

    
                            
                        elif has_val ==-2: # 车尾单面可见，车尾无侧边
                            # print("# 车尾单面可见，车尾无侧边")
                            objs[truckid[1]].box2d= merge2DBox(objs[truckid[1]].box2d,objs[trailbackid[1]].box2d)
                            objs[truckid[1]].xmin,objs[truckid[1]].ymin,objs[truckid[1]].xmax,objs[truckid[1]].ymax = objs[truckid[1]].box2d[0],objs[truckid[1]].box2d[1],objs[truckid[1]].box2d[2],objs[truckid[1]].box2d[3]
                            objs[trailbackid[1]].type = "Dontcare"
                        else: # 车头和车尾都没有侧边
                            # print("# 车头和车尾都没有侧边")
                            # print("two box ", 'donot have any value')
                            pass
                            
            else:
                # 单独车尾，TRUCK
                objs[trailbackid[1]].type = "TRUCK"
    objs_cp = []
    for i,obj in enumerate(objs):
        cls = obj.type
        if cls =="AIV":
            cls="TRUCK"
        if cls in ["CAR","PD","Rider","Three","BUS","TRUCK","TRUCKHEAD","VAN","SPECIALCAR"]:
            
            keypoints_down = obj.keypoint_down.copy()
            # print(obj.box2d)
            # img3= show_original_label(img3,obj.box2d,keypoints_down,cls)
            objs_cp.append(obj)
        if cls == "Dontcare":
            keypoints_down = obj.keypoint_down.copy()
            # img3= show_original_label(img3,obj.box2d,keypoints_down,cls)
    objs = objs_cp
    # cv2.imwrite(os.path.join('./two_box',name +'_keypoint.jpg'),img3)
    return objs


if __name__ == "__main__":
    img_path = "/home/utopilot/workspace/infer/test11-anno/3029_a0_20221009_111348_604_402.jpg"
    img = cv2.imread(img_path)
    label_path = "/home/utopilot/workspace/infer/test11-anno/3029_a0_20221009_111348_604_402.txt"
    objs = read_label(label_path)
    vis_two_box(img, objs, "test")
    