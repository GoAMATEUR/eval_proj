import numpy as np
import pdb

def mergeKeypoint(keypoint_head,keypoint_back):
    keypoint_head =keypoint_head.reshape(4,2)
    keypoint_back =keypoint_back.reshape(4,2)
    keypoint_merge = np.zeros([4,2])
    keypoint_merge[0] = keypoint_head[0]
    keypoint_merge[1] = keypoint_back[1]
    keypoint_merge[2] = keypoint_head[2]
    keypoint_merge[3] = keypoint_back[3]
    
    return keypoint_merge.reshape(-1)

def merge2DBox(keypoint_head_2dbox,keypoint_back_2dbox):

    keypoint_box = np.zeros([4])
    keypoint_box[0] =min(keypoint_head_2dbox[0], keypoint_back_2dbox[0])
    keypoint_box[1] =min(keypoint_head_2dbox[1], keypoint_back_2dbox[1])
    keypoint_box[2] =max(keypoint_head_2dbox[2], keypoint_back_2dbox[2])
    keypoint_box[3] =max(keypoint_head_2dbox[3], keypoint_back_2dbox[3])
    
    return keypoint_box

def generateOrthogonalCorners3d(obj,ry):
    """
    generate corners3d representation for this object
    :return corners_3d: (8, 3) corners of box3d in camera coord
    """
    l, h, w = obj.l, obj.h, obj.w
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]  # 上前右,上前左，上后左，上后右，下前右，下前左，下后左，下后右 
    y_corners = [h, h, h, h, 0, 0, 0, 0]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

    
    R = np.array([[np.cos(ry), 0, np.sin(ry)],
                    [0, 1, 0],
                    [-np.sin(ry), 0, np.cos(ry)]])

    corners3d = np.vstack([x_corners, y_corners, z_corners])  # (3, 8)
    corners3d = np.dot(R, corners3d).T
    corners3d = corners3d + obj.t

    return corners3d
    
def mergeOrthogonal3DBoundingbox(obj_head,obj_back,ry=0):
    box_head = generateOrthogonalCorners3d(obj_head,ry)
    box_back = generateOrthogonalCorners3d(obj_back,ry)
    box_merge = box_head.copy()
    box_merge[0] = box_head[0]
    box_merge[1] = box_head[1]
    box_merge[2] = box_back[2]
    box_merge[3] = box_back[3]
    box_merge[4] = box_head[4]
    box_merge[5] = box_head[5]
    box_merge[6] = box_back[6]
    box_merge[7] = box_back[7]
    return box_merge

def merge3DBoundingbox(obj_head,obj_back):
    box_head = generateOrthogonalCorners3d(obj_head,obj_head.ry)
    box_back = generateOrthogonalCorners3d(obj_back,obj_back.ry)
    box_merge = box_head.copy()
    box_merge[0] = box_head[0]
    box_merge[1] = box_head[1]
    box_merge[2] = box_back[2]
    box_merge[3] = box_back[3]
    box_merge[4] = box_head[4]
    box_merge[5] = box_head[5]
    box_merge[6] = box_back[6]
    box_merge[7] = box_back[7]
    return box_merge

def merge3Dbox(obj_head,obj_back):
    box_orthogonal = mergeOrthogonal3DBoundingbox(obj_head,obj_back,ry=0)
    top_h = max(obj_back.h,obj_head.h)
    
    obj_head.l = abs(box_orthogonal[0][0]- box_orthogonal[3][0])
    obj_head.h = top_h
    obj_head.w = abs(box_orthogonal[0][2]- box_orthogonal[1][2])
    obj_head.alpha =obj_back.alpha
    obj_head.ry =obj_back.ry
    
    truckheight = obj_head.t[1]
    box_with_ry =merge3DBoundingbox(obj_head,obj_back)
    obj_head.t = np.mean(box_with_ry,axis=0)
    obj_head.t[1]= truckheight
#     print("merge box center :",np.mean(box_with_ry,axis=0))
#     print("obj_head.t :",obj_head.t)
    return obj_head,obj_back
    
def dot_product_angle(v1, v2):
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        print("Zero magnitude vector!")
    else:
        vector_dot_product = np.dot(v1, v2)
        # angle2 = np.arctan2(v1[1], v1[0])
        # print("vector_dot_product ",vector_dot_product)
        arccos = np.arccos(vector_dot_product / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        # print("arccos ",arccos)
        if v1[1]<0:
            angle = -np.degrees(arccos)
        else:
            angle = np.degrees(arccos)

        return angle
    return 0
    
def getYawFromKeypoint(head_vector,back_vector):
    norm = np.sqrt(np.sum(head_vector*head_vector)+0.00001)
    head_vector /= norm
    norm = np.sqrt(np.sum(back_vector*back_vector)+0.00001)
    back_vector /= norm
    yaw = dot_product_angle(head_vector, back_vector)
    return yaw 
    
def twoBoxIsParallel(obj_head,obj_back):
    left_front_vector_head = obj_head.keypoint_down[:2] # 左前
    left_rear_vector_head = obj_head.keypoint_down[2:4] # 左后
    # print(left_front_vector_head)
    # print(left_rear_vector_head)
    hascalvector_head =False
    hascalvector_back =False
    if left_front_vector_head[0]!=-1.0 and left_front_vector_head[1]!=-1.0 and left_rear_vector_head[0]!=-1.0 and left_rear_vector_head[1]!=-1.0:
        vector_head = left_front_vector_head - left_rear_vector_head
        hascalvector_head =True
        left_front_vector_back = obj_back.keypoint_down[:2] # 左前
        left_rear_vector_back = obj_back.keypoint_down[2:4] # 左后
        if left_front_vector_back[0]!=-1.0 and left_front_vector_back[1]!=-1.0 and left_rear_vector_back[0]!=-1.0 and left_rear_vector_back[1]!=-1.0:
           vector_back = left_front_vector_back - left_rear_vector_back
           hascalvector_back =True
        else:
            right_front_vector_back = obj_back.keypoint_down[4:6] # 右前
            right_rear_vector_back = obj_back.keypoint_down[6:8] # 右后
            if right_front_vector_back[0]!=-1.0 and right_front_vector_back[1]!=-1.0 and right_rear_vector_back[0]!=-1.0 and right_rear_vector_back[1]!=-1.0:
                return 1,45
            else:
                return -2,0
        
    if not hascalvector_head:
        right_front_vector_head = obj_head.keypoint_down[4:6] # 右前
        right_rear_vector_head = obj_head.keypoint_down[6:8] # 右后
        if right_front_vector_head[0]!=-1.0 and right_front_vector_head[1]!=-1.0 and right_rear_vector_head[0]!=-1.0 and right_rear_vector_head[1]!=-1.0:
            vector_head = right_front_vector_head - right_rear_vector_head
            hascalvector_head =True
            right_front_vector_back = obj_back.keypoint_down[4:6] # 右前
            right_rear_vector_back = obj_back.keypoint_down[6:8] # 右后
            # print(right_rear_vector_back)
            # print(right_front_vector_back)
            if right_front_vector_back[0]!=-1.0 and right_front_vector_back[1]!=-1.0 and right_rear_vector_back[0]!=-1.0 and right_rear_vector_back[1]!=-1.0:
                vector_back = right_front_vector_back - right_rear_vector_back
                hascalvector_back =True
            else:
                left_front_vector_back = obj_back.keypoint_down[:2] # 左前
                left_rear_vector_back = obj_back.keypoint_down[2:4] # 左后
                if left_front_vector_back[0]!=-1.0 and left_front_vector_back[1]!=-1.0 and left_rear_vector_back[0]!=-1.0 and left_rear_vector_back[1]!=-1.0:
                    return 1, 45
                else:
                    return 0,0
        else:
            return -1 ,0 
    

               
    if hascalvector_head and hascalvector_back:
        angle = getYawFromKeypoint(vector_head,vector_back)
        return 1,angle
    else:
        print("there is no result")
        return 0,0



    
    
    

