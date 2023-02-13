import numpy as np

def nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

def nms_eara(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    clses = dets[:, -1]
    clses = clses.astype(int)
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = areas.argsort()[::-1]
    
    
    keep = []
    
    while order.size > 0:
        i = order[0]
        cl = clses[i]
        keep.append(i)
        
        # calculate area of intersection
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        
        inter = w * h
        ovr = inter /((x1[order[1:]]-x2[order[1:]])*(y1[order[1:]]-y2[order[1:]]))
        
        # if iou > thres, merge
        #inds = np.where(ovr <= thresh)[0]
        if cl != 1 and cl != 2:
            inds = np.where(np.logical_or(np.logical_or(clses[order[1:]] == 1, ovr <= thresh), clses[order[1:]] == 2))[0]
        else:
            # inds = np.where(np.logical_or(np.logical_or(clses[order[1:]] != cl, ovr <= thresh), clses[order[1:]] == 1))[0]
            
            inds = np.where(ovr <= thresh)[0]
        
        order = order[inds+1]
    return keep
    

def nms_inside(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    allkeep = order.copy()
    nokeep = []
    n = 0
    while n <order.size-1:
        
        i = order[n]
        i_cmp = n+1
        xx1 = np.maximum(x1[i], x1[order[i_cmp:]])
        yy1 = np.maximum(y1[i], y1[order[i_cmp:]])
        xx2 = np.minimum(x2[i], x2[order[i_cmp:]])
        yy2 = np.minimum(y2[i], y2[order[i_cmp:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
#         print(inter.shape) 
        ovr = inter / (areas[i] + areas[order[i_cmp:]] - inter)
        ovr_inside = inter / (areas[order[i_cmp:]])
        inds = np.where((ovr >0)&(ovr_inside>thresh))[0]+i_cmp
        #print(inds)
        nokeep.extend(inds.tolist())
        n =n+1
        #order = np.delete(order, 0)
        
    keep = []
    for kp in allkeep:
        #print(kp, nokeep)
        if kp not in nokeep:
            keep.append(kp)

    return keep