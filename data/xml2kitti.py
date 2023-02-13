#!/usr/bin/python

# pip install lxml
import sys
sys.path.append("/home/utopilot/workspace/infer/mono_infer_vgg/")
import numpy as np 
# sys.path.append('./')
from data.cube_parser_maxus import CubeParser
from data.xml_parser import parse_xml_honghu_to_kitti
from tqdm import tqdm
import argparse
import json
import os
from shutil import copyfile

class Voc2Coco(object):
    def __init__(self, ):
        self.m_image_set = "/home/utopilot/workspace/infer/data_1209/aiv_test/test_gt.txt"
        #self.m_image_ext = _args.image_ext if _args.image_ext is None else 'jpg'
        self.m_save_path = "/home/utopilot/workspace/infer/data_1209/aiv_test/gt_txt/"
        #self.m_honghu = _args.honghu
        self.m_xml_index = None
#         self.dirty_file = open(os.path.join('data', os.path.basename(self.m_image_set) + '-dirty-info.txt'), 'w')
        os.makedirs(self.m_save_path, exist_ok=True)
        self.m_json_dict = {"images": [],
                            "type": "instances",
                            "annotations": [],
                            "categories": []}

        self._load_xml_index()

    def _load_xml_index(self):
        print(self.m_image_set)
        with open(self.m_image_set, 'r') as fsets:
            self.m_xml_index = list(
                map(lambda x: x.strip().replace("test-GT","labels").replace(".jpg", ".xml"), fsets.readlines()))

    def box_to_string(self,
                    name,
                    wlh,
                    center,
                    bbox_2d,
                    truncation,
                    occlusion,
                    alpha,
                    yaw,
                    cube,
                    vislines,
                    vid,uuid):
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
        uuid = '{:d}'.format(uuid)

        output = name + trunc + occ + a + bb + hwl + xyz + y + cub + vis + vid
#         print(output)
       

        return output
    
    
    def convert(self):
        missed_categories = set()
        # 遍历所有的XML文件
        bnd_id = 1
        count =0
        image_id = 1

        cube_parser = CubeParser()

        for xml_idx in tqdm(range(len(self.m_xml_index)), ncols=100, desc="VOC2COCO"):
            
            abs_xml_path = self.m_xml_index[xml_idx]
            # if (abs_xml_path).split('/')[-1].split('.')[0]!='FV_20210713_063213_124':
            #     continue
            if not os.path.exists(abs_xml_path):
                raise ValueError("Non existed xml path: %s" % abs_xml_path)

                
            print(abs_xml_path)
            annotaion = parse_xml_honghu_to_kitti(abs_xml_path)
            print('annotaion=',annotaion)
            if annotaion is None:
                continue
            image_height = annotaion['image']['height']
            image_width = annotaion['image']['width']
            image_des = annotaion['image']
            image_des['id'] = image_id

            abs_image_path = abs_xml_path.replace('xml', 'jpg').replace('Label', 'Image')
            #if not os.path.exists(abs_image_path):
                #tqdm.write("Non existed pic path: %s" % abs_image_path)
                #continue

            self.m_json_dict['images'].append(image_des)

            objs = annotaion['annotation']
            if len(objs) == 0:
                tqdm.write(" There is no objects in xml : %s" % os.path.basename(abs_xml_path))
            label_name = os.path.basename(abs_xml_path).replace('xml','txt')
            label_path = os.path.join(self.m_save_path,label_name)
#             print(label_path,label_name)
            save_dic=label_path.replace(label_name,'')
#             print(save_dic)
            os.makedirs(save_dic,exist_ok=True)
            if os.path.exists(label_path):
                count +=1
            with open(label_path, "w") as label_file:
                for obj in objs :
                    category = obj['name']
                    bndbox = obj['bndbox']
                    uuid=obj['uuid']
                    vid=obj['vehicle_id']
                    
                    xmin, ymin, xmax, ymax = bndbox
                    #处理2.5d的信息
                    cube= cube_parser(obj, image_width=image_width, image_height=image_height)#lf lb rf rb 看不见的为None
                    vis=obj['vislines']                   
                    if 'real_3d' in obj and len(obj['real_3d']) > 0:
#                         print (obj['real_3d'])
                        u, v, z ,x, y , w, h, l, occluded,alpha, yaw = obj['real_3d']['x'], obj['real_3d']['y'], obj['real_3d']['z'],obj['real_3d']['ox'], obj['real_3d']['oy'], obj['real_3d']['w'], obj['real_3d']['h'] ,\
                        obj['real_3d']['l'], obj['real_3d']['occluded'],obj['real_3d']['alpha'],obj['real_3d']['yaw']
                        output = self.box_to_string(category,[w,l,h],[x,y,z],bndbox,0,occluded,alpha,yaw,cube,vis,vid,uuid)
                        label_file.write(output + '\n')
                    else:
                        try:
                            occluded = int(obj['attributes']['DisplaySituation'])
                        except:
                            occluded = -1
                        output = self.box_to_string(category,[0.0,0.,0.0],[0.0,0.0,0.0],bndbox,0,occluded,0.0,0.0,cube,vis,vid,uuid)
                        label_file.write(output + '\n')
        print("duplicate", count)
                        
def parse_args():
    parser = argparse.ArgumentParser(
        "Transform pascal format annotations into coco format!")
    parser.add_argument('--image_set', help='xml list file.',default='/workspace/data/AIV/test10-GT-anno.txt')#lidar-image-kitti-train-0910.txt honghu30-train-0901.txt honghu-test-242526-1.txt
    parser.add_argument('--save_path', help='Path to save json annotation file.',default='/workspace/data/testdata/OD-crane/ID2_rtg_test10-GT/gttxt/')
    parser.add_argument('--honghu', help='is honghu data.',default=False)
    parser.add_argument('--dataset', help='Must specify the dataset, '
                                        'and script will load responding label file automaticly.',default='CalmCarMaxusDataset')
    parser.add_argument('--check-size', dest='check_size',
                        help="Whether to check image size")
    parser.add_argument('--image-ext', dest='image_ext',
                        help="Image's extention")



    return parser.parse_args()

def match_label(img_name, label_dir = "/home/utopilot/workspace/infer/data_1209/OD_Vehicle_PD_Rider_OD/"):
    basename = img_name.split(".")[0]
    label_name = basename+".xml"
    label_path = None
    
    for i in range(1, 12):
        label_folder = os.path.join(label_dir, "Label"+str(i).zfill(7))
        imgs = os.listdir(label_folder)
        if label_name in imgs:
            label_path = os.path.join(label_folder, label_name)
            break
        
    return label_path

if __name__ == '__main__':
    # args = parse_args()
    cvt = Voc2Coco()
    cvt.convert()
    # imgpaths = "/home/utopilot/workspace/infer/data_1209/aiv_test/test_gt.txt"
    # savedir = "/home/utopilot/workspace/infer/data_1209/aiv_test/labels/"
    # for line in open(imgpaths, 'r'):
    #     line = line.strip()
        
    #     img_name = os.path.basename(line)
    #     foldername = line.split('/')[-2]
    #     xml_path = match_label(img_name)
        
        
    #     dst_path =f"/home/utopilot/workspace/infer/data_1209/aiv_test/labels/{foldername}/"
    #     print(xml_path, dst_path+img_name.replace("jpg", "xml"))
    #     os.makedirs(dst_path, exist_ok=True)
    #     copyfile(xml_path, dst_path+img_name.replace("jpg", "xml"))
