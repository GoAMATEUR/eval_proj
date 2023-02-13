# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
# import pdb
import os
import uuid as ud
from lxml import etree
import sys
import cv2
import math
import numpy as np

# BGR
VERTICAL_COLORS = [(255, 255, 255), (0, 125, 255), (0, 255, 0),
                   (255, 0, 0), (0, 255, 255), (0, 0, 255)]  # white x x blue x red

"""
每张图片对应一个标注实例, 标注实例的格式如下
annotation_dict = {
    'image': {
    'file_name': xx, 
    'height': xx, 
    'width': xx , 
    'worker': xxx, 标注人员},
    'annotation': [object_dict, object_dict, object_dict]
}
其中
object_dict = {
    'bndbox': [xmin, ymin, xmax, y_max], rectangle 必选, float
    'points': [[x1, y1], [x2, y2], ...], polygon 必选, float
    'type': 'rectangle' / 'polygon', 可选
    'uuid': '3dbf02e8-5a19-44e5-b0de-65155ee8fbac', 可选
    'attributes': dict{xx:xx, xx:xx, xx:xx}, 可选
    'pose': xx, 未使用
    'truncated': xx, 未使用
    'difficult': 0 / 1, 未使用
    'score: 0.75, 可选
    'name': 'car',
    'edges': [dict(name="bottom_edge", bndbox=[x1, y1, x2, y2])]
}
"""
CHN_TAG=['卡车-货车','面包车','皮卡车','小轿车']
ENG_TAG=['truck','van','pickup','car']

CHN_TAG_ATTRS=['正常','截断','背景遮挡-上下','背景遮挡-左右','拥挤','左-前','左-后','右-前','右-后','左','右','前','后']
ENG_TAG_ATTRS=['normal','truncated','occlusion_sx','occlusion_zy','crowded','left-front','left-behind','right-front','right-behind','left','right','front','behind']


def _get_check_element(root, name, length, default='default'):
    """
     Acquire xml element, and check whether it is valid?
     :param root: root element
     :param name: element name
     :param length: nums of child element, if 0, return all child elements without check
     :return: element value, xml element
     """
    elements = root.findall(name)
    # print(elements)

    if 0 == len(elements):
        print('cannot find',name,'from',root)
        # print(etree.Element(str(default)))
        return etree.Element(str(default))  # 这里的默认值写不上来，目前全都是None，先这样吧
        # return str(default)

    if len(elements) != length:
        raise RuntimeError('The nums of %s is supposed to be %d, '
                           'but is %d' % (name, length, len(elements)))
    if 1 == length:
        elements = elements[0]

    return elements


def maxus2kitti(obj_alpha):
    OBJ_ALPHA = float(obj_alpha)
    if OBJ_ALPHA > 180:
        OBJ_ALPHA = -(180 - (OBJ_ALPHA - 180))
    if OBJ_ALPHA <= 0 and OBJ_ALPHA > -90:
        OBJ_ALPHA = (-(90 + OBJ_ALPHA) * 3.14 / 180)
    if OBJ_ALPHA > 0 and OBJ_ALPHA <= 90:
        OBJ_ALPHA = (-(90 + OBJ_ALPHA) * 3.14 / 180)
    if OBJ_ALPHA > 90 and OBJ_ALPHA <= 180:
        OBJ_ALPHA = (270 - OBJ_ALPHA) * 3.14 / 180
    if OBJ_ALPHA > -180 and OBJ_ALPHA <= -90:
        OBJ_ALPHA = (-(90 + OBJ_ALPHA) * 3.14 / 180)

    return OBJ_ALPHA


def camera_parameter(intrinsic_matrix, rotation_matrix, translation_vector):
    camera_intrinsic = {
        # R，旋转矩阵
        "R": rotation_matrix,
        # t，平移向量
        "T": translation_vector,
        # 焦距，f/dx, f/dy
        "f": [intrinsic_matrix[0], intrinsic_matrix[4]],
        # principal point，主点，主轴与像平面的交点
        "c": [intrinsic_matrix[2], intrinsic_matrix[5]],

        "calib": np.array([[intrinsic_matrix[0], 0, intrinsic_matrix[2], 0],
                           [0, intrinsic_matrix[4], intrinsic_matrix[5], 0],
                           [0, 0, 1., 0]], dtype=np.float32)

    }
    return camera_intrinsic


def parse_xml(xml_path, image_ext='png'):
    """
    解析xml文件, 并返回解析后的结果
    :param xml_path: xml文件的绝对路径
    :param image_ext: xml文件对应图片格式, 可以是'png', 'jpg'等
    :return:
    """

    annotation = {'image': None, 'camera_param': None, 'annotation': []}

    try:
        xml_tree = etree.parse(xml_path)
    except:
        print(xml_path,' error')
        return None
    
    xml_root = xml_tree.getroot()

    name = os.path.splitext(os.path.basename(xml_path))[0]
    image_name = "%s.%s" % (name, image_ext)
    image_prefix = os.path.dirname(xml_path)
#     img = cv2.imread(
#         '/workspace/UserData/ccdetection3d-release_2021.04.27/3D-data/ysg/images/' + image_name, cv2.IMREAD_COLOR)

    # parse image size
    size_element = _get_check_element(xml_root, 'size', 1)

    # 部分xml文件中的size保存的是float类型的数据,因此需要转换一下
    image_width = float(_get_check_element(size_element, 'width', 1).text)
    image_height = float(_get_check_element(size_element, 'height', 1).text)

    # 获取标注人员
    worker = _get_check_element(xml_root, 'worker', 1).text

    annotation['image'] = dict(
        image_prefix=image_prefix,
        file_name=image_name,
        height=image_height,
        width=image_width
    )
#     print(image_prefix,':',image_name)
    # parse objects
    objs = xml_root.findall('object')

    for obj in objs:
        # 获取标注格式, 即 rectangle / polygon
        ann_type = _get_check_element(obj, 'type', 1, 'rectangle').text

        # 获取uuid
        uuid = _get_check_element(obj, 'uuid', 1).text

        # 获取目标名字
        name = _get_check_element(obj, 'name', 1).text
        if name in CHN_TAG:
            index=CHN_TAG.index(name)
            name=ENG_TAG[index]

        # pose
        pose = _get_check_element(obj, 'pose', 1).text

        # trucated
        truncated = _get_check_element(obj, 'truncated', 1).text

        # difficult
        difficult = _get_check_element(obj, 'difficult', 1).text

        # score
        score = _get_check_element(obj, 'score', 1).text

        # 获取详细标注
        numeric_points = None
        numeric_bndbox = None

        if ann_type is None or 'rectangle' == ann_type:
            bndbox_node = _get_check_element(obj, 'bndbox', 1)
            if _get_check_element(bndbox_node, 'xmin', 1).text =='非数字':continue
            xmin = float(_get_check_element(bndbox_node, 'xmin', 1).text)
            ymin = float(_get_check_element(bndbox_node, 'ymin', 1).text)
            xmax = float(_get_check_element(bndbox_node, 'xmax', 1).text)
            ymax = float(_get_check_element(bndbox_node, 'ymax', 1).text)
            numeric_bndbox = [xmin, ymin, xmax, ymax]
        elif ann_type in ['polygon', 'polyline']:
            points_node = _get_check_element(obj, 'points', 1)
            numeric_points = []
            sub_points_node = points_node.findall('point')
            for sub_point_node in sub_points_node:
                x = float(_get_check_element(sub_point_node, 'x', 1).text)
                y = float(_get_check_element(sub_point_node, 'y', 1).text)
                numeric_points.append([x, y])
        else:
            raise RuntimeError("Unspported ann type: %s" % ann_type)

        # parse 3d annotation
        edges = []
        children = obj.findall('child')
        for child in children:
            edge_name = _get_check_element(child, 'name', 1).text
            edge_bndbox_node = _get_check_element(child, 'bndbox', 1)
            x1 = float(_get_check_element(edge_bndbox_node, 'x1', 1).text)
            y1 = float(_get_check_element(edge_bndbox_node, 'y1', 1).text)
            x2 = float(_get_check_element(edge_bndbox_node, 'x2', 1).text)
            y2 = float(_get_check_element(edge_bndbox_node, 'y2', 1).text)
            edges.append(dict(
                name=edge_name,
                type=_get_check_element(child, 'type', 1).text,
                uuid=_get_check_element(child, 'uuid', 1).text,
                pose=_get_check_element(child, 'pose', 1).text,
                truncated=_get_check_element(child, 'truncated', 1).text,
                difficult=_get_check_element(child, 'difficult', 1).text,
                bndbox=[x1, y1, x2, y2]
            ))

        # 读取属性
        attrs_node = _get_check_element(obj, 'attributes', 1)
        if len(attrs_node) == 0:
            attrs_dict = None
        else:
            attrs_dict = dict()
            for attr_node in attrs_node:
                attr_tag = attr_node.tag
                attr_value = attr_node.text
                #标签映射
                if name in CHN_TAG_ATTRS:
                    index=CHN_TAG_ATTRS.index(name)
                    name=ENG_TAG_ATTRS[index]
                    
                attrs_dict[attr_tag] = attr_value

        bbox_3d = dict()
        projections = _get_check_element(obj, 'box_projection', 1)
        # zhaoxin-------
        if projections is not None:
            for point in projections:
                if _get_check_element(point, 'x', 1).text != 'None':
                    try:
                        x = float(_get_check_element(point, 'x', 1).text)
                        y = float(_get_check_element(point, 'y', 1).text)
                        bbox_3d[point.tag] = [x, y]
                    except:
                        continue

        real_3d = dict()
        label_3d = obj.find('real_3d')
        if label_3d is not None:
            # print(xml_path,_get_check_element(label_3d, 'x', 1).text)
            # 洗掉一部分明明没有3d信息还硬要标为None的数据
            if _get_check_element(label_3d, 'x', 1).text != "None":
                real_3d = dict(
                    x=float(_get_check_element(label_3d, 'x', 1).text),
                    y=float(_get_check_element(label_3d, 'y', 1).text),
                    z=float(_get_check_element(label_3d, 'z', 1).text),
                    w=float(_get_check_element(label_3d, 'w', 1).text),
                    h=float(_get_check_element(label_3d, 'h', 1).text),
                    l=float(_get_check_element(label_3d, 'l', 1).text),
                    yaw=float(_get_check_element(label_3d, 'yaw', 1).text),
                    pitch=float(_get_check_element(label_3d, 'pitch', 1).text),
                    roll=float(_get_check_element(label_3d, 'roll', 1).text))
#                 print(real_3d)
                x = real_3d['x']
                y = real_3d['y']
                z = real_3d['z']
                w = real_3d['w']
                h = real_3d['h']
                l = real_3d['l']
                yaw = real_3d['yaw']

#                 pi = 3.1415926
#                 r = math.sqrt((w / 2) ** 2 + (l / 2) ** 2)
#                 alpa = np.arctan(w / l) * 180 / pi - yaw
#                 belta = np.arctan(w / l) * 180 / pi + yaw
#                 #           print(r*math.sin(math.radians(alpa)),r*math.sin(math.radians(belta)))
#                 left_front_up = [
#                     x - r * math.sin(math.radians(alpa)), y - h / 2, z + r * math.cos(math.radians(alpa))]
#                 left_front_down = [
#                     x - r * math.sin(math.radians(alpa)), y + h / 2, z + r * math.cos(math.radians(alpa))]
#                 right_front_up = [
#                     x + r * math.sin(math.radians(belta)), y - h / 2, z + r * math.cos(math.radians(belta))]
#                 right_front_down = [
#                     x + r * math.sin(math.radians(belta)), y + h / 2, z + r * math.cos(math.radians(belta))]
#                 left_behind_up = [
#                     x - r * math.sin(math.radians(belta)), y - h / 2, z - r * math.cos(math.radians(belta))]
#                 left_behind_down = [
#                     x - r * math.sin(math.radians(belta)), y + h / 2, z - r * math.cos(math.radians(belta))]
#                 right_behind_up = [
#                     x + r * math.sin(math.radians(alpa)), y - h / 2, z - r * math.cos(math.radians(alpa))]
#                 right_behind_down = [
#                     x + r * math.sin(math.radians(alpa)), y + h / 2, z - r * math.cos(math.radians(alpa))]

#                 points = [left_front_up, left_front_down, right_front_up, right_front_down,
#                           left_behind_up, left_behind_down, right_behind_up, right_behind_down]

#                 f, c = [0.0, 0.0], [0.0, 0.0]
#                 f[0] = 2071.35204
#                 f[1] = 2070.93690
#                 c[0] = 958.392422
#                 c[1] = 639.700971

#                 u = x / z * f[0] + c[0]
#                 v = y / z * f[1] + c[1]
#                 u = round(u, 4)
#                 v = round(v, 4)
#                 #             print(u,v)
#                 cv2.circle(img, (int(u), int(v)), 1, (0, 0, 255), 4)

#                 i = 0
#                 upoint = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
#                 vpoint = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
#                 for point in points:
#                     #                 print(point[0],point[1],point[2])
#                     upoint[i] = point[0] / point[2] * f[0] + c[0]
#                     vpoint[i] = point[1] / point[2] * f[1] + c[1]
#                     upoint[i] = round(upoint[i], 4)
#                     vpoint[i] = round(vpoint[i], 4)
#                     upoint[i] = int(upoint[i])
#                     vpoint[i] = int(vpoint[i])
#                     i = i + 1

#                 for i in range(4):
#                     cv2.line(img, (upoint[i], vpoint[i]), (upoint[i + 4],
#                                                            vpoint[i + 4]), VERTICAL_COLORS[5], thickness=2)
#                 for i in range(0, 4, 3):
#                     cv2.line(img, (upoint[i], vpoint[i]), (upoint[1],
#                                                            vpoint[1]), VERTICAL_COLORS[5], thickness=2)
#                     cv2.line(img, (upoint[i], vpoint[i]), (upoint[2],
#                                                            vpoint[2]), VERTICAL_COLORS[5], thickness=2)
#                 for i in range(4, 8, 3):
#                     cv2.line(img, (upoint[i], vpoint[i]), (upoint[5],
#                                                            vpoint[5]), VERTICAL_COLORS[5], thickness=2)
#                     cv2.line(img, (upoint[i], vpoint[i]), (upoint[6],
#                                                            vpoint[6]), VERTICAL_COLORS[5], thickness=2)

        obj_annotation = dict(
            type=ann_type,
            uuid=uuid,
            name=name,
            pose=pose,
            truncated=truncated,
            difficult=difficult,
            bndbox=numeric_bndbox,
            points=numeric_points,
            attributes=attrs_dict,
            score=score,
            edges=edges,
            bbox_3d=bbox_3d,
            real_3d=real_3d,
        )
        annotation['annotation'].append(obj_annotation)
#     cv2.imwrite(
#         '/workspace/UserData/ccdetection3d-release_2021.04.27/3D-data/ysg/GT/' + image_name, img)
    return annotation


def get_default(obj, key, default=None):
    if key not in obj.keys():
        return default
    else:
        return obj[key] if obj[key] is not None else default


def dump_xml(annotaion, xml_path):
    """
    将annotation中的内容保存到xml文件
    :param annotaion:
    :param xml_path:
    :return:
    """
    file_name = annotaion['image']['file_name']
    width = annotaion['image']['width']
    height = annotaion['image']['height']

    worker = get_default(annotaion['image'], 'worker', 'Unkown')

    root = etree.Element("annotation")

    etree.SubElement(root, "worker").text = worker
    etree.SubElement(root, "folder").text = "CalmCar"
    etree.SubElement(root, "filename").text = file_name

    source_node = etree.SubElement(root, "source")
    etree.SubElement(source_node, "database").text = "Unknown"

    size_node = etree.SubElement(root, "size")
    etree.SubElement(size_node, "width").text = str(width)
    etree.SubElement(size_node, "height").text = str(height)
    etree.SubElement(size_node, "depth").text = str(3)

    etree.SubElement(root, 'segmented').text = str(0)

    # objects
    objs = annotaion['annotation']
    for obj in objs:
        name = obj['name']
        uuid = get_default(obj, 'uuid', ud.uuid1())
        ann_type = get_default(obj, 'type', None)
        pose = get_default(obj, 'pose', 'Unspecified')
        truncated = get_default(obj, 'truncated', 0)
        difficult = get_default(obj, 'difficult', 0)
        attributes = get_default(obj, 'attributes', None)
        bndbox = get_default(obj, 'bndbox', None)
        points = get_default(obj, 'points', None)
        score = get_default(obj, 'score', None)
        edges = get_default(obj, 'edges', None)

        # 如果没有指定标注类型, 比如老的标注文件, 那么根据bndbox和point是否为None来确定类型
        if ann_type is None:
            if bndbox is None and points is not None:
                ann_type = "polygon"
            elif points is None and bndbox is not None:
                ann_type = 'rectangle'
            else:
                raise RuntimeError("Unsupported type")

        obj_node = etree.SubElement(root, 'object')
        etree.SubElement(obj_node, 'name').text = name
        etree.SubElement(obj_node, 'type').text = ann_type
        etree.SubElement(obj_node, 'uuid').text = str(uuid)

        attr_node = etree.SubElement(obj_node, 'attributes')
        if attributes is not None:
            for tag, value in attributes.items():
                sub_attr_node = etree.SubElement(attr_node, tag)
                sub_attr_node.text = str(value)

        etree.SubElement(obj_node, 'pose').text = str(pose)
        etree.SubElement(obj_node, 'truncated').text = str(truncated)
        etree.SubElement(obj_node, 'difficult').text = str(difficult)

        if score is not None:
            etree.SubElement(obj_node, 'score').text = str(score)

        if 'rectangle' == ann_type:
            bndbox_node = etree.SubElement(obj_node, 'bndbox')
            etree.SubElement(bndbox_node, 'xmin').text = str(bndbox[0])
            etree.SubElement(bndbox_node, 'ymin').text = str(bndbox[1])
            etree.SubElement(bndbox_node, 'xmax').text = str(bndbox[2])
            etree.SubElement(bndbox_node, 'ymax').text = str(bndbox[3])

        elif 'polygon' == ann_type:
            points_node = etree.SubElement(obj_node, 'points')
            for sub_points in points:
                sub_points_node = etree.SubElement(points_node, 'point')
                etree.SubElement(sub_points_node, 'x').text = str(
                    sub_points[0])
                etree.SubElement(sub_points_node, 'y').text = str(
                    sub_points[1])

        if edges is not None:
            for edge in edges:
                edge_node = etree.SubElement(obj_node, 'child')
                etree.SubElement(edge_node, 'name').text = str(edge['name'])
                etree.SubElement(edge_node, 'uuid').text = str(edge['uuid'])
                etree.SubElement(edge_node, 'type').text = str(edge['type'])
                etree.SubElement(edge_node, 'pose').text = str(edge['pose'])
                etree.SubElement(edge_node, 'truncated').text = str(
                    edge['truncated'])
                etree.SubElement(edge_node, 'difficult').text = str(
                    edge['difficult'])

                edge_bndbox_node = etree.SubElement(edge_node, 'bndbox')
                etree.SubElement(edge_bndbox_node, 'x1').text = str(
                    edge['bndbox'][0])
                etree.SubElement(edge_bndbox_node, 'y1').text = str(
                    edge['bndbox'][1])
                etree.SubElement(edge_bndbox_node, 'x2').text = str(
                    edge['bndbox'][2])
                etree.SubElement(edge_bndbox_node, 'y2').text = str(
                    edge['bndbox'][3])

    doc = etree.tostring(root, encoding='UTF-8',
                         pretty_print=True, xml_declaration=True)
    with open(xml_path, 'wb') as fxml:
        fxml.write(doc)


def parse_xml_maxus(xml_path, image_ext='jpg'):
#     pdb.set_trace()
    printpic = False
    annotation = {'image': None, 'camera_param': None, 'annotation': []}
    try:
        xml_tree = etree.parse(xml_path)
    except:
        return None
    xml_root = xml_tree.getroot()
    name = os.path.splitext(os.path.basename(xml_path))[0]
    image_name = "%s.%s" % (name, image_ext)
    image_prefix = os.path.dirname(xml_path).replace('anno', 'img')

    # find <CaptureInfo>
#     captureinfo = _get_check_element(xml_root, 'CaptureInfo', 1)
#     lidar2camparam = _get_check_element(captureinfo, 'Lidar2CamParam', 1)
#     intrinsic_matrix = [1] * 9
#     translation_vector = [1] * 3
#     info = _get_check_element(lidar2camparam, 'intrinsic_matrix0', 1).text
#     if info is not None:
#         intrinsic_matrix[0] = float(info)
#     else:
#         return None
#     info = _get_check_element(lidar2camparam, 'intrinsic_matrix1', 1).text
#     if info is not None:
#         intrinsic_matrix[1] = float(info)
#     else:
#         return None
#     info = _get_check_element(lidar2camparam, 'intrinsic_matrix2', 1).text
#     if info is not None:
#         intrinsic_matrix[2] = float(info)
#     else:
#         return None
#     info = _get_check_element(lidar2camparam, 'intrinsic_matrix3', 1).text
#     if info is not None:
#         intrinsic_matrix[3] = float(info)
#     else:
#         return None
#     info = _get_check_element(lidar2camparam, 'intrinsic_matrix4', 1).text
#     if info is not None:
#         intrinsic_matrix[4] = float(info)
#     else:
#         return None
#     info = _get_check_element(lidar2camparam, 'intrinsic_matrix5', 1).text
#     if info is not None:
#         intrinsic_matrix[5] = float(info)
#     else:
#         return None
#     info = _get_check_element(lidar2camparam, 'intrinsic_matrix6', 1).text
#     if info is not None:
#         intrinsic_matrix[6] = float(info)
#     else:
#         return None
#     info = _get_check_element(lidar2camparam, 'intrinsic_matrix7', 1).text
#     if info is not None:
#         intrinsic_matrix[7] = float(info)
#     else:
#         return None
#     info = _get_check_element(lidar2camparam, 'intrinsic_matrix8', 1).text
#     if info is not None:
#         intrinsic_matrix[8] = float(info)
#     else:
#         return None
#     info = _get_check_element(lidar2camparam, 'rotation_vector0', 1).text
#     if info is not None:
#         rotation_vector0 = float(info)
#     else:
#         return None
#     info = _get_check_element(lidar2camparam, 'rotation_vector1', 1).text
#     if info is not None:
#         rotation_vector1 = float(info)
#     else:
#         return None
#     info = _get_check_element(lidar2camparam, 'rotation_vector2', 1).text
#     if info is not None:
#         rotation_vector2 = float(info)
#     else:
#         return None
#     p = (float(rotation_vector0), float(rotation_vector1), float(rotation_vector2))
#     rotation_matrix = cv2.Rodrigues(p)[0]

#     info = _get_check_element(lidar2camparam, 'translation_vector0', 1).text
#     if info is not None:
#         translation_vector[0] = float(info)
#     else:
#         return None
#     info = _get_check_element(lidar2camparam, 'translation_vector1', 1).text
#     if info is not None:
#         translation_vector[1] = float(info)
#     else:
#         return None
#     info = _get_check_element(lidar2camparam, 'translation_vector2', 1).text
#     if info is not None:
#         translation_vector[2] = float(info)
#     else:
#         return None
#     camera_intrinsic = camera_parameter(
#         intrinsic_matrix, rotation_matrix, translation_vector)

#     annotation['camera_param'] = camera_intrinsic

    # parse image size
    # find <CaptureInfo>
#     try:
#         captureinfo = _get_check_element(xml_root, 'CaptureInfo', 1)
# #         cameraparam = _get_check_element(captureinfo, 'CameraParam', 1)
# #         imgwidth = float(_get_check_element(cameraparam, 'imgWidth', 1).text)
# #         imgheight = float(_get_check_element(cameraparam, 'imgHeight', 1).text)
#         imgwidth = float(3840)
#         imgheight = float(2160)
#     except:
#         return None

    # image info
    imgwidth = float(3840)
    imgheight = float(2160)
    annotation['image'] = dict(
        image_prefix=image_prefix,
        file_name=image_name,
        height=imgheight,
        width=imgwidth
    )

    # find <LabelInfo>
    labelinfo = _get_check_element(xml_root, 'LabelInfo', 1)

    # parse <Vehicle>
    try:
        vehicle = _get_check_element(labelinfo, 'Vehicle', 1)
        objs = vehicle.findall('Object')
        for obj in objs:
            # labeltype
            labeltype = obj.attrib['labelType']
            if labeltype == 'QPOS':  # 先不管全样本
                continue

            # 标注格式, 即 rectangle / polygon
            ann_type = "rectangle"
            # 获取ID
            uuid = int(obj.attrib['ID'])
            # 获取目标名字
            name = obj.attrib['kind']
            if name == 'UNKNOWN':  # imwrite <kind-UNKNOWN>
                printpic = True
                print(image_name, "kind unknown")
            # 获取详细标注
            numeric_points = None
            numeric_bndbox = None
            # 读取属性
            attrs_dict = dict()
            attr_tag_dis = "DisplaySituation"
            attr_value_dis = obj.attrib['isCovered']
            attrs_dict[attr_tag_dis] = attr_value_dis
            attr_tag_dir = "direction"
            orientation = obj.attrib['orientation']
            attrs_dict[attr_tag_dir] = orientation
            if orientation == 'UNKNOWN':  # imwrite <orientation-UNKNOWN>
                printpic = True
                print(image_name, "orientation unknown")

            # 获取一些坐标信息
            bodyrectx = int(obj.attrib['bodyRectX'])
            bodyrecty = int(obj.attrib['bodyRectY'])
            bodyrectwidth = int(obj.attrib['bodyRectWidth'])
            bodyrectheight = int(obj.attrib['bodyRectHeight'])
            sidecorupx = int(obj.attrib['sideCorUpX'])
            sidecorupy = int(obj.attrib['sideCorUpY'])
            sidecordownx = int(obj.attrib['sideCorDownX'])
            sidecordowny = int(obj.attrib['sideCorDownY'])
            xlft_vis=int(obj.attrib['xLft_vis'])
            xrght_vis=int(obj.attrib['xRght_vis'])

            #挂车左右可见线处理
            if name == 'trailerback' and (orientation == 'LEFT_FRONT' or orientation == 'RIGHT_REAR' or orientation == 'LEFT_REAR' or orientation == 'RIGHT_FRONT'):
                if sidecorupy==sidecordowny:
                    if orientation == 'LEFT_FRONT' or orientation == 'RIGHT_REAR':
                        if xlft_vis<bodyrectx:
                            sidecorupy=get_y_pos(sidecorupx,sidecorupy,bodyrectx,bodyrecty,xlft_vis)
                            sidecordowny=get_y_pos(sidecordownx,sidecordowny,bodyrectx,(bodyrecty+bodyrectheight-1),xlft_vis)
                            sidecorupx=xlft_vis
                            sidecordownx=xlft_vis
                        else:
                            sidecorupx=xlft_vis
                            sidecordownx=xlft_vis
                            sidecorupy=bodyrecty
                            sidecordowny=bodyrecty+height-1
                    else:
                        if xrght_vis>(bodyrectx+bodyrectwidth-1):
                            sidecorupy=get_y_pos(sidecorupx,sidecorupy,(bodyrectx+bodyrectwidth-1),bodyrecty,xrght_vis)
                            sidecordowny=get_y_pos(sidecordownx,sidecordowny,(bodyrectx+bodyrectwidth-1),(bodyrecty+bodyrectheight-1),xrght_vis)
                            sidecorupx=xrght_vis
                            sidecordownx=xRght_vis
                        else:
                            sidecorupx=xrght_vis
                            sidecordownx=xrght_vis
                            sidecorupy=bodyrecty
                            sidecordowny=bodyrecty+height-1
                else:
                    pass

            #删去一些露出太少的卡车
            if name =='truck' and (orientation == 'LEFT_FRONT' or orientation == 'RIGHT_REAR' or orientation == 'LEFT_REAR' or orientation == 'RIGHT_FRONT'):
                vis=xrght_vis-xlft_vis
                if orientation == 'LEFT_FRONT' or orientation == 'RIGHT_REAR':
                    side_length=abs(sidecordownx-bodyrectx)
                else:
                    side_length=abs((bodyrectx+bodyrectwidth-1)-sidecordownx)
                if vis<(side_length/6):
                    print('delete a truck')
                    continue
                
            # 2d bbox
            if orientation == 'FRONT' or orientation == 'REAR':
                numeric_bndbox = [bodyrectx, bodyrecty, bodyrectx + bodyrectwidth - 1, bodyrecty + bodyrectheight - 1]
            elif orientation == 'LEFT':
                numeric_bndbox = [bodyrectx, bodyrecty, sidecordownx, sidecordowny]
            elif orientation == 'RIGHT':
                numeric_bndbox = [sidecorupx, sidecorupy, bodyrectx, bodyrecty + bodyrectheight - 1]
            elif orientation == 'LEFT_REAR' or orientation == 'RIGHT_FRONT':
                numeric_bndbox = [bodyrectx, min(bodyrecty,sidecorupy), max(sidecorupx, sidecordownx), max((bodyrecty + bodyrectheight - 1),sidecordowny)]
            elif orientation == 'LEFT_FRONT' or orientation == 'RIGHT_REAR':
                numeric_bndbox = [min(sidecorupx, sidecordownx), min(bodyrecty,sidecorupy), bodyrectx + bodyrectwidth - 1,max((bodyrecty + bodyrectheight - 1),sidecordowny)]
            else:
                print("orientation error")
                numeric_bndbox = [min(sidecorupx,sidecordownx,bodyrectx),min(bodyrecty,sidecorupy),max(sidecorupx, sidecordownx, (bodyrectx + bodyrectwidth - 1)),max((bodyrecty + bodyrectheight - 1),sidecordowny)]
    #         print(numeric_bndbox)

            # 2.5d points
            pointa = (bodyrectx, bodyrecty + bodyrectheight - 1)
            pointb = (bodyrectx + bodyrectwidth - 1, bodyrecty + bodyrectheight - 1)
            pointc = (sidecordownx, sidecordowny)
            if orientation != 'UNKNOWN':  # imwrite <orientation-UNKNOWN>
                cubepoints = []
                cubepoints.append(dict(name="pointa", location=[pointa[0], pointa[1]]))
                cubepoints.append(dict(name="pointb", location=[pointb[0], pointb[1]]))
                cubepoints.append(dict(name="pointc", location=[pointc[0], pointc[1]]))
            else:
                cubepoints = None
                printpic = True
                print(image_name, "orientation unknown")
            # cubepoints [{'name': 'pointa', 'location': [1935, 1070]}, {'name': 'pointb', 'location': [1950, 1070]}, {'name': 'pointc', 'location': [1935, 1055]}]
    #         print('cubepoints',cubepoints)

            # 3d
            real_3d = dict()
            try:
                lidarObj = _get_check_element(obj, 'lidarObj', 1)
                if lidarObj.attrib['validFlag'] == '1':
                    obj_l = float(_get_check_element(lidarObj, 'objLength', 1).text)
                    obj_w = float(_get_check_element(lidarObj, 'objWidth', 1).text)
                    obj_h = float(_get_check_element(lidarObj, 'objHeight', 1).text)
                    obj_x = float(_get_check_element(lidarObj, 'objWx', 1).text)
                    obj_y = float(_get_check_element(lidarObj, 'objWy', 1).text)
                    obj_z = float(_get_check_element(lidarObj, 'objWz', 1).text)
                    obj_alpha = float(_get_check_element(lidarObj, 'objAzimuth', 1).text)
                    obj_alpha = maxus2kitti(obj_alpha)
                    # 车体坐标系 -> 相机坐标系
                    loc_lidar = [[float(obj_x), float(
                        obj_y), float(obj_z)]]
                    #             print(loc_lidar)
                    loc_lidar = np.asarray(loc_lidar)
                    R = np.asarray(camera_intrinsic["R"])
                    T = np.asarray(camera_intrinsic["T"])
                    joint_num = len(loc_lidar)
                    joint_cam = np.zeros((joint_num, 3))  # joint camera
                    for i in range(joint_num):  # joint i
                        joint_cam[i] = np.dot(R, loc_lidar[i]) + T  # R * (pt - T)
                    #             print(joint_cam)#[[ -4.2730145   -1.04935013 172.87431682]]
                    # 相机坐标系 -> 像素坐标系
                    obj_x = float(joint_cam[0][0])
                    obj_y = float(joint_cam[0][1])
                    obj_z = float(joint_cam[0][2])
                    calib = camera_intrinsic["calib"]
                    f, c = [0.0, 0.0], [0.0, 0.0]
                    # print("calib[0, 0]",calib[0, 0])
                    f[0] = calib[0, 0]
                    f[1] = calib[1, 1]
                    c[0] = calib[0, 2]
                    c[1] = calib[1, 2]
                    u = obj_x / obj_z * f[0] + c[0]
                    v = obj_y / obj_z * f[1] + c[1]
                    d = obj_z
                    u = round(u, 4)
                    v = round(v, 4)
                    d = round(d, 4)

                    real_3d = dict(x=u, y=v, z=d, w=obj_w, h=obj_h, l=obj_l, yaw=obj_alpha)
            except:
                pass

            obj_annotation = dict(
                type=ann_type,
                uuid=uuid,
                name=name,
                bndbox=numeric_bndbox,
                points=numeric_points,
                attributes=attrs_dict,
                cubepoints=cubepoints,
                real_3d=real_3d,
            )
            annotation['annotation'].append(obj_annotation)
    except:
        pass
        

    # parse <Rider>
    try:
        rider = _get_check_element(labelinfo, 'Rider', 1)
        objs = rider.findall('Object')
        for obj in objs:
            # 标注格式, 即 rectangle / polygon
            ann_type = "rectangle"
            # 获取ID
            uuid = int(obj.attrib['ID'])
            # 获取目标名字
            name = "Rider"
            # 获取详细标注
            numeric_points = None
            numeric_bndbox = None
            # 读取属性
            attrs_dict = dict()
            attr_tag_dis = "DisplaySituation"
            attr_value_dis = obj.attrib['isCorvered']
            attrs_dict[attr_tag_dis] = attr_value_dis
            attr_tag_dir = "direction"
            orientation = None
            attrs_dict[attr_tag_dir] = orientation

            # 获取一些坐标信息
            bodyrectx = int(obj.attrib['bodyRectX'])
            bodyrecty = int(obj.attrib['bodyRectY'])
            bodyrectwidth = int(obj.attrib['bodyRectWidth'])
            bodyrectheight = int(obj.attrib['bodyRectHeight'])
#             wheelpts0x = int(obj.attrib['wheelPts0X'])
#             wheelpts0y = int(obj.attrib['wheelPts0Y'])
#             wheelpts1x = int(obj.attrib['wheelPts1X'])
#             wheelpts1y = int(obj.attrib['wheelPts1Y'])
#             headrectx = int(obj.attrib['headRectX'])
#             headrecty = int(obj.attrib['headRectY'])
#             headrectwidth = int(obj.attrib['headRectWidth'])
#             headrectheight = int(obj.attrib['headRectHeight'])

            # 2d bbox
            numeric_bndbox = [bodyrectx, bodyrecty, bodyrectx +
                            bodyrectwidth - 1, bodyrecty + bodyrectheight - 1]

            # 2.5d points
            cubepoints = None

            # 3d
            real_3d = dict()
            try:
                lidarObj = _get_check_element(obj, 'lidarObj', 1)
                if lidarObj.attrib['validFlag'] == '1':
                    obj_l = float(_get_check_element(lidarObj, 'objLength', 1).text)
                    obj_w = float(_get_check_element(lidarObj, 'objWidth', 1).text)
                    obj_h = float(_get_check_element(lidarObj, 'objHeight', 1).text)
                    obj_x = float(_get_check_element(lidarObj, 'objWx', 1).text)
                    obj_y = float(_get_check_element(lidarObj, 'objWy', 1).text)
                    obj_z = float(_get_check_element(lidarObj, 'objWz', 1).text)
                    obj_alpha = float(_get_check_element(lidarObj, 'objAzimuth', 1).text)
                    #如果是处理鸿鹄数据集，使用：
                    obj_alpha = maxus2kitti(obj_alpha)
                    
                    #如果是处理大通27万张数据，使用：
#                     ang = wheelpts0y - wheelpts1y
#                     if (abs(float(obj_alpha)) < 60) or (abs(float(obj_alpha)) > 120):
#                         if ang < 0 and obj_alpha < 9999.0:
#                             if (float(obj_alpha) >= 0):
#                                 obj_alpha = str(float(obj_alpha) - 180)
#                             else:
#                                 obj_alpha = str(float(obj_alpha) + 180)
#                             obj_alpha = maxus2kitti(obj_alpha)
#                     else:
#                         if (wheelpts0x - wheelpts1x) < 0:
#                             if (float(obj_alpha) >= 0):
#                                 obj_alpha = str(float(obj_alpha) - 180)
#                             else:
#                                 obj_alpha = obj_alpha
#                             obj_alpha = maxus2kitti(obj_alpha)

#                         if (wheelpts0x - wheelpts1x) > 0:
#                             if (float(obj_alpha) >= 0):
#                                 obj_alpha = obj_alpha
#                             else:
#                                 obj_alpha = str(float(obj_alpha) + 180)
#                             obj_alpha = maxus2kitti(obj_alpha)

                    # 车体坐标系 -> 相机坐标系
                    loc_lidar = [[float(obj_x), float(
                        obj_y), float(obj_z)]]
                    #             print(loc_lidar)
                    loc_lidar = np.asarray(loc_lidar)
                    R = np.asarray(camera_intrinsic["R"])
                    T = np.asarray(camera_intrinsic["T"])
                    joint_num = len(loc_lidar)
                    joint_cam = np.zeros((joint_num, 3))  # joint camera
                    for i in range(joint_num):  # joint i
                        joint_cam[i] = np.dot(R, loc_lidar[i]) + T  # R * (pt - T)
                    #             print(joint_cam)#[[ -4.2730145   -1.04935013 172.87431682]]
                    # 相机坐标系 -> 像素坐标系
                    obj_x = float(joint_cam[0][0])
                    obj_y = float(joint_cam[0][1])
                    obj_z = float(joint_cam[0][2])
                    calib = camera_intrinsic["calib"]
                    f, c = [0.0, 0.0], [0.0, 0.0]
                    # print("calib[0, 0]",calib[0, 0])
                    f[0] = calib[0, 0]
                    f[1] = calib[1, 1]
                    c[0] = calib[0, 2]
                    c[1] = calib[1, 2]
                    u = obj_x / obj_z * f[0] + c[0]
                    v = obj_y / obj_z * f[1] + c[1]
                    d = obj_z
                    u = round(u, 4)
                    v = round(v, 4)
                    d = round(d, 4)

                    real_3d = dict(x=u, y=v, z=d, w=obj_w, h=obj_h, l=obj_l, yaw=obj_alpha)
            except:
                pass
                
            obj_annotation = dict(
                    type=ann_type,
                    uuid=uuid,
                    name=name,
                    bndbox=numeric_bndbox,
                    points=numeric_points,
                    attributes=attrs_dict,
                    cubepoints=cubepoints,
                    real_3d=real_3d,
            )
#             print('rider',obj_annotation)
            annotation['annotation'].append(obj_annotation)
    except:
        pass
        
    
    # parse <PD>
    try:
        pd = _get_check_element(labelinfo, 'PD', 1)
        objs = pd.findall('Object')
        for obj in objs:
            # 标注格式, 即 rectangle / polygon
            ann_type = "rectangle"
            # 获取ID
            uuid = int(obj.attrib['ID'])
            # 获取目标名字
            name = "PD"
            # 获取详细标注
            numeric_points = None
            numeric_bndbox = None
            # 读取属性
            attrs_dict = dict()
            attr_tag_dis = "DisplaySituation"
            attr_value_dis = obj.attrib['isCorvered']
            attrs_dict[attr_tag_dis] = attr_value_dis
            attr_tag_dir = "direction"
            orientation = None
            attrs_dict[attr_tag_dir] = orientation

            # 获取一些坐标信息
            bodyrectx = int(obj.attrib['bodyRectX'])
            bodyrecty = int(obj.attrib['bodyRectY'])
            bodyrectwidth = int(obj.attrib['bodyRectWidth'])
            bodyrectheight = int(obj.attrib['bodyRectHeight'])

            # 2d bbox
            numeric_bndbox = [bodyrectx, bodyrecty, bodyrectx +
                            bodyrectwidth - 1, bodyrecty + bodyrectheight - 1]

            # 2.5d points
            cubepoints = None

            # 3d
            real_3d = dict()
            try:
                lidarObj = _get_check_element(obj, 'lidarObj', 1)
                if lidarObj.attrib['validFlag'] == '1':
                    obj_l = float(_get_check_element(lidarObj, 'objLength', 1).text)
                    obj_w = float(_get_check_element(lidarObj, 'objWidth', 1).text)
                    obj_h = float(_get_check_element(lidarObj, 'objHeight', 1).text)
                    obj_x = float(_get_check_element(lidarObj, 'objWx', 1).text)
                    obj_y = float(_get_check_element(lidarObj, 'objWy', 1).text)
                    obj_z = float(_get_check_element(lidarObj, 'objWz', 1).text)
                    # obj_alpha = float(_get_check_element(
                    #     lidarObj, 'objAzimuth', 1).text)  # 注意PD的yaw都是0.0
                    # obj_alpha = maxus2kitti(obj_alpha)
                    obj_alpha = None
                    # 车体坐标系 -> 相机坐标系
                    loc_lidar = [[float(obj_x), float(
                        obj_y), float(obj_z)]]
                    #             print(loc_lidar)
                    loc_lidar = np.asarray(loc_lidar)
                    R = np.asarray(camera_intrinsic["R"])
                    T = np.asarray(camera_intrinsic["T"])
                    joint_num = len(loc_lidar)
                    joint_cam = np.zeros((joint_num, 3))  # joint camera
                    for i in range(joint_num):  # joint i
                        joint_cam[i] = np.dot(R, loc_lidar[i]) + T  # R * (pt - T)
                    #             print(joint_cam)#[[ -4.2730145   -1.04935013 172.87431682]]
                    # 相机坐标系 -> 像素坐标系
                    obj_x = float(joint_cam[0][0])
                    obj_y = float(joint_cam[0][1])
                    obj_z = float(joint_cam[0][2])
                    calib = camera_intrinsic["calib"]
                    f, c = [0.0, 0.0], [0.0, 0.0]
                    # print("calib[0, 0]",calib[0, 0])
                    f[0] = calib[0, 0]
                    f[1] = calib[1, 1]
                    c[0] = calib[0, 2]
                    c[1] = calib[1, 2]
                    u = obj_x / obj_z * f[0] + c[0]
                    v = obj_y / obj_z * f[1] + c[1]
                    d = obj_z
                    u = round(u, 4)
                    v = round(v, 4)
                    d = round(d, 4)

                    real_3d = dict(x=u, y=v, z=d, w=obj_w, h=obj_h, l=obj_l, yaw=obj_alpha)
            except:
                pass

            obj_annotation = dict(
                type=ann_type,
                uuid=uuid,
                name=name,
                bndbox=numeric_bndbox,
                points=numeric_points,
                attributes=attrs_dict,
                cubepoints=cubepoints,
                real_3d=real_3d,
            )
            annotation['annotation'].append(obj_annotation)
    except:
        pass

    # parse <barricade>
    try:
        barricade = _get_check_element(labelinfo, 'Barricade', 1)
        objs = barricade.findall('Object')
        for obj in objs:
            # 标注格式, 即 rectangle / polygon
            ann_type = "rectangle"
            # 获取ID
            uuid = int(obj.attrib['ID'])
            # 获取目标名字
            name = obj.attrib['kind']
            if name != 'CONES':  # 如果不是锥桶就pass不要
                continue
            # 获取详细标注
            numeric_points = None
            numeric_bndbox = None
            # 读取属性
            attrs_dict = dict()
#             attr_tag_dis = "DisplaySituation"
#             attr_value_dis = obj.attrib['isCovered']
#             attrs_dict[attr_tag_dis] = attr_value_dis

            # 获取一些坐标信息
            bodyrectx = float(obj.attrib['bodyRectX'])
            bodyrecty = float(obj.attrib['bodyRectY'])
            bodyrectwidth = float(obj.attrib['bodyRectWidth'])
            bodyrectheight = float(obj.attrib['bodyRectHeight'])

            # 2d bbox
            numeric_bndbox = [int(bodyrectx), int(bodyrecty), int(bodyrectx +
                            bodyrectwidth - 1), int(bodyrecty + bodyrectheight - 1)]

            # 2.5d points
            cubepoints = None

            # 3d
            real_3d = dict()
            
            obj_annotation = dict(
                type=ann_type,
                uuid=uuid,
                name=name,
                bndbox=numeric_bndbox,
                points=numeric_points,
                attributes=attrs_dict,
                real_3d=real_3d,
            )
            annotation['annotation'].append(obj_annotation)
    except:
        pass
    # if printpic == True:
    #     print("\n", image_name,
    #           "has been printed. Please check the corresponding folder.")
    #     cv2.imwrite(
    #         '/workspace/GroupShare/testdata/demo-3d/cornercase/' + image_name, img)
    
    #这一句删去了无样本的数据
    if len(annotation['annotation']) == 0:
        return None
    
    return annotation


def parse_xml_dlaas(xml_path, image_ext='jpg'):
    """
    解析xml文件, 并返回解析后的结果
    :param xml_path: xml文件的绝对路径
    :param image_ext: xml文件对应图片格式, 可以是'png', 'jpg'等
    :return:
    """

    annotation = {'image': None, 'annotation': []}

    xml_tree = etree.parse(xml_path)
    xml_root = xml_tree.getroot()

    name = os.path.splitext(os.path.basename(xml_path))[0]
    image_name = "%s.%s" % (name, image_ext)

    # parse image size
    size_element = _get_check_element(xml_root, 'size', 1)

    # 部分xml文件中的size保存的是float类型的数据,因此需要转换一下
    image_width = float(_get_check_element(size_element, 'width', 1).text)
    image_height = float(_get_check_element(size_element, 'height', 1).text)

    # 获取标注人员
    worker = _get_check_element(xml_root, 'worker', 1).text

    annotation['image'] = dict(
        file_name=image_name,
        height=image_height,
        width=image_width,
        worker=worker,
    )
    # parse objects
    objs = xml_root.findall('object')

    for obj in objs:
        # 获取标注格式, 即 rectangle / polygon
        ann_type = _get_check_element(obj, 'type', 1, 'rectangle').text

        # 获取uuid
        uuid = _get_check_element(obj, 'uuid', 1).text

        # 获取目标名字
        name = _get_check_element(obj, 'name', 1).text

        # pose
        pose = _get_check_element(obj, 'pose', 1).text

        # trucated
        truncated = _get_check_element(obj, 'truncated', 1).text

        # difficult
        difficult = _get_check_element(obj, 'difficult', 1).text

        # score
        score = _get_check_element(obj, 'score', 1).text

        # 获取详细标注
        numeric_points = None
        numeric_bndbox = None

        if ann_type is None or 'rectangle' == ann_type:
            bndbox_node = _get_check_element(obj, 'bndbox', 1)
            xmin = float(_get_check_element(bndbox_node, 'xmin', 1).text)
            ymin = float(_get_check_element(bndbox_node, 'ymin', 1).text)
            xmax = float(_get_check_element(bndbox_node, 'xmax', 1).text)
            ymax = float(_get_check_element(bndbox_node, 'ymax', 1).text)
            numeric_bndbox = [xmin, ymin, xmax, ymax]

            # parse 2.5d annotation
            edges = []
            point1 = _get_check_element(bndbox_node, 'point-1', 1).text
            point2 = _get_check_element(bndbox_node, 'point-2', 1).text
            point3 = _get_check_element(bndbox_node, 'point-3', 1).text
            point4 = _get_check_element(bndbox_node, 'point-4', 1).text
            # point5 = _get_check_element(bndbox_node, 'point-5', 1).text
            # print(type(point1),point1)
            # 使用真正的x y的最大最小值进行比较 如果第一个点的x y坐标都在bbox的内部，才进行竖棱的换算；否则直接进行底棱的换算
            if point1 is not None and point2 is not None:
                # print('start processing point1-5')
                point1x = float(point1.split(",")[0])
                point1y = float(point1.split(",")[1])
                point2x = float(point2.split(",")[0])
                point2y = float(point2.split(",")[1])
                # print(type(point1x),point1y)
                if min(xmin, xmax) <= point1x <= max(xmin, xmax) and (min(ymin, ymax) - 1) <= point1y <= (
                        max(ymin, ymax) + 2):
                    # print('point1&2 within the bbox')
                    # 1 side edge and 1 bottom edge
                    edge_name = 'side-edge'
                    edge_bbox = [point1x, point1y, point2x, point2y]
                    edges.append(dict(name=edge_name, type='line', uuid=None,
                                      pose='Unspecified', truncated='0', difficult='0', bndbox=edge_bbox))

                    if point3 is not None and point4 is not None:
                        point3x = float(point3.split(",")[0])
                        point3y = float(point3.split(",")[1])
                        point4x = float(point4.split(",")[0])
                        point4y = float(point4.split(",")[1])
                        edge_name = 'bottom-edge'
                        edge_bbox = [point3x, point3y, point4x, point4y]
                        edges.append(dict(name=edge_name, type='line', uuid=None,
                                          pose='Unspecified', truncated='0', difficult='0', bndbox=edge_bbox))

                else:
                    # only 1 bottom-edge without side-edge
                    if point3 is not None and point4 is not None:
                        point3x = float(point3.split(",")[0])
                        point3y = float(point3.split(",")[1])
                        point4x = float(point4.split(",")[0])
                        point4y = float(point4.split(",")[1])
                        edge_name = 'bottom-edge'
                        edge_bbox = [point3x, point3y, point4x, point4y]
                        edges.append(dict(name=edge_name, type='line', uuid=None,
                                          pose='Unspecified', truncated='0', difficult='0', bndbox=edge_bbox))
                    elif point3 is None:
                        print('image_name=', image_name, 'numeric_bndbox=',
                              numeric_bndbox, 'point1&2 out of bbox but point3 is None')
                    else:
                        print('image_name=', image_name, 'numeric_bndbox=',
                              numeric_bndbox, 'point1&2 out of bbox but point4 is None')
                        # raise RuntimeError("point1&2 out of bbox but point3&4 error: %s" % image_name)

            else:
                # print('Do not have edges')
                pass

        elif ann_type in ['polygon', 'polyline']:
            points_node = _get_check_element(obj, 'points', 1)
            numeric_points = []
            sub_points_node = points_node.findall('point')
            for sub_point_node in sub_points_node:
                x = float(_get_check_element(sub_point_node, 'x', 1).text)
                y = float(_get_check_element(sub_point_node, 'y', 1).text)
                numeric_points.append([x, y])
        else:
            raise RuntimeError("Unspported ann type: %s" % ann_type)

        # 通过angle读取属性,0表示没有方向，1-8表示如下八个方位：
        attrs_dict = dict()
        angle = _get_check_element(obj, 'angle', 1).text

        if angle == '0':
            attrs_dict = None
        else:
            attr_tag = 'direction'

            if angle == '1':
                attr_value = 'front'
            elif angle == '2':
                attr_value = 'right-front'
            elif angle == '3':
                attr_value = 'right'
            elif angle == '4':
                attr_value = 'right-behind'
            elif angle == '5':
                attr_value = 'behind'
            elif angle == '6':
                attr_value = 'left-behind'
            elif angle == '7':
                attr_value = 'left'
            elif angle == '8':
                attr_value = 'left-front'
            else:
                raise RuntimeError("Unspported angle type: %s" % angle)

            attrs_dict[attr_tag] = attr_value

        # 如果是人标了方位角，不要管它，设置为None。
        if 'Person' in name:  # 注意in关键词是大小写敏感
            attrs_dict = None

        bbox_3d = dict()
        projections = _get_check_element(obj, 'box_projection', 1)
        for point in projections:
            x = float(_get_check_element(point, 'x', 1).text)
            y = float(_get_check_element(point, 'y', 1).text)
            bbox_3d[point.tag] = [x, y]

        real_3d = dict()
        label_3d = obj.find('real_3d')
        if label_3d is not None:
            real_3d = dict(
                x=float(_get_check_element(label_3d, 'x', 1).text),
                y=float(_get_check_element(label_3d, 'y', 1).text),
                z=float(_get_check_element(label_3d, 'z', 1).text),
                w=float(_get_check_element(label_3d, 'w', 1).text),
                h=float(_get_check_element(label_3d, 'h', 1).text),
                l=float(_get_check_element(label_3d, 'l', 1).text),
                yaw=float(_get_check_element(label_3d, 'yaw', 1).text),
                pitch=float(_get_check_element(label_3d, 'pitch', 1).text),
                roll=float(_get_check_element(label_3d, 'roll', 1).text),
            )

        obj_annotation = dict(
            type=ann_type,
            uuid=uuid,
            name=name,
            pose=pose,
            truncated=truncated,
            difficult=difficult,
            bndbox=numeric_bndbox,
            points=numeric_points,
            attributes=attrs_dict,
            score=score,
            edges=edges,
            bbox_3d=bbox_3d,
            real_3d=real_3d,
        )
        annotation['annotation'].append(obj_annotation)

    return annotation


def parse_xml_honghu_to_kitti(xml_path,image_ext='jpg'):
    printpic = False
    annotation = {'image': None, 'camera_param': None, 'annotation': []}
    try:
        xml_tree = etree.parse(xml_path)
    except:
        return None
    xml_root = xml_tree.getroot()
    name = os.path.splitext(os.path.basename(xml_path))[0]
    image_name = "%s.%s" % (name, image_ext)
    image_prefix = os.path.dirname(xml_path).replace('Label', 'Image')

    # find <CaptureInfo>
    # captureinfo = _get_check_element(xml_root, 'CaptureInfo', 1)
#     lidar2camparam = _get_check_element(xml_root, 'Lidar2CamParam', 1)
#     intrinsic_matrix = [1] * 9
#     translation_vector = [1] * 3
#     info = _get_check_element(lidar2camparam, 'intrinsic_matrix0', 1).text
#     if info is not None:
#         intrinsic_matrix[0] = float(info)
#     else:
#         return None  
#     info = _get_check_element(lidar2camparam, 'intrinsic_matrix1', 1).text
#     if info is not None:
#         intrinsic_matrix[1] = float(info)
#     else:
#         return None
#     info = _get_check_element(lidar2camparam, 'intrinsic_matrix2', 1).text
#     if info is not None:
#         intrinsic_matrix[2] = float(info)
#     else:
#         return None
#     info = _get_check_element(lidar2camparam, 'intrinsic_matrix3', 1).text
#     if info is not None:
#         intrinsic_matrix[3] = float(info)
#     else:
#         return None
#     info = _get_check_element(lidar2camparam, 'intrinsic_matrix4', 1).text
#     if info is not None:
#         intrinsic_matrix[4] = float(info)
#     else:
#         return None
#     info = _get_check_element(lidar2camparam, 'intrinsic_matrix5', 1).text
#     if info is not None:
#         intrinsic_matrix[5] = float(info)
#     else:
#         return None
#     info = _get_check_element(lidar2camparam, 'intrinsic_matrix6', 1).text
#     if info is not None:
#         intrinsic_matrix[6] = float(info)
#     else:
#         return None
#     info = _get_check_element(lidar2camparam, 'intrinsic_matrix7', 1).text
#     if info is not None:
#         intrinsic_matrix[7] = float(info)
#     else:
#         return None
#     info = _get_check_element(lidar2camparam, 'intrinsic_matrix8', 1).text
#     if info is not None:
#         intrinsic_matrix[8] = float(info)
#     else:
#         return None
#     info = _get_check_element(lidar2camparam, 'rotation_vector0', 1).text
#     if info is not None:
#         rotation_vector0 = float(info)
#     else:
#         return None
#     info = _get_check_element(lidar2camparam, 'rotation_vector1', 1).text
#     if info is not None:
#         rotation_vector1 = float(info)
#     else:
#         return None
#     info = _get_check_element(lidar2camparam, 'rotation_vector2', 1).text
#     if info is not None:
#         rotation_vector2 = float(info)
#     else:
#         return None
#     p = (float(rotation_vector0), float(rotation_vector1), float(rotation_vector2))
#     rotation_matrix = cv2.Rodrigues(p)[0]

#     info = _get_check_element(lidar2camparam, 'translation_vector0', 1).text
#     if info is not None:
#         translation_vector[0] = float(info)
#     else:
#         return None
#     info = _get_check_element(lidar2camparam, 'translation_vector1', 1).text
#     if info is not None:
#         translation_vector[1] = float(info)
#     else:
#         return None
#     info = _get_check_element(lidar2camparam, 'translation_vector2', 1).text
#     if info is not None:
#         translation_vector[2] = float(info)
#     else:
#         return None
#     camera_intrinsic = camera_parameter(
#         intrinsic_matrix, rotation_matrix, translation_vector)

#     annotation['camera_param'] = camera_intrinsic

    # image size    
    imgwidth = int(1920)
    imgheight = int(1208)
    

    # image info
    annotation['image'] = dict(
        image_prefix=image_prefix,
        file_name=image_name,
        height=imgheight,
        width=imgwidth
    )

    # find <LabelInfo>
    labelinfo = _get_check_element(xml_root, 'LabelInfo', 1)

    # parse <Vehicle>
    try:
        vehicle = _get_check_element(labelinfo, 'Vehicle', 1)
        objs = vehicle.findall('Object')
        for obj in objs:
            labeltype = obj.attrib['labelType']
            if labeltype == 'QPOS':  # 不管全样本
                continue

            uuid = int(obj.attrib['ID'])
            try:
                name = obj.attrib['kind']
            except:
                name = 'UNKNOWN'
            try:
                vehicle_id = int(obj.attrib['vehicleID'])
            except:
                vehicle_id = -1
#             if name == 'UNKNOWN':  # 如果标成了unkonwn写下来，一般不会出现
#                 printpic = True
#                 print(image_name, "name unknown")
            
            # 读取属性
            attrs_dict = dict()
            if labeltype != "QPOS":
                attr_tag_dis = "DisplaySituation"
                attr_value_dis = obj.attrib['isCovered']
                attrs_dict[attr_tag_dis] = attr_value_dis
                attr_tag_dir = "direction"
                orientation = obj.attrib['orientation']
                attrs_dict[attr_tag_dir] = orientation
            else:
                orientation = None

            # 获取一些坐标信息
            bodyrectx = float(obj.attrib['bodyRectX'])
            bodyrecty = float(obj.attrib['bodyRectY'])
            bodyrectwidth = float(obj.attrib['bodyRectWidth'])
            bodyrectheight = float(obj.attrib['bodyRectHeight'])
            ytop_vis,ybuttom_vis,xlft_vis,xrght_vis=None,None,None,None #可见线，顺序：上下左右
            if labeltype != "QPOS":
                sidecorupx = float(obj.attrib['sideCorUpX'])
                sidecorupy = float(obj.attrib['sideCorUpY'])
                sidecordownx = float(obj.attrib['sideCorDownX'])
                sidecordowny = float(obj.attrib['sideCorDownY'])
                xlft_vis=float(obj.attrib['xLft_vis'])
                xrght_vis=float(obj.attrib['xRght_vis'])
                try:
                    ytop_vis=float(obj.attrib['yTop_vis'])
                except:
                    pass
            vislines=[ytop_vis,ybuttom_vis,xlft_vis,xrght_vis]
            

            #挂车左右可见线处理
            if name == 'trailerback' and (orientation == 'LEFT_FRONT' or orientation == 'RIGHT_REAR' or orientation == 'LEFT_REAR' or orientation == 'RIGHT_FRONT'):
                if sidecorupy==sidecordowny:
                    if orientation == 'LEFT_FRONT' or orientation == 'RIGHT_REAR':
                        if xlft_vis<bodyrectx:
                            sidecorupy=get_y_pos(sidecorupx,sidecorupy,bodyrectx,bodyrecty,xlft_vis)
                            sidecordowny=get_y_pos(sidecordownx,sidecordowny,bodyrectx,(bodyrecty+bodyrectheight-1),xlft_vis)
                            sidecorupx=xlft_vis
                            sidecordownx=xlft_vis
                        else:
                            sidecorupx=xlft_vis
                            sidecordownx=xlft_vis
                            sidecorupy=bodyrecty
                            sidecordowny=bodyrecty+height-1
                    else:
                        if xrght_vis>(bodyrectx+bodyrectwidth-1):
                            sidecorupy=get_y_pos(sidecorupx,sidecorupy,(bodyrectx+bodyrectwidth-1),bodyrecty,xrght_vis)
                            sidecordowny=get_y_pos(sidecordownx,sidecordowny,(bodyrectx+bodyrectwidth-1),(bodyrecty+bodyrectheight-1),xrght_vis)
                            sidecorupx=xrght_vis
                            sidecordownx=xRght_vis
                        else:
                            sidecorupx=xrght_vis
                            sidecordownx=xrght_vis
                            sidecorupy=bodyrecty
                            sidecordowny=bodyrecty+height-1
                else:
                    pass

            #删去一些露出太少的卡车
            if name =='truck' and (orientation == 'LEFT_FRONT' or orientation == 'RIGHT_REAR' or orientation == 'LEFT_REAR' or orientation == 'RIGHT_FRONT'):
                vis=xrght_vis-xlft_vis
                if orientation == 'LEFT_FRONT' or orientation == 'RIGHT_REAR':
                    side_length=abs(sidecordownx-bodyrectx)
                else:
                    side_length=abs((bodyrectx+bodyrectwidth-1)-sidecordownx)
                if vis<(side_length/6):
                    print('delete a truck')
                    continue

            # 2d bbox
            numeric_bndbox = None
            if orientation == 'FRONT' or orientation == 'REAR':
                numeric_bndbox = [bodyrectx, bodyrecty, bodyrectx + bodyrectwidth - 1, bodyrecty + bodyrectheight - 1]
            elif orientation == 'LEFT' or orientation == 'RIGHT':
                numeric_bndbox = [bodyrectx, bodyrecty, sidecordownx, sidecordowny]
            # elif orientation == 'RIGHT':
            #     numeric_bndbox = [sidecorupx, sidecorupy, bodyrectx, bodyrecty + bodyrectheight - 1]
            elif orientation == 'LEFT_REAR' or orientation == 'RIGHT_FRONT':
                numeric_bndbox = [bodyrectx, min(bodyrecty,sidecorupy), max(sidecorupx, sidecordownx), max((bodyrecty + bodyrectheight - 1),sidecordowny)]
            elif orientation == 'LEFT_FRONT' or orientation == 'RIGHT_REAR':
                numeric_bndbox = [min(sidecorupx, sidecordownx), min(bodyrecty,sidecorupy), bodyrectx + bodyrectwidth - 1,max((bodyrecty + bodyrectheight - 1),sidecordowny)]
            else:
                # print("orientation error")
                if labeltype == "QPOS":
                    numeric_bndbox = [bodyrectx, bodyrecty, bodyrectx + bodyrectwidth - 1, bodyrecty + bodyrectheight - 1]
                else:
                    numeric_bndbox = [min(sidecorupx,sidecordownx,bodyrectx),min(bodyrecty,sidecorupy),max(sidecorupx, sidecordownx, (bodyrectx + bodyrectwidth - 1)),max((bodyrecty + bodyrectheight - 1),sidecordowny)]
#             print(numeric_bndbox)
            
            # 2d bbox protect
            edge_protect = False
            if edge_protect:
                X_MIN, X_MAX, Y_MIN, Y_MAX = 0, imgwidth, 0, imgheight
                numeric_bndbox[0] = X_MIN if numeric_bndbox[0] < X_MIN else X_MAX if numeric_bndbox[0] > X_MAX else numeric_bndbox[0]
                numeric_bndbox[2] = X_MIN if numeric_bndbox[2] < X_MIN else X_MAX if numeric_bndbox[2] > X_MAX else numeric_bndbox[2]
                numeric_bndbox[1] = Y_MIN if numeric_bndbox[1] < Y_MIN else Y_MAX if numeric_bndbox[1] > Y_MAX else numeric_bndbox[1]
                numeric_bndbox[3] = Y_MIN if numeric_bndbox[3] < Y_MIN else Y_MAX if numeric_bndbox[3] > Y_MAX else numeric_bndbox[3]


            # 2.5d points
            cubepoints = None
            if labeltype != "QPOS":
                #接地三个点
                pointa = (bodyrectx, bodyrecty + bodyrectheight - 1)
                pointb = (bodyrectx + bodyrectwidth - 1, bodyrecty + bodyrectheight - 1)
                pointc = (sidecordownx, sidecordowny)
                #接地点对应的上面三个点
                pointau = (bodyrectx, bodyrecty)
                pointbu = (bodyrectx + bodyrectwidth - 1, bodyrecty)
                pointcu = (sidecorupx, sidecorupy)
                
                cubepoints = []
                cubepoints.append(dict(name="pointa", location=[pointa[0], pointa[1]]))
                cubepoints.append(dict(name="pointb", location=[pointb[0], pointb[1]]))
                cubepoints.append(dict(name="pointc", location=[pointc[0], pointc[1]]))
                cubepoints.append(dict(name="pointau", location=[pointau[0], pointau[1]]))
                cubepoints.append(dict(name="pointbu", location=[pointbu[0], pointbu[1]]))
                cubepoints.append(dict(name="pointcu", location=[pointcu[0], pointcu[1]]))

            # 3d
            real_3d = dict()
            try:
                lidarObj = _get_check_element(obj, 'lidarObj', 1)
                if lidarObj.attrib['validFlag'] == '1':
                    obj_l = float(_get_check_element(lidarObj, 'objLength', 1).text)
                    obj_w = float(_get_check_element(lidarObj, 'objWidth', 1).text)
                    obj_h = float(_get_check_element(lidarObj, 'objHeight', 1).text)
                    obj_x = float(_get_check_element(lidarObj, 'objWx', 1).text)
                    obj_y = float(_get_check_element(lidarObj, 'objWy', 1).text)
                    obj_z = float(_get_check_element(lidarObj, 'objWz', 1).text)
                    obj_yaw = float(_get_check_element(lidarObj, 'objAzimuth', 1).text)
                    obj_yaw = maxus2kitti(obj_yaw)
                    # 车体坐标系 -> 相机坐标系
                    loc_lidar = [[float(obj_x), float(
                        obj_y), float(obj_z)]]
                    #             print(loc_lidar)
                    loc_lidar = np.asarray(loc_lidar)
                    R = np.asarray(camera_intrinsic["R"])
                    T = np.asarray(camera_intrinsic["T"])
                    joint_num = len(loc_lidar)
                    joint_cam = np.zeros((joint_num, 3))  # joint camera
                    for i in range(joint_num):  # joint i
                        joint_cam[i] = np.dot(R, loc_lidar[i]) + T  # R * (pt - T)
                    #             print(joint_cam)#[[ -4.2730145   -1.04935013 172.87431682]]
                    # 相机坐标系 -> 像素坐标系
                    obj_x = float(joint_cam[0][0])
                    obj_y = float(joint_cam[0][1])
                    obj_z = float(joint_cam[0][2])
                    obj_alpha = obj_yaw - math.atan2(obj_z,obj_x)
                    calib = camera_intrinsic["calib"]
                    f, c = [0.0, 0.0], [0.0, 0.0]
                    # print("calib[0, 0]",calib[0, 0])
                    f[0] = calib[0, 0]
                    f[1] = calib[1, 1]
                    c[0] = calib[0, 2]
                    c[1] = calib[1, 2]
                    u = obj_x / obj_z * f[0] + c[0]
                    v = obj_y / obj_z * f[1] + c[1]
                    d = obj_z
                    u = round(u, 4)
                    v = round(v, 4)
                    d = round(d, 4)
                    
                    real_3d = dict(x=u, y=v, z=d, ox=obj_x, oy=obj_y, w=obj_w, h=obj_h, l=obj_l, occluded = int(attr_value_dis) ,alpha=obj_alpha, yaw= obj_yaw)
            except:
                pass

            obj_annotation = dict(
                uuid=uuid,
                name=name,
                bndbox=numeric_bndbox,
                attributes=attrs_dict,
                cubepoints=cubepoints,
                real_3d=real_3d,
                vislines=vislines,
                vehicle_id=vehicle_id
            )
            annotation['annotation'].append(obj_annotation)
            # print(annotation)
    except:
        pass

    # parse <Rider>
    try:
        rider = _get_check_element(labelinfo, 'Rider', 1)
        objs = rider.findall('Object')
        for obj in objs:
            uuid = int(obj.attrib['ID'])
            vehicle_id=-1
            name = "Rider"
            numeric_bndbox = None
            # 读取属性
            attrs_dict = dict()
            attr_tag_dis = "DisplaySituation"
            try:
                attr_value_dis = obj.attrib['isCorvered']
            except:
                attr_value_dis = obj.attrib['isCovered']
            attrs_dict[attr_tag_dis] = attr_value_dis
            attr_tag_dir = "direction"
            orientation = None
            attrs_dict[attr_tag_dir] = orientation

            # 获取一些坐标信息
            bodyrectx = float(obj.attrib['bodyRectX'])
            bodyrecty = float(obj.attrib['bodyRectY'])
            bodyrectwidth = float(obj.attrib['bodyRectWidth'])
            bodyrectheight = float(obj.attrib['bodyRectHeight'])

            # 2d bbox
            numeric_bndbox = [bodyrectx, bodyrecty, bodyrectx +
                            bodyrectwidth - 1, bodyrecty + bodyrectheight - 1]
            # 2d bbox protect
            edge_protect = False
            if edge_protect:
                X_MIN, X_MAX, Y_MIN, Y_MAX = 0, imgwidth, 0, imgheight
                numeric_bndbox[0] = X_MIN if numeric_bndbox[0] < X_MIN else X_MAX if numeric_bndbox[0] > X_MAX else numeric_bndbox[0]
                numeric_bndbox[2] = X_MIN if numeric_bndbox[2] < X_MIN else X_MAX if numeric_bndbox[2] > X_MAX else numeric_bndbox[2]
                numeric_bndbox[1] = Y_MIN if numeric_bndbox[1] < Y_MIN else Y_MAX if numeric_bndbox[1] > Y_MAX else numeric_bndbox[1]
                numeric_bndbox[3] = Y_MIN if numeric_bndbox[3] < Y_MIN else Y_MAX if numeric_bndbox[3] > Y_MAX else numeric_bndbox[3]

            # 2.5d points
            cubepoints = None

            # 3d
            real_3d = dict()
            try:
                lidarObj = _get_check_element(obj, 'lidarObj', 1)
                if lidarObj.attrib['validFlag'] == '1':
                    obj_l = float(_get_check_element(lidarObj, 'objLength', 1).text)
                    obj_w = float(_get_check_element(lidarObj, 'objWidth', 1).text)
                    obj_h = float(_get_check_element(lidarObj, 'objHeight', 1).text)
                    obj_x = float(_get_check_element(lidarObj, 'objWx', 1).text)
                    obj_y = float(_get_check_element(lidarObj, 'objWy', 1).text)
                    obj_z = float(_get_check_element(lidarObj, 'objWz', 1).text)
                    obj_yaw = float(_get_check_element(lidarObj, 'objAzimuth', 1).text)
                    obj_yaw = maxus2kitti(obj_yaw)
                    
                    # 车体坐标系 -> 相机坐标系
                    loc_lidar = [[float(obj_x), float(
                        obj_y), float(obj_z)]]
                    #             print(loc_lidar)
                    loc_lidar = np.asarray(loc_lidar)
                    R = np.asarray(camera_intrinsic["R"])
                    T = np.asarray(camera_intrinsic["T"])
                    joint_num = len(loc_lidar)
                    joint_cam = np.zeros((joint_num, 3))  # joint camera
                    for i in range(joint_num):  # joint i
                        joint_cam[i] = np.dot(R, loc_lidar[i]) + T  # R * (pt - T)
                    #             print(joint_cam)#[[ -4.2730145   -1.04935013 172.87431682]]
                    # 相机坐标系 -> 像素坐标系
                    obj_x = float(joint_cam[0][0])
                    obj_y = float(joint_cam[0][1])
                    obj_z = float(joint_cam[0][2])
                    obj_alpha = obj_yaw - math.atan2(obj_z,obj_x)
                    calib = camera_intrinsic["calib"]
                    f, c = [0.0, 0.0], [0.0, 0.0]
                    # print("calib[0, 0]",calib[0, 0])
                    f[0] = calib[0, 0]
                    f[1] = calib[1, 1]
                    c[0] = calib[0, 2]
                    c[1] = calib[1, 2]
                    u = obj_x / obj_z * f[0] + c[0]
                    v = obj_y / obj_z * f[1] + c[1]
                    d = obj_z
                    u = round(u, 4)
                    v = round(v, 4)
                    d = round(d, 4)

                    real_3d = dict(x=u, y=v, z=d, ox=obj_x, oy=obj_y, w=obj_w, h=obj_h, l=obj_l, occluded = int(attr_value_dis) ,alpha=obj_alpha, yaw= obj_yaw)
            except:
                pass

            #可见线
            ytop_vis,ybuttom_vis,xlft_vis,xrght_vis=None,None,None,None #可见线，顺序：上下左右
            vislines=[ytop_vis,ybuttom_vis,xlft_vis,xrght_vis]
            
            obj_annotation = dict(
                uuid=uuid,
                name=name,
                bndbox=numeric_bndbox,
                attributes=attrs_dict,
                cubepoints=cubepoints,
                real_3d=real_3d,
                vislines=vislines,
                vehicle_id=vehicle_id
            )
            annotation['annotation'].append(obj_annotation)
    except:
        pass

    # parse <PD>
    try:
        pd = _get_check_element(labelinfo, 'PD', 1)
        objs = pd.findall('Object')
        numeric_bndbox = None
        for obj in objs:
            labeltype = obj.attrib['labelType']
            if labeltype == 'QPOS':  # 不管全样本
                continue
            uuid = int(obj.attrib['ID'])
            name = "PD"
            vehicle_id=-1
            # 读取属性
            attrs_dict = dict()
            if labeltype != "QPOS":
                attr_tag_dis = "DisplaySituation"
                try:
                    attr_value_dis = obj.attrib['isCorvered']
                except:
                    attr_value_dis = obj.attrib['isCovered']
                attrs_dict[attr_tag_dis] = attr_value_dis
                attr_tag_dir = "direction"
                orientation = None
                attrs_dict[attr_tag_dir] = orientation
            
            # 获取一些坐标信息
            bodyrectx = float(obj.attrib['bodyRectX'])
            bodyrecty = float(obj.attrib['bodyRectY'])
            bodyrectwidth = float(obj.attrib['bodyRectWidth'])
            bodyrectheight = float(obj.attrib['bodyRectHeight'])
            ytop_vis,ybuttom_vis,xlft_vis,xrght_vis=None,None,None,None #可见线，顺序：上下左右
            if labeltype != "QPOS":
                ytop_vis=float(obj.attrib['yTop_vis'])
                ybuttom_vis=float(obj.attrib['yButtom_vis'])
            vislines=[ytop_vis,ybuttom_vis,xlft_vis,xrght_vis]

            # 2d bbox
            numeric_bndbox = [bodyrectx, bodyrecty, bodyrectx +
                            bodyrectwidth - 1, bodyrecty + bodyrectheight - 1]
#             print(numeric_bndbox)
            # 2d bbox protect
            edge_protect = False
            if edge_protect:
                X_MIN, X_MAX, Y_MIN, Y_MAX = 0, imgwidth, 0, imgheight
                numeric_bndbox[0] = X_MIN if numeric_bndbox[0] < X_MIN else X_MAX if numeric_bndbox[0] > X_MAX else numeric_bndbox[0]
                numeric_bndbox[2] = X_MIN if numeric_bndbox[2] < X_MIN else X_MAX if numeric_bndbox[2] > X_MAX else numeric_bndbox[2]
                numeric_bndbox[1] = Y_MIN if numeric_bndbox[1] < Y_MIN else Y_MAX if numeric_bndbox[1] > Y_MAX else numeric_bndbox[1]
                numeric_bndbox[3] = Y_MIN if numeric_bndbox[3] < Y_MIN else Y_MAX if numeric_bndbox[3] > Y_MAX else numeric_bndbox[3]

            # 2.5d points
            cubepoints = None

            # 3d
            real_3d = dict()
            try:
                lidarObj = _get_check_element(obj, 'lidarObj', 1)
                if lidarObj.attrib['validFlag'] == '1':
                    obj_l = float(_get_check_element(lidarObj, 'objLength', 1).text)
                    obj_w = float(_get_check_element(lidarObj, 'objWidth', 1).text)
                    obj_h = float(_get_check_element(lidarObj, 'objHeight', 1).text)
                    obj_x = float(_get_check_element(lidarObj, 'objWx', 1).text)
                    obj_y = float(_get_check_element(lidarObj, 'objWy', 1).text)
                    obj_z = float(_get_check_element(lidarObj, 'objWz', 1).text)
                    obj_yaw = float(_get_check_element(
                        lidarObj, 'objAzimuth', 1).text)
                    obj_yaw = maxus2kitti(obj_yaw)
                    # obj_alpha = None
                    # 车体坐标系 -> 相机坐标系
                    loc_lidar = [[float(obj_x), float(
                        obj_y), float(obj_z)]]
                    #             print(loc_lidar)
                    loc_lidar = np.asarray(loc_lidar)
                    R = np.asarray(camera_intrinsic["R"])
                    T = np.asarray(camera_intrinsic["T"])
                    joint_num = len(loc_lidar)
                    joint_cam = np.zeros((joint_num, 3))  # joint camera
                    for i in range(joint_num):  # joint i
                        joint_cam[i] = np.dot(R, loc_lidar[i]) + T  # R * (pt - T)
                    #             print(joint_cam)#[[ -4.2730145   -1.04935013 172.87431682]]
                    # 相机坐标系 -> 像素坐标系
                    obj_x = float(joint_cam[0][0])
                    obj_y = float(joint_cam[0][1])
                    obj_z = float(joint_cam[0][2])
                    obj_alpha = obj_yaw - math.atan2(obj_z,obj_x)
                    calib = camera_intrinsic["calib"]
                    f, c = [0.0, 0.0], [0.0, 0.0]
                    # print("calib[0, 0]",calib[0, 0])
                    f[0] = calib[0, 0]
                    f[1] = calib[1, 1]
                    c[0] = calib[0, 2]
                    c[1] = calib[1, 2]
                    u = obj_x / obj_z * f[0] + c[0]
                    v = obj_y / obj_z * f[1] + c[1]
                    d = obj_z
                    u = round(u, 4)
                    v = round(v, 4)
                    d = round(d, 4)

                    real_3d = dict(x=u, y=v, z=d, ox=obj_x, oy=obj_y, w=obj_w, h=obj_h, l=obj_l, occluded = int(attr_value_dis) ,alpha=obj_alpha, yaw= obj_yaw)
            except:
                pass
            
            obj_annotation = dict(
                uuid=uuid,
                name=name,
                bndbox=numeric_bndbox,
                attributes=attrs_dict,
                cubepoints=cubepoints,
                real_3d=real_3d,
                vislines=vislines,
                vehicle_id=vehicle_id
            )
            annotation['annotation'].append(obj_annotation)
    except:
        pass
    # if printpic == True:
    #     print("\n", image_name,
    #           "has been printed. Please check the corresponding folder.")
    #     cv2.imwrite(
    #         '/workspace/GroupShare/testdata/demo-3d/cornercase/' + image_name, img)
    
    # parse <Crane>
    try:
        crane = _get_check_element(labelinfo, 'Crane', 1)
        objs = crane.findall('Object')
        for obj in objs:
            labeltype = obj.attrib['labelType']
            if labeltype == 'QPOS':  # 不管全样本
                continue
#                 label=0
            else:
                label=1

            name = "Crane"
            vehicle_id=-1
            uuid = int(obj.attrib['ID'])
            
            # 读取属性
            attrs_dict = dict()
            if labeltype != "QPOS":
                attr_tag_dis = "DisplaySituation"
                attr_value_dis = 2
                attrs_dict[attr_tag_dis] = attr_value_dis
                attr_tag_dir = "direction"
                orientation = obj.attrib['orientation']
                attrs_dict[attr_tag_dir] = orientation
            else:
                orientation = None

            # 获取一些坐标信息
            bodyrectx = float(obj.attrib['bodyRectX'])
            bodyrecty = float(obj.attrib['bodyRectY'])
            bodyrectwidth = float(obj.attrib['bodyRectWidth'])
            bodyrectheight = float(obj.attrib['bodyRectHeight'])
            ytop_vis,ybuttom_vis,xlft_vis,xrght_vis=None,None,None,None #可见线，顺序：上下左右
            if labeltype != "QPOS":
                sidecorupx = float(obj.attrib['sideCorUpX'])
                sidecorupy = float(obj.attrib['sideCorUpY'])
                sidecordownx = float(obj.attrib['sideCorDownX'])
                sidecordowny = float(obj.attrib['sideCorDownY'])
            vislines=[ytop_vis,ybuttom_vis,xlft_vis,xrght_vis]

            # 2d bbox
            numeric_bndbox = None
            if orientation == 'FRONT' or orientation == 'REAR':
                numeric_bndbox = [bodyrectx, bodyrecty, bodyrectx + bodyrectwidth - 1, bodyrecty + bodyrectheight - 1]
            elif orientation == 'LEFT' or orientation == 'RIGHT':
                numeric_bndbox = [bodyrectx, bodyrecty, sidecordownx, sidecordowny]
            # elif orientation == 'RIGHT':
            #     numeric_bndbox = [sidecorupx, sidecorupy, bodyrectx, bodyrecty + bodyrectheight - 1]
            elif orientation == 'LEFT_REAR' or orientation == 'RIGHT_FRONT':
                numeric_bndbox = [bodyrectx, min(bodyrecty,sidecorupy), max(sidecorupx, sidecordownx), max((bodyrecty + bodyrectheight - 1),sidecordowny)]
            elif orientation == 'LEFT_FRONT' or orientation == 'RIGHT_REAR':
                numeric_bndbox = [min(sidecorupx, sidecordownx), min(bodyrecty,sidecorupy), bodyrectx + bodyrectwidth - 1,max((bodyrecty + bodyrectheight - 1),sidecordowny)]
            else:
                # print("orientation error")
                if labeltype == "QPOS":
                    numeric_bndbox = [bodyrectx, bodyrecty, bodyrectx + bodyrectwidth - 1, bodyrecty + bodyrectheight - 1]
                else:
                    numeric_bndbox = [min(sidecorupx,sidecordownx,bodyrectx),min(bodyrecty,sidecorupy),max(sidecorupx, sidecordownx, (bodyrectx + bodyrectwidth - 1)),max((bodyrecty + bodyrectheight - 1),sidecordowny)]
#             print(numeric_bndbox)
            
            # 2d bbox protect
            edge_protect = False
            if edge_protect:
                X_MIN, X_MAX, Y_MIN, Y_MAX = 0, imgwidth, 0, imgheight
                numeric_bndbox[0] = X_MIN if numeric_bndbox[0] < X_MIN else X_MAX if numeric_bndbox[0] > X_MAX else numeric_bndbox[0]
                numeric_bndbox[2] = X_MIN if numeric_bndbox[2] < X_MIN else X_MAX if numeric_bndbox[2] > X_MAX else numeric_bndbox[2]
                numeric_bndbox[1] = Y_MIN if numeric_bndbox[1] < Y_MIN else Y_MAX if numeric_bndbox[1] > Y_MAX else numeric_bndbox[1]
                numeric_bndbox[3] = Y_MIN if numeric_bndbox[3] < Y_MIN else Y_MAX if numeric_bndbox[3] > Y_MAX else numeric_bndbox[3]


            # 2.5d points
            cubepoints = None
            if labeltype != "QPOS":
                #接地三个点
                pointa = (bodyrectx, bodyrecty + bodyrectheight - 1)
                pointb = (bodyrectx + bodyrectwidth - 1, bodyrecty + bodyrectheight - 1)
                pointc = (sidecordownx, sidecordowny)
                #接地点对应的上面三个点
                pointau = (bodyrectx, bodyrecty)
                pointbu = (bodyrectx + bodyrectwidth - 1, bodyrecty)
                pointcu = (sidecorupx, sidecorupy)
                
                cubepoints = []
                cubepoints.append(dict(name="pointa", location=[pointa[0], pointa[1]]))
                cubepoints.append(dict(name="pointb", location=[pointb[0], pointb[1]]))
                cubepoints.append(dict(name="pointc", location=[pointc[0], pointc[1]]))
                cubepoints.append(dict(name="pointau", location=[pointau[0], pointau[1]]))
                cubepoints.append(dict(name="pointbu", location=[pointbu[0], pointbu[1]]))
                cubepoints.append(dict(name="pointcu", location=[pointcu[0], pointcu[1]]))

            # 3d
            real_3d = dict()

            obj_annotation = dict(
                uuid=uuid,
                name=name,
                label=label,
                bndbox=numeric_bndbox,
                attributes=attrs_dict,
                cubepoints=cubepoints,
                real_3d=real_3d,
                vislines=vislines,
                vehicle_id=vehicle_id
            )
            annotation['annotation'].append(obj_annotation)
            # print(annotation)
    except:
        pass
    
    
    if len(annotation['annotation']) == 0:
        return None

#     print(annotation)

    return annotation
