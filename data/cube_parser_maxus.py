import os
from tqdm import tqdm


class CubeParser(object):
    def __init__(self):
        self.reset()
        # record errors' uuid
        self.errors = []
        # warnings, uuid
        self.warnings = []

    def reset(self):
        # Left Front Vertical Edge
        self.lfd = None
        self.lfu =None
        # Right Front Vertical Edge
        self.rfd = None
        self.rfu =None
        # Left Behind Vertical Edge
        self.lbd = None
        self.lbu =None
        # Right Behind Vertical Edge
        self.rbd = None
        self.rbu = None

        self.side_edge = None

    def _get_y_pos(self, x, side_edge):
        if side_edge[0] - side_edge[2] == 0:
            k = 0
        else:
            k = (side_edge[1] - side_edge[3]) / (side_edge[0] - side_edge[2])
        b = side_edge[1] - k * side_edge[0]
        return k * x + b

    def _get_x_pos(self, y, side_edge):
        if side_edge[1] - side_edge[3] == 0:
            k = 0
        else:
            k = (side_edge[0] - side_edge[2]) / (side_edge[1] - side_edge[3])
        b = side_edge[0] - k * side_edge[1]
        return k * y + b

    def get_y_pos(self, x, side_edge):
        y = self._get_y_pos(x, side_edge)
        if self.image_height is not None and y >= self.image_height:
            x2h = self._get_x_pos(self.image_height, side_edge)
            x = x2h
            y = self.image_height
        return x, y

    def paserLeft(self, cube):
        self.lfd = cube[0]['location']
        self.lbd = cube[2]['location']
        self.lfu = cube[3]['location']
        self.lbu = cube[5]['location']

        return [self.lfd, self.lbd, self.rfd, self.rbd, self.lfu, self.lbu, self.rfu, self.rbu]

    def parserRight(self, cube):
        # self.rfd = cube[0]['location']
        # self.rbd = cube[2]['location']
        # self.rfu = cube[3]['location']
        # self.rbu = cube[5]['location']

        self.rfd = cube[2]['location']
        self.rbd = cube[0]['location']
        self.rfu = cube[5]['location']
        self.rbu = cube[3]['location']

        return [self.lfd, self.lbd, self.rfd, self.rbd, self.lfu, self.lbu, self.rfu, self.rbu]

    def parserFront(self, cube):
        self.lfd = cube[1]['location']
        self.rfd = cube[0]['location']
        self.lfu = cube[4]['location']
        self.rfu = cube[3]['location']

        return [self.lfd, self.lbd, self.rfd, self.rbd, self.lfu, self.lbu, self.rfu, self.rbu]

    def parserBehind(self, cube):
        self.lbd = cube[0]['location']
        self.rbd = cube[1]['location']
        self.lbu = cube[3]['location']
        self.rbu = cube[4]['location']

        return [self.lfd, self.lbd, self.rfd, self.rbd, self.lfu, self.lbu, self.rfu, self.rbu]


    def parserLeftBehind(self, cube):

        self.lfd = cube[2]['location']
        self.lbd = cube[0]['location']
        self.rbd = cube[1]['location']
        self.lfu = cube[5]['location']
        self.lbu = cube[3]['location']
        self.rbu = cube[4]['location']

        return [self.lfd, self.lbd, self.rfd, self.rbd, self.lfu, self.lbu, self.rfu, self.rbu]

    def parserRightBehind(self, cube):

        self.lbd = cube[0]['location']
        self.rbd = cube[1]['location']
        self.rfd = cube[2]['location']
        self.lbu = cube[3]['location']
        self.rbu = cube[4]['location']
        self.rfu = cube[5]['location']

        return [self.lfd, self.lbd, self.rfd, self.rbd, self.lfu, self.lbu, self.rfu, self.rbu]

    def parserLeftFront(self, cube):

        self.rfd = cube[0]['location']
        self.lfd = cube[1]['location']
        self.lbd = cube[2]['location']
        self.rfu = cube[3]['location']
        self.lfu = cube[4]['location']
        self.lbu = cube[5]['location']

        return [self.lfd, self.lbd, self.rfd, self.rbd, self.lfu, self.lbu, self.rfu, self.rbu]

    def parserRightFront(self, cube):

        self.rbd = cube[2]['location']
        self.rfd = cube[0]['location']
        self.lfd = cube[1]['location']
        self.rbu = cube[5]['location']
        self.rfu = cube[3]['location']
        self.lfu = cube[4]['location']

        return [self.lfd, self.lbd, self.rfd, self.rbd, self.lfu, self.lbu, self.rfu, self.rbu]

    def __call__(self, obj_ann: dict, image_width=None, image_height=None):
        self.reset()
        self.image_width = image_width
        self.image_height = image_height

        if 'attributes' not in obj_ann.keys():
            self.errors.append(obj_ann['uuid'])
            return None

        if obj_ann['attributes'] is None:
            self.errors.append(obj_ann['uuid'])
            return None

        if 'direction' not in obj_ann['attributes'].keys():#Vehicle和PD全样本obj_ann['attributes']是{}，所以会return None
            self.errors.append(obj_ann['uuid'])
            return None
        
        if obj_ann['attributes']['direction'] is None: #Rider和PD正样本的属性是{'DisplaySituation': '0', 'direction': None} 所以会return None
            return None
        
        direction = obj_ann['attributes']['direction']
#         print('direction',direction)
        vedges = []
        
        if obj_ann['cubepoints']:#vehicle正样本会进来，如果方向是UNKONWN,返回的vedges是None
            # pointa = obj_ann['cubepoints'][0]['location']
            # pointb = obj_ann['cubepoints'][1]['location']
            # pointc = obj_ann['cubepoints'][2]['location']
            # pointau = obj_ann['cubepoints'][3]['location']
            # pointbu = obj_ann['cubepoints'][4]['location']
            # pointcu = obj_ann['cubepoints'][5]['location']
            #         print(pointa,pointb,pointc)

            if 'LEFT' == direction:
                vedges = self.paserLeft(obj_ann['cubepoints'])
            elif 'RIGHT' == direction:
                vedges = self.parserRight(obj_ann['cubepoints'])
            elif 'FRONT' == direction:
                vedges = self.parserBehind(obj_ann['cubepoints'])
            elif 'REAR' == direction:
                vedges = self.parserFront(obj_ann['cubepoints'])
            elif 'LEFT_FRONT' == direction:
                vedges = self.parserLeftBehind(obj_ann['cubepoints'])
            elif 'RIGHT_FRONT' == direction:
                vedges = self.parserRightBehind(obj_ann['cubepoints'])
            elif 'LEFT_REAR' == direction:
                vedges = self.parserLeftFront(obj_ann['cubepoints'])
            elif 'RIGHT_REAR' == direction:
                vedges = self.parserRightFront(obj_ann['cubepoints'])
            else:
                vedges = None

            if vedges is not None:
                for idx, vedge in enumerate(vedges):
                    if vedge is not None:
                        vedges[idx] = list(map(float, vedge))

#         print(vedges)
        return vedges

