# **Evaluator**
## **文件结构**
```
eval
├── evaluator.py             # 包含评测Evaluator类
├── eval_utils              
│   ├── __init__.py
│   ├── label_parser.py      # LabelParser类处理标注，使得标注关键点、box顶点等在图片内
│   ├── eval_kitti_utils.py  # kitti_utils,有改动
│   ├── parse_results.py     # 保存测评结果为xls及视频
│   ├── eval_twobox_utils.py     
│   ├── eval_vis_two_box.py  # vis_two_box()合并车头车尾
└── └── README.md
```

## **评测方法**

### **1. 数据准备**

```
gt_path    = 标记txt文件所在文件夹路径
det_path   = 推理txt文件所在文件夹路径
test_file  = 记录参与评测的所有图片路径的txt文件，图片名与txt需对应
calib_path = 带相机内参的xml标注文件的文件夹路径(Optional)
```

### **2. 初始化`Evaluator`类**

```python
from evaluator import Evaluator
evaluator = Evaluator(test_file: str, save_dir: str, gt_path: str, det_path: str, img_shape: str，crop_coor: str, calib_path: str)
```
**参数:**

+ **test_file**: 记录参与评测的所有图片路径的txt文件.

+ **save_dir**: 评测结果保存文件夹路径.

+ **gt_path**: 标记txt文件所在文件夹路径.

+ **det_path**: 推理txt文件所在文件夹路径.

+ **img_shape**: 图片原始尺寸(w, h)，如(1936, 1220).

+ **crop_coor**: 裁切roi的对角顶点坐标，如[8, 28, 1928, 1220]. 

+ **calib_path(Optional)**: 带相机内参的xml标注文件的文件夹路径，默认为None.

**对于不同场景，只需改变图像尺寸及裁切尺寸.** 裁切方式为竖直方向由下而上裁切，水平方向中心裁切。如img_shape=(1936, 1220)，roi_shape=(1920, 1152)，裁切范围[8,68,1928,1220]

### **3. 运行评测**
调用```Evaluator```的```evaluate```方法进行评测。

```python
evaluator.evaluate(eval2D: bool, eval25D: bool, eval3D: bool, vizGT: bool, video:bool, channel: int, box_size_range:list, eval_cls:list)
```

**参数:**

+ **eval2D**: 是否进行2D评测.

+ **eval25D**: 是否进行2.5D评测

+ **eval3D**: 是否进行3D评测.

+ **vizGT**: 可视化2.5D推理结果时是否同时可视化真值.

+ **video**: 是否将2.5D推理可视化结果生成视频.

+ **box_size_range(Optional)**: 2D评测时根据box大小分段的界限. 如指定为[32, 96]，则分段为*32-96*、*>96*. 默认为[32, 96]

+ **channel(Optinal)**: 评测时是否根据深度分级，默认为1.

+ **eval_cls**: 参与测评的类别的列表，如["PD", "TRUCK"]，不指定则评测全部类别.

或直接运行：

```
$ python eval/evaluator.py  --det           [det_path]               
                            --gt            [gt_path]                     
                            --output        [save_dir]             
                            --image_txt     [test_file]      
                            --calib_path    [calib_path]
                            --vis_gt        [vizGT]     
                            --vis_video     [video] 
                            --eval_3d 
                            --eval_25d 
                            --eval_2d
                            --channel        [channel]
                            --box_size_range [range, ...]
                            --image_size     [img_width, img_height]
                            --crop           [lx, ly, rx, ry]
                            --cls            [cls, ...]
```

### **4. 评测结果**

若同时执行所有评测，则在输出路径下会得到以下结果:

```
save_dir
├── eval_2D_results.xls             # 2D测评结果
├── eval_25D_results.xls            # 2.5D测评结果
├── eval_3D_results.xls             # 3D测评结果
├── 25D.avi                         # 2.5D推理(与真值真值)可视化视频
├── 25D                             # 2.5D推理(真值)可视化图片
├── 3D                              # 3D 推理可视化图片
└── Metrics                         # 2D评测原始数据
```

+ **eval_2D_results.xls:** 包含根据类别与box大小分别统计的检出数量、漏检总数、真值总数、漏检率、误检总数、预测总数、误检率、Precision、Recall、box的中心位置及大小的相对、绝对误差。

+ **eval_3D_results.xls：** 包含3D目标深度误差、尺寸误差、偏航角误差

+ **eval_25D_results.xls：** 计算cube error. 包含真值距离、推理值底边两点距离、绝对误差、相对误差






