# 测试

## 文件结构

```
test
├── test.py                 # 测试
├── pic_list.py             # 生成数据集图片路径txt
└── README.md
```

## 测试流程

### 1. 数据准备

对于需要测试的数据集，需生成包含所有图片路径的txt文件，即image_txt，可用如下脚本生成.

```
$ python test/pic_list.py --dataset [dataset_path] --output [image_txt_path]
```

若有calibration文件，默认calib_path下有与图片名对应的xml文件，否则需要修改代码中calib匹配方式(test/test.py, line: 815)。若无则传入固定的calib文件。

### 2. 测试

```
$ python test/test.py --image_txt   [image_txt_path]             
                          --output_dir  [output_dir]
                          --model_path  [model_path]
                          --calib_path  [calib_path]
                          --image_size  [image_width, image_height]
                          --crop        [lx, ly, rx, ry]
                          --input_size  [input_width, input_height]
                          --vis_25d 
                          --vis_3d
                          --vis_video
```

其中参数含义为

+ --image_txt: 包含测试图片路径的.txt文件路径

+ --output_dir: 输出路径

+ --model_path: 测试模型.pth路径

+ --calib_path[可选]: xml标注文件路径，默认为None

+ --image_size: 原始图片大小，**调用时输入--image_size width height，以下尺寸参数同理**

+ --crop: 原始图片上裁切roi区域对角坐标

+ --input_size[可选]: 模型输入大小，**aiv下默认为[640, 384]，hh需指定为[640, 320]**

+ --vis_25d[可选]: 输出推理2.5D可视化结果

+ --vis_3D[可选]: 输出推理3D可视化结果

+ --viz_video[可选]: 将可视化结果合并为视频

如：

```
$ python test/test_aiv.py --image_txt ./test_image.txt --output_dir ./output/ --model_path ./model_checkpoint_100.pth --image_size 1936 1152  --crop 8 28 1928 1152 --vis_25d --vis_video
```

### 3. 推理结果

```
output_dir
├── 3D         # 3D可视化结果   
├── 25D        # 2.5D可视化结果
├── det_txt    # 推理结果txt
├── 25D.avi    # 2.5D可视化结果视频
└── 3D.avi     # 3D可视化结果视频
```