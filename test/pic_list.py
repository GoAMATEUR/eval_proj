import os
import os.path
import sys
import argparse
# def gen_pic_list(bag_path):
#     pics = []
#     pic_path = os.path.join(bag_path, 'record')
#     txt_path = os.path.join(bag_path, 'pic_list.txt')
#     for file in os.listdir(pic_path):
#         # print(file)
#         pics.append(file)
    
#     with open(txt_path, 'w') as f:
#         for pic in pics:
#             f.write(pic_path + '/' + pic + os.linesep)

def get_pic_list(path, idx):
    count = 0
    files = []
    for root, dirs, files in os.walk(path):
        print(root)
        for file in files:
            count += 1
            # print(file)
            if file.split(".")[-1] != "jpg":
                print(file)
                continue
            filepath = os.path.join(root, file)
            #files.append(filepath)
            with open(f"/home/uto/workspace/infer/bsd_ori/test_all.txt", 'a', encoding="utf8") as f:
                f.write(filepath.replace("uto", "utopilot").replace("night_city", "img") + os.linesep)
    print(path, count)

if __name__ == '__main__':
    # bag_path = '/workspace/Engineering/infer/CORNER_CASE/rosbag_2022_09_30_11_38_29/'
    

    bag_path = "/home/uto/workspace/infer/bsd_ori/bsd_ori/"
    get_pic_list(bag_path, i)