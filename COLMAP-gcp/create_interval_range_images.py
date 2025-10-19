"""
这个py文件主要用于从原始的360张图像中下采样出一些图片, 比如10张-20张-30张等等
同时还要准备相应的相机位姿文件
"""
import os
import math
import shutil

ori_root = "/D/data/zt/project/colmap/batch_exp_one/exp_R50_pitch40"
ori_image_path = os.path.join(ori_root, "images")
ue_pos_path = os.path.join(ori_root, "ue_pos.txt")
downsample_folder_path = "/D/data/zt/project/colmap/batch_exp_two"

with open(ue_pos_path,"r") as f:
    ue_pos_lines = f.readlines()

pic_name_list = sorted(os.listdir(ori_image_path))

for angle in range(0, 360, 10):
    angle += 10
    pic_num = 180 * angle / 360  # 这个角度范围内有多少张图片
    start_num = math.ceil(pic_num / 2)
    end_num = int(pic_num - start_num)

    start_pic_name_list = pic_name_list[:start_num]
    end_pic_name_list = pic_name_list[-1*end_num:]
    start_ue_pos_lines = ue_pos_lines[:start_num]
    end_ue_pos_lines = ue_pos_lines[-1*end_num:]

    # 生成所需范围内的images和相应的pos
    range_pic_name_list = start_pic_name_list + end_pic_name_list
    range_ue_pos_lines = start_ue_pos_lines + end_ue_pos_lines

    downsample_image_path = os.path.join(downsample_folder_path, "exp_R50_pitch40_angle{}".format(angle))
    os.mkdir(downsample_image_path)

    for pic_name in range_pic_name_list:
        shutil.copy(os.path.join(ori_image_path, pic_name), os.path.join(downsample_image_path, pic_name))

    with open(os.path.join(downsample_image_path, "ue_pos.txt"),"w") as f:
        f.writelines(range_ue_pos_lines)

print("over")