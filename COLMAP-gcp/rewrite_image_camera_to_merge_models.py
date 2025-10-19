"""
当我们希望把两次重建的模型简单的合在一起(假设此时已经做好匹配了)然后去统一的转成mesh
这个时候我们的model其实很简单，所有的二维点和三维点都是空的。！！！所以也不涉及到别的文件里的问题
如果不去做任何更改,Image_id会相互冲突
因此希望把images.txt中的image_id依次排列就可以了。
逻辑顺序：
1、读取model 1的总共image数量 N1
2、对于model 2中的images.txt中的image_id 都加上 N1

在前置和后置的文件中,都需要将bin文件先转化为txt,然后再把txt转回bin
"""
import os
import sys
import sqlite3
import numpy as np
import copy
import argparse
###################################################
############### 把两部分的cameras拼起来 ##############
###################################################
def prepare_model1_cameras(model1_path):
    """
    input:model1的位置
    output:修改后的cameras的下面内容
    """
    # 先确认images.txt的位置
    caeras1_path = os.path.join(model1_path,"cameras.txt")
    
    # 先把caeras1这个文件给读出来
    with open(caeras1_path, 'r') as f:
        lines = f.readlines()
        lines_content = lines[3:]  # 每张image的具体位姿信息
    
    # 读出camera的内容
    correct_lines_content = copy.deepcopy(lines_content)
        
    return correct_lines_content


def prepare_model2_cameras(model1_path, model2_path):
    """
    input:两个sparse model的位置
    output:修改后的model2的cameras内容
    """
    # 先确认cameras.txt的位置
    cameras1_path = os.path.join(model1_path,"cameras.txt")
    cameras2_path = os.path.join(model2_path,"cameras.txt")

    # 先读取cameras1中的图片数量，其实就是读取txt文件中的第三行本身就写好的camera number
    with open(cameras1_path, 'r') as f:
        lines = f.readlines()
        contain_number_line = lines[2]  # 包含image number的那一行
        contain_number_line = contain_number_line.strip()
        cameras_num_1 = int(contain_number_line.split(":")[1][1:])
    print("model1的相机数量为{}个".format(cameras_num_1))
    
    # 先把cameras2这个文件给读出来
    with open(cameras2_path, 'r') as f:
        lines = f.readlines()
        lines_head = lines[:3]  # 文件头里附加的一些内容
        lines_content = lines[3:]  # 每张image的具体位姿信息
    
    # 修改调整camera数量
    cameras_num_2 = lines_head[-1].split(",")[0].split(":")[1][1:]
    full_cameras_num = cameras_num_1 + int(cameras_num_2)
    lines_head[2] = "# Number of cameras: {}\n".format(str(full_cameras_num))

    # 修改调整cameraID
    correct_lines_content = copy.deepcopy(lines_content)

    for c,line in enumerate(lines_content):  # 选取image param的那一行
        line = line.strip()
        ori_camera_ID = int(line.split(" ")[0])
        new_camera_ID = ori_camera_ID + cameras_num_1
        # 替换正确的图片编号，并且替换lines内容
        correct_line = str(new_camera_ID) + line[len(str(ori_camera_ID)):] + "\n"
        correct_lines_content[c] = correct_line

    # 确保images.txt最后一定有个空行
    if correct_lines_content[-1] != "\n":
        correct_lines_content.append("\n")

    return cameras_num_1, lines_head, correct_lines_content


def merge_cameras(model1_path, model2_path, merge_path):
    """
    输出:将model1和model2的cameras写在一起
    """
    cameras_merge_path = os.path.join(merge_path, "cameras.txt")

    cameras1_content = prepare_model1_cameras(model1_path)
    cameras1_num, cameras2_head, cameras2_content = prepare_model2_cameras(model1_path, model2_path)  # 这里读到的是修改后的内容

    with open(cameras_merge_path, "w") as new_cameras_file:
        new_cameras_file.writelines(cameras2_head)
        new_cameras_file.writelines(cameras1_content[:])
        new_cameras_file.writelines(cameras2_content[:])
    
    print("融合的cameras文件写入完毕")


###################################################
############### 把两部分的images拼起来 ##############
###################################################
###### 这里还差camera_ID的修改 #########
def prepare_model1_images(model1_path):
    """
    input:model1的位置
    output:修改后的images的下面内容
    """
    # 先确认images.txt的位置
    images1_path = os.path.join(model1_path,"images.txt")
    
    # 先把images2这个文件给读出来
    with open(images1_path, 'r') as f:
        lines = f.readlines()
        lines_content = lines[4:]  # 每张image的具体位姿信息
    
    # 修改调整imageID
    correct_lines_content = copy.deepcopy(lines_content)

    for c,line in enumerate(lines_content):
        if c%2 == 0:  # 如果是有相机外参的那一行，直接抄下来
            correct_lines_content[c] = line
        
        else:  # 如果不含相机外参，就直接变成空行
            correct_line = "\n"
            correct_lines_content[c] = correct_line

    # 确保images.txt最后一定有个空行
    if correct_lines_content[-1] != "\n":
        correct_lines_content.append("\n")

    # with open(images1_path, "w") as new_images_file:
    #     new_images_file.writelines(lines_head)
    #     new_images_file.writelines(correct_lines_content[:])
        
    return correct_lines_content


def prepare_model2_images(model1_path, model2_path):
    """
    input:两个sparse model的位置
    output:修改后的model2的images内容
    """
    # 先确认images.txt的位置
    images1_path = os.path.join(model1_path,"images.txt")
    images2_path = os.path.join(model2_path,"images.txt")

    # 先读取images1中的图片数量，其实就是读取txt文件中的第四行本身就写好的image number
    with open(images1_path, 'r') as f:
        lines = f.readlines()
        contain_number_line = lines[3]  # 包含image number的那一行
        part1 = contain_number_line.split(",")[0]
        images_num_1 = int(part1.split(":")[1][1:])
    print("model1的图片数量为{}张".format(images_num_1))

    # 再读取cameras1中的相机数量
    cameras1_num, _, _ = prepare_model2_cameras(model1_path, model2_path)
    
    # 先把images2这个文件给读出来
    with open(images2_path, 'r') as f:
        lines = f.readlines()
        lines_head = lines[:4]  # 文件头里附加的一些内容
        lines_content = lines[4:]  # 每张image的具体位姿信息
    
    # 修改调整image数量
    images_num_2 = lines_head[-1].split(",")[0].split(":")[1][1:]
    full_images_num = images_num_1 + int(images_num_2)
    lines_head[3] = "# Number of images: {}, mean observations per image: unknown\n".format(str(full_images_num))

    # 修改调整imageID和cameraID
    correct_lines_content = copy.deepcopy(lines_content)

    for c,line in enumerate(lines_content):  # 选取image param的那一行
        if c%2 == 0:  # 如果是有相机外参的那一行，就改变image_id和cameraID
            line = line.strip()
            IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME = line.split(" ")
            new_img_ID = int(IMAGE_ID) + images_num_1
            new_camera_ID = int(CAMERA_ID) + cameras1_num
            new_name = NAME[:-4]+"_model2"+NAME[-4:]  # 防止两个model的图片的命名有重复，所以给第二个模型的图片改一下名字
            # 替换正确的图片编号和camera编号
            correct_line = "{} {} {} {} {} {} {} {} {} {}\n".format(new_img_ID, QW, QX, QY, QZ, TX, TY, TZ, new_camera_ID, new_name)
            correct_lines_content[c] = correct_line
        
        else:  # 如果不含相机外参，就直接变成空行
            correct_line = "\n"
            correct_lines_content[c] = correct_line

    # 确保images.txt最后一定有个空行
    if correct_lines_content[-1] != "\n":
        correct_lines_content.append("\n")

    # # 修改images文件
    # with open(images2_path, "w") as new_images_file:
    #     new_images_file.writelines(lines_head)
    #     new_images_file.writelines(correct_lines_content[:])

    # # 创建一个空的images3D文件
    # with open(os.path.join(model1_path,"points3D.txt"), "w") as f:
    #     pass

    return lines_head, correct_lines_content


def merge_images(model1_path, model2_path, merge_path):
    """
    输出:将model1和model2的images写在一起
    """
    images_merge_path = os.path.join(merge_path, "images.txt")

    images1_content = prepare_model1_images(model1_path)
    images2_head, images2_content = prepare_model2_images(model1_path, model2_path)  # 这里读到的是修改后的内容

    with open(images_merge_path, "w") as new_images_file:
        new_images_file.writelines(images2_head)
        new_images_file.writelines(images1_content[:])
        new_images_file.writelines(images2_content[:])
    
    print("融合的images文件写入完毕")

###################################################
############### 把两部分的images拼起来 ##############
###################################################

def merge_points3D(merge_path):
    """
    创建一个空的points3D文件
    """
    with open(os.path.join(merge_path,"points3D.txt"), "w") as f:
        pass

if __name__ == '__main__':
    # 传入重建项目的文件夹的路径
    parser = argparse.ArgumentParser()
    parser.add_argument("--model1_path", help = "第一个模型的位置,示例/D/data/zt/project/colmap/cuptest/sparse/model/0", required=True)
    parser.add_argument("--model2_path", help = "第二个模型的位置,示例/D/data/zt/project/colmap/boardtest/sparse/model/0", required=True)
    parser.add_argument("--merge_path", help = "merge模型的位置,示例/D/data/zt/project/colmap/cup_board_merge_model", required=True)
    args = parser.parse_args()

    print("----------创建融合的cameras文件------------")
    merge_cameras(args.model1_path, args.model2_path, args.merge_path)
    print("----------创建融合的images文件------------")
    merge_images(args.model1_path, args.model2_path, args.merge_path)
    print("----------创建融合的points3D文件(空)------------")
    merge_points3D(args.merge_path)



