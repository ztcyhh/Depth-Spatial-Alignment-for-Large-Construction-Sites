"""
目的: 将相机位姿从某个初始化的相机坐标系转到世界坐标系下来。
这里用的是GCP得到的R和t
根据已知的R和t去改变相机位姿, 并且写到images.txt中
代码运行示例
python transform_images.py --project_folder /D/data/zt/project/colmap/test_no_gcp --align_output_path /home/zt/Project/COLMAP_GroundControlPoints-main/output/outs.txt
"""
import numpy as np
import os
import copy
from scipy.spatial.transform import Rotation
import argparse
import shutil


def read_R_t():
    """
    这里输入的是手动对齐得到的T矩阵
    输出: 世界坐标系变换的R和t
    """

    # 将转换矩阵存储为矩阵形式
    transformation_matrix = np.array([[0.224067211151,0.869423747063,-0.684582471848,6.478889465332],
                                      [-1.036455035210,0.409600019455,0.180957585573,-1.182596802711],
                                      [0.387700527906,0.592526018620,0.879407823086,0.000241063171],
                                      [0.000000000000,0.000000000000,0.000000000000,1.000000000000]])
    R = transformation_matrix[0:3,0:3]
    t = transformation_matrix[0:3,3].reshape(3,1)

    return R,t

def matrix2quaternion(Rm):
    """
    输入:旋转矩阵,matrix-numpy形式
    输出:四元数,array行向量形式
    """
    q = Rotation.from_matrix(Rm).as_quat()
    return q

def quaternion2matrix(q):
    """
    输入:四元数,list或array行向量形式
    输出:旋转矩阵,matrix-numpy形式
    """
    R = Rotation.from_quat(q).as_matrix()
    return R

def matrix_norm(R,t):
    """
    可能旋转矩阵不是正交的，这个不是标准型，要换到标准型下来
    """
    R_orth, _ = np.linalg.qr(R)  # 正交化
    R_orth = np.sign(R) * np.abs(R_orth)  # 保证符号相同
    # 调整t
    t_new = R_orth @ np.linalg.inv(R) @ t
    
    return R_orth, t_new


def trans_images_txt(images_path, new_images_path, R_o2n, t_o2n):
    """
    读取原有的images文件夹,并调整到新世界坐标系下来
    """
    print("----------------------------------")
    print("开始改写旧的images文件,具体存储位置在{}".format(images_path))
    print("----------------------------------")
    # 读取原有的images文件，形成一个images外参列表
    images_param_and_ori = [] # It will contain IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME for each image
    #new_file = open('{}'.format(final_images), 'w')
    with open(images_path, 'r') as f:
        lines = f.readlines()
        lines_head = lines[:4]  # 文件头里附加的一些内容
        lines_content = lines[4:]  # 每张image的具体位姿信息

        for c,line in enumerate(lines_content):
            if c%2 == 0:
                line = line.strip()
                img_params = line.split(" ", 9)
                images_param_and_ori.append(img_params)
    
    # 准备开始创建新的images文件
    images_new = copy.deepcopy(images_param_and_ori)

    for i, param in enumerate(images_param_and_ori):
        # 1、读取原始的旋转矩阵和平移矩阵
        _, QW, QX, QY, QZ, TX, TY, TZ, _, _ = param
        q_o = np.array([QX, QY, QZ, QW]).astype(np.float32)
        t_o = np.array([TX, TY, TZ]).reshape(3,1).astype(np.float32)
        R_o = quaternion2matrix(q_o)  # 四元数转矩阵形式

        # 2、生成新的旋转四元数和平移矩阵
        # 不妨认为R_o是描述相机坐标系相对于世界坐标系，也就是Pc = R * Pw + t
        R_n = np.dot(R_o,np.linalg.inv(R_o2n))
        t_n = -np.dot(np.dot(R_o,np.linalg.inv(R_o2n)),t_o2n) + t_o  # 验证下来应该是对的
        # 正交化处理
        R_n, t_n = matrix_norm(R_n, t_n)

        q_n = matrix2quaternion(R_n)  # 顺序是qx，qy，qz，qw

        # 3、创建新的images列表
        images_new[i][1] = str(q_n[3]) # QW
        images_new[i][2] = str(q_n[0]) # QX
        images_new[i][3] = str(q_n[1]) # QY
        images_new[i][4] = str(q_n[2]) # QZ
        images_new[i][5] = str(t_n[0,0]) # TX
        images_new[i][6] = str(t_n[1,0]) # TY
        images_new[i][7] = str(t_n[2,0]) # TZ

    # 写成新的images文件
    with open(new_images_path,"w") as f:
        f.writelines(lines_head)
        for new_param in images_new:
            f.write("{} {} {} {} {} {} {} {} {} {}\n\n".format(*new_param))

    print("----------------------------------")
    print("已改写成新的images文件(不包含points2D,只有相机位姿)\n具体存储位置在{}".format(new_images_path))
    print("----------------------------------")

def create_new_camera_points3D(old_images_path, new_images_path):
    # 准备好原始cameras的路径（points3D不需要）
    old_model_path = os.path.dirname(old_images_path)
    old_camera_path = os.path.join(old_model_path, "cameras.txt")

    # 准备好新的cameras和points3D的路径
    new_model_path = os.path.dirname(new_images_path)
    new_camera_path = os.path.join(new_model_path, "cameras.txt")
    new_points3D_path = os.path.join(new_model_path, "points3D.txt")

    # 创建新的cameras文件（其实直接复制就可以了，关系不是太大）
    if os.path.exists(new_camera_path):
        os.remove(new_camera_path)
    shutil.copy(old_camera_path, new_camera_path)
    print("-------------\n创建了一个新的cameras文件\n-----------------")

    # 创建新的points3D文件（直接一个空的文件就可以了） / 但是这样不能看到他对不对，可能还得转换一下原始的points3D？
    with open(new_points3D_path,"w") as f:
        print("-------------\n创建了一个新的points3D文件\n-----------------")

# 传入重建项目的文件夹的路径
parser = argparse.ArgumentParser()
parser.add_argument("-p", "--project_folder", help = "colmap重建项目的文件夹路径,示例/D/data/zt/project/colmap/cuptest", required=True)
args = parser.parse_args()

images_path = os.path.join(args.project_folder,"sparse/images.txt")
if not os.path.exists(os.path.join(args.project_folder, "sparse_manual")):
    os.mkdir(os.path.join(args.project_folder, "sparse_manual"))
new_images_path = os.path.join(args.project_folder,"sparse_manual/images.txt")

if __name__ == "__main__":
    R_o2n, t_o2n = read_R_t()
    trans_images_txt(images_path, new_images_path, R_o2n, t_o2n)
    create_new_camera_points3D(images_path, new_images_path)