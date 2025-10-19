import numpy as np
import os
import copy
from scipy.spatial.transform import Rotation
import argparse
import shutil

############### 比较坐标转换前后的相机光心 ##############
from sklearn.linear_model import RANSACRegressor
from sklearn.svm import SVR
# pip install scikit-learn
import math

#################################################
############读取原始和转变后的光心坐标#############
#################################################
old_images_path = "/D/data/zt/project/colmap/boardtest/sparse/model/0/images.txt"
new_images_path = "/D/data/zt/project/colmap/boardtest/sparse/model/manual/images.txt"

# 读原始的光心坐标
with open(old_images_path, "r") as f:
    lines = f.readlines()
old_cc_array = np.zeros((1,3))
for index, line in enumerate(lines[4::2]):
    # 读取images的外参
    param = [float(num) for num in line.split()[1:8]]
    qw, qx, qy, qz, tx, ty, tz = param
    q = np.array([qx, qy, qz, qw])  # 用scipy的形式,行向量
    t = np.array([tx, ty, tz]).reshape(-1, 1)

    # 根据colmap自己的规则，计算光心坐标
    R = Rotation.from_quat(q).as_matrix()
    t1 = -np.dot(np.transpose(R), t)

    # 拼成一个n*3的矩阵
    if index == 0:
        old_cc_array = t1.T
    else:
        old_cc_array = np.vstack((old_cc_array, t1.T))

# 读转变后的光心坐标
with open(new_images_path, "r") as f:
    lines = f.readlines()
new_cc_array = np.zeros((1,3))
for index, line in enumerate(lines[4::2]):
    # 读取images的外参
    param = [float(num) for num in line.split()[1:8]]
    qw, qx, qy, qz, tx, ty, tz = param
    q = np.array([qx, qy, qz, qw])  # 用scipy的形式,行向量
    t = np.array([tx, ty, tz]).reshape(-1, 1)

    # 根据colmap自己的规则，计算光心坐标
    R = Rotation.from_quat(q).as_matrix()
    t1 = -np.dot(np.transpose(R), t)
    

    # 拼成一个n*3的矩阵
    if index == 0:
        new_cc_array = t1.T
    else:
        new_cc_array = np.vstack((new_cc_array, t1.T))

def read_R_t(align_output_path):
    """
    这里输入的是做完align之后的outs.txt位置
    示例: '/home/zt/Project/COLMAP_GroundControlPoints-main/output/outs.txt'
    """

    with open(align_output_path, 'r') as file:
        lines = file.readlines()
        # 提取缩放因子scale数据
        scale = float(lines[1].strip().split()[1])

        # 提取Transformation matrix数据
        matrix_lines = lines[3:7]  # 提取第4到第7行的数据
        transformation_matrix = []
        for line in matrix_lines:
            matrix_row = [float(num) for num in line.strip().split()]
            transformation_matrix.append(matrix_row)
        
        # 将转换矩阵存储为矩阵形式
        transformation_matrix = np.array(transformation_matrix)
        R = transformation_matrix[0:3,0:3]
        t = transformation_matrix[0:3,3].reshape(3,1)

    return R,t

R, t = read_R_t("/home/zt/Project/COLMAP_GroundControlPoints-main/output/outs.txt")

pred_new_cc_array = np.transpose(np.dot(R, old_cc_array.T) + t)

print(pred_new_cc_array, new_cc_array)



