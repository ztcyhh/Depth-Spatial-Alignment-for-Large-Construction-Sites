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

    return scale, R, t

scale, R, t = read_R_t("/home/zt/Project/COLMAP_GroundControlPoints-main/output/outs.txt")
print(np.linalg.det(R), np.linalg.det(R/scale))



