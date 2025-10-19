"""
目的: 根据已知的R和t将原始无方向无尺度点云换到真实世界坐标系下来

要注意我们需要的是colmap的完整的一个工程 / 包括sfm步骤和mvs步骤

需要读outs.txt / 所以在之前就需要执行对齐的操作, 计算出R和t

执行代码: python transform_PCD.py --project_folder /D/data/zt/project/colmap/test_no_gcp --align_output_path /home/zt/Project/COLMAP_GroundControlPoints-main/output/outs.txt
"""
import numpy as np
import os
import copy
from scipy.spatial.transform import Rotation
import open3d as o3d
import argparse

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

def trans_pcd(ori_pcd_path, proc_pcd_path, R, t):
    """
    input:
        1)原始重建的点云模型路径
        2)通过基于gcp的相关算法计算得到的R和t
    output:
        只是做一个print
        "变换成功,点云模型存储在proc_pcd_path"
    """
    print("··········开始进行坐标变换··········")
    ori_pcd  = o3d.io.read_point_cloud(ori_pcd_path)

    # 获得一下原始点坐标矩阵
    points_array = np.asarray(ori_pcd.points)

    # 旋转点云坐标矩阵
    points_array_correct = (np.dot(R,points_array.T) + t).T

    # 生成旋转后的proc_pcd
    proc_pcd = copy.deepcopy(ori_pcd)
    proc_pcd.points = o3d.utility.Vector3dVector(points_array_correct)

    # 保存pcd，打印输出
    o3d.io.write_point_cloud(proc_pcd_path, proc_pcd)
    print("坐标变换成功,变换后的点云模型存储在“{}”位置".format(proc_pcd_path))


# 传入重建项目的文件夹的路径
parser = argparse.ArgumentParser()
parser.add_argument("-p", "--project_folder", help = "colmap重建项目的文件夹路径,示例/D/data/zt/project/colmap/test_no_gcp", required=True)
parser.add_argument("-a", "--align_output_path", default="/home/zt/Project/COLMAP_GroundControlPoints-main/output/outs.txt", required=True)
args = parser.parse_args()

ori_pcd_path = os.path.join(args.project_folder,"dense/result.ply")
proc_pcd_path = os.path.join(args.project_folder,"dense/dense_proc.pcd")

if __name__ == "__main__":
    R, t = read_R_t(args.align_output_path)
    trans_pcd(ori_pcd_path, proc_pcd_path, R, t)
