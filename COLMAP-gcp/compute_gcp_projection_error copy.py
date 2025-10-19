import numpy as np
from scipy.spatial import KDTree


# 读取points3D.txt文件并解析为numpy数组
def parse_points3D_file(file_path):
    points = []
    with open(file_path, 'r') as file:
        for line in file:
            # 跳过注释行
            if line.startswith('#') or len(line.strip()) == 0:
                continue
            # 按照空格分隔数据
            data = line.split()
            # 解析点的坐标 (X, Y, Z)
            point = [float(data[1]), float(data[2]), float(data[3])]
            points.append(point)
    # 转换为numpy数组
    return np.array(points)


# 读取outs.txt文件中的transformation matrix
def parse_transformation_matrix(file_path):
    transformation_matrix = []
    with open(file_path, 'r') as file:
        for line in file:
            # 检查是否是Transformation matrix的部分
            if "Transformation matrix:" in line:
                # 读取接下来的4行，构成4x4的变换矩阵
                for _ in range(4):
                    line = next(file).strip()
                    matrix_row = [float(x) for x in line.split()]
                    transformation_matrix.append(matrix_row)
    # 转换为numpy数组
    return np.array(transformation_matrix)


# 定义函数，将变换矩阵作用到N个3D点上
def apply_transformation_to_points(T, points):
    """
    将变换矩阵 T 应用于多个3D点。
    
    Args:
    T (numpy.ndarray): 4x4的变换矩阵
    points (numpy.ndarray): (N, 3) 的3D点数组
    
    Returns:
    numpy.ndarray: 变换后的(N, 3) 的3D点数组
    """
    # 检查输入的维度是否为(N, 3)
    if points.shape[1] != 3:
        raise ValueError("points数组的维度应该为(N, 3)")
    
    # 将每个点转换为齐次坐标 (N, 3) -> (N, 4)
    ones_column = np.ones((points.shape[0], 1))  # 创建N个1作为齐次坐标的最后一列
    points_homogeneous = np.hstack((points, ones_column))  # 将1追加到点的坐标后
    
    # 对每个点进行矩阵乘法 T * points_homogeneous.T
    transformed_points_homogeneous = np.dot(points_homogeneous, T.T)  # 进行矩阵乘法并转置回原维度
    
    # 转换回3D坐标 (N, 3)
    transformed_points = transformed_points_homogeneous[:, :3] / transformed_points_homogeneous[:, 3][:, np.newaxis]
    
    return transformed_points


# 读取ground_truth.txt文件中的点坐标，并解析为(N, 3)的numpy数组
def parse_ground_truth_file(file_path):
    points = []
    with open(file_path, 'r') as file:
        for line in file:
            # 按照逗号分隔数据
            data = line.strip().split(',')
            # 解析点的坐标 (X, Y, Z)
            point = [float(data[1]), float(data[2]), float(data[3])]
            points.append(point)
    # 转换为numpy数组
    return np.array(points)


def find_nearest_neighbors(points_gt, points_pred):
    """
    对于每个ground truth点，在predicted points中找到最近的点，并计算XYZ偏差。
    
    Args:
    points_gt (numpy.ndarray): (N, 3) 的ground truth点坐标数组
    points_pred (numpy.ndarray): (M, 3) 的predicted points坐标数组 (M <= N)
    
    Returns:
    numpy.ndarray: 每个预测点与其最近的真实点之间的XYZ偏差 (M, 3)
    numpy.ndarray: 每个预测点对应的ground truth点索引
    """
    # 构建 KDTree 用于最近邻搜索
    tree = KDTree(points_pred)
    
    # 对于每个预测点，找到最近的真实点
    distances, indices = tree.query(points_gt)
    
    # 计算偏差
    deviation = points_gt - points_pred[indices]
    
    # 计算平均偏差
    average_deviation = np.mean(deviation, axis=0)
    
    return deviation


T = parse_transformation_matrix("/home/zt/Project/COLMAP_GroundControlPoints-main/output/outs.txt")
points = parse_points3D_file("/home/zt/Project/COLMAP_GroundControlPoints-main/output/txt_outs/points3D.txt")
points_new = apply_transformation_to_points(T, points)
points_gt = parse_ground_truth_file("/home/zt/Project/COLMAP_GroundControlPoints-main/temp/Ground_Truth_rest.txt")

print(find_nearest_neighbors(points_gt, points_new))


