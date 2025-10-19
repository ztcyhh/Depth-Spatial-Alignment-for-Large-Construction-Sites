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


def calculate_validation_errors(points_gt_validate, points_pred_all_aligned):
    """
    通过KDTree匹配 'gt_validate' 点和 'pred_all_aligned' 点，
    并计算误差统计。
    
    Args:
       points_gt_validate (numpy.ndarray): (M, 3) 验证GCP的真实坐标 (来自 Ground_Truth_rest.txt)
       points_pred_all_aligned (numpy.ndarray): (N, 3) *所有*GCP的变换后坐标 (来自 points3D.txt + T)
    
    Returns:
       dict: 包含误差统计的字典
    """
    print(f"开始验证: {points_gt_validate.shape[0]} 个真实点 (GT_Validate)")
    print(f"匹配目标: {points_pred_all_aligned.shape[0]} 个变换后点 (RECON_Aligned)")

    # 1. 在所有变换后的点上构建KDTree (您的正确逻辑)
    tree = KDTree(points_pred_all_aligned)
    
    # 2. 为每个 "真实" 验证点 找到 "变换后" 点中的最近邻
    #    我们假设这个最近邻就是其对应的点
    distances, indices = tree.query(points_gt_validate)
    
    # 3. 提取匹配的预测点
    matched_pred_points = points_pred_all_aligned[indices]
    
    # 4. 计算偏差 (Error = True_Coord - Aligned_Pred_Coord)
    errors = points_gt_validate - matched_pred_points
    
    # --- 5. 计算关键统计指标 ---
    
    # # MAE (Mean Absolute Error)
    # mae_per_axis = np.mean(np.abs(errors), axis=0)
    # # 距离的平均值是总的MAEv
    # total_mae = np.mean(distances) 
    
    # # RMSE (Root Mean Square Error)
    # rmse_per_axis = np.sqrt(np.mean(errors**2, axis=0))
    # # 距离的均方根是总的RMSE
    # total_rmse = np.sqrt(np.mean(distances**2))
    
    # # Mean Error (用于检查系统偏差)
    # mean_error = np.mean(errors, axis=0)

    mean_distance = np.mean(distances)
    
    # results = {
    #     'count': errors.shape[0],
    #     'mean_error': mean_error,
    #     'mae_per_axis': mae_per_axis,
    #     'total_mae': total_mae,
    #     'rmse_per_axis': rmse_per_axis,
    #     'total_rmse': total_rmse,
    #     'raw_errors': errors,      # (可选)
    #     'raw_distances': distances, # (可选)
    #     'mean_distance': mean_distance # (可选)
    # }
    return mean_distance

def calculate_std_dispersion(points):
    """
    方法二：计算点云到其3D质心的均方根(RMS)距离。
    (这是 "整体标准差" 的正确方法)
    """
    if points.shape[0] < 2:
        return 0.0
    
    # 我们可以通过捷径计算
    # 1. 分别计算三个轴的标准差
    #    ddof=0 表示使用总体标准差 (N)，而不是样本标准差 (N-1)
    #    这与 np.mean 的行为一致。
    std_devs = np.std(points, axis=0, ddof=0)
    
    # 2. std_devs 是 [sigma_x, sigma_y, sigma_z]
    #    我们要计算 sqrt(sigma_x^2 + sigma_y^2 + sigma_z^2)
    #    这等同于计算这个向量的L2范数(欧几里得长度)
    
    rms_distance = np.linalg.norm(std_devs)
    
    return std_devs,rms_distance


T = parse_transformation_matrix("/home/zt/Project/COLMAP_GroundControlPoints-main/output/outs.txt")
points = parse_points3D_file("/home/zt/Project/COLMAP_GroundControlPoints-main/output/txt_outs/points3D.txt")
points_new = apply_transformation_to_points(T, points)
points_gt = parse_ground_truth_file("/home/zt/Project/COLMAP_GroundControlPoints-main/temp/Ground_Truth_rest.txt")
points_gt_other = parse_ground_truth_file("/home/zt/Project/COLMAP_GroundControlPoints-main/temp/Ground_Truth_random.txt")
# 计算points_gt_other的数量
points_gt_other_count = points_gt_other.shape[0]
mean_distance = calculate_validation_errors(points_gt, points_new)
std_devs,rms_distance = calculate_std_dispersion(points_gt_other)

with open("/home/zt/Project/COLMAP_GroundControlPoints-main/temp/gcp_error.txt", "a") as f:
    f.write(f"{points_gt_other_count},{std_devs[0]},{std_devs[1]},{std_devs[2]},{rms_distance},{mean_distance}\n")


