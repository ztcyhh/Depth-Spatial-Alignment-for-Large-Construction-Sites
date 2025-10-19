"""
从反投影的深度图中得到点云边缘
因为我们希望去校正点云边缘，所以还需要把边界框也给存下来，这个事实上也代表了深度
"""
import open3d as o3d
import cv2
import numpy as np
from pixel_coord2world_coord import get_K_Rt, get_depth, pixel2world, pixel2world_known_depth
import time
import math
import random
from scipy.spatial import Delaunay
import copy 


# 滑动窗口循环遍历整张图
def iter_image_bbox(depth_map, pcd, cam_path, ori_depth_vis_img_path, img_save_path, depth_thres, required_real_length=1):
    """
    用滑动窗口来遍历整张图
    """
    depth_confirm_map = np.zeros_like(depth_map)  # 确认depth_map是否都被遍历
    depth_confirm_map[np.isnan(depth_map)] = 1.0  # 首先将nan的部分都填上，因为这部分我们不遍历（这个避免了很多问题）
    depth_confirm_map[:2, :] = 1.0  # 边界都不考虑
    depth_confirm_map[-2:, :] = 1.0  # 边界都不考虑
    depth_confirm_map[:, :2] = 1.0  # 边界都不考虑
    depth_confirm_map[:, -2:] = 1.0  # 边界都不考虑

    left_top_point = find_top_left_zero(depth_confirm_map)

    iter_time = 0

    bbox_info = list()

    while left_top_point:
        left_top_point = find_top_left_zero(depth_confirm_map)

        # 在最后一步的时候提前跳出，避免报错
        if not left_top_point:
            break
        
        IS_EDGE_BBOX, u_pixel, v_pixel, represent_depth = define_edge_bbox(depth_map, cam_path, ori_depth_vis_img_path, img_save_path, left_top_point, depth_thres, required_real_length)
        
        depth_confirm_map[left_top_point[1]:(left_top_point[1] + v_pixel), left_top_point[0]:(left_top_point[0] + u_pixel)] = 1.0  # 已经遍历的部分填充
        
        if IS_EDGE_BBOX:
            iter_time += 1
            # time1 = time.time()
            bbox_3d = generate_3d_bbox(represent_depth, cam_path, left_top_point, u_pixel, v_pixel, required_real_length)
            crop_pcd = crop_point_from_3d_bbox(pcd, bbox_3d)
            # time2 = time.time()
            # print(time2 - time1)

            if iter_time == 1:
                # 如果已经循环了一次，那么就打开原来的图，然后往上继续画框
                ori_depth_vis_img_path = img_save_path
                crop_pcd_merge = crop_pcd
            
            # elif iter_time == 5:  # 快速的测试跳出
            #     break
            
            else:
                crop_pcd_merge = crop_pcd_merge + crop_pcd  # 和之前的滑动窗口中的点云合并到一起
                # crop_pcd_merge.remove_duplicated_points()

            bbox_info_item = dict()
            crop_bbox_3d = crop_pcd.get_axis_aligned_bounding_box()
            min_bound = crop_bbox_3d.get_min_bound()
            max_bound = crop_bbox_3d.get_max_bound()

            bbox_info_item["u_min"] = left_top_point[0]
            bbox_info_item["v_min"] = left_top_point[1]
            bbox_info_item["u_max"] = left_top_point[0] + u_pixel
            bbox_info_item["v_max"] = left_top_point[1] + v_pixel

            bbox_info_item["x_min"] = min_bound[0]
            bbox_info_item["y_min"] = min_bound[1]
            bbox_info_item["z_min"] = min_bound[2]
            bbox_info_item["x_max"] = max_bound[0]
            bbox_info_item["y_max"] = max_bound[1]
            bbox_info_item["z_max"] = max_bound[2]

            bbox_info.append(bbox_info_item)

    return crop_pcd_merge, bbox_info
        
# 判断框内的深度差,从而确定是否为边缘部分
def define_edge_bbox(depth_map, cam_path, ori_depth_vis_img_path, img_save_path, left_top_point, depth_thres, required_real_length=1):
    """
    判断滑动窗口内是否存在边缘点
    返回的是True or False,该窗口是不是边缘窗口
    """
    percentile = 10  # 表示选择滑动窗口内的代表性深度（前n%的深度）

    # 选取滑动窗口内的部分
    u_pixel, v_pixel, u_temp_vis, v_temp_vis = cal_edge_bbox_length(depth_map, cam_path, left_top_point, required_real_length)
    depth_region = depth_map[left_top_point[1]:(left_top_point[1] + v_pixel), left_top_point[0]:(left_top_point[0] + u_pixel)]
    depth_region = depth_region[~np.isnan(depth_region)]  # 排除NAN值的部分

    # 将矩阵展平为一维数组
    flat_matrix = depth_region.ravel()

    # 计算分位数
    lower_percentile = percentile / 100
    upper_percentile = 1 - lower_percentile

    # 使用quantile函数计算临界值
    lower_depth = np.quantile(flat_matrix, lower_percentile)
    upper_depth = np.quantile(flat_matrix, upper_percentile)

    represent_depth = lower_depth  # 认为突起的部分是我们的目标，所以将lower_depth作为代表性深度

    # print(upper_depth, lower_depth)

    if upper_depth - lower_depth > depth_thres:
        u_pixel = max(u_pixel, u_temp_vis)
        v_pixel = max(v_pixel, v_temp_vis)
        left_top_point_revise = (max(0, int(left_top_point[0] - u_pixel/2)), max(0, int(left_top_point[1] - v_pixel/2)))
        vis_edge_bbox(ori_depth_vis_img_path, img_save_path, left_top_point_revise, u_pixel, v_pixel)  # 可视化bbox
        return (True, u_pixel, v_pixel, represent_depth)
    else:
        return (False, u_pixel, v_pixel, represent_depth)


def generate_3d_bbox(represent_depth, cam_path, left_top_point, u_pixel, v_pixel, required_real_length):
    """
    根据二维框和depth_map创建三维视锥(8*3)，由八个点确定的视锥边界
    """
    # 计算边界的u、v坐标
    left_top_u, left_top_v = left_top_point
    right_top_u = left_top_u + u_pixel
    right_top_v = left_top_v
    left_bottom_u = left_top_u
    left_bottom_v = left_top_v + v_pixel
    right_bottom_u = right_top_u
    right_bottom_v = left_bottom_v

    # 转换成真实世界点
    close_depth = represent_depth
    far_depth = represent_depth + 1

    real_left_top_close = pixel2world_known_depth(cam_path, (left_top_u, left_top_v), close_depth)
    real_right_top_close = pixel2world_known_depth(cam_path, (right_top_u, right_top_v), close_depth)
    real_right_bottom_close = pixel2world_known_depth(cam_path, (right_bottom_u, right_bottom_v), close_depth)
    real_left_bottom_close = pixel2world_known_depth(cam_path, (left_bottom_u, left_bottom_v), close_depth)
    real_left_top_far = pixel2world_known_depth(cam_path, (left_top_u, left_top_v), far_depth)
    real_right_top_far = pixel2world_known_depth(cam_path, (right_top_u, right_top_v), far_depth)
    real_right_bottom_far = pixel2world_known_depth(cam_path, (right_bottom_u, right_bottom_v), far_depth)
    real_left_bottom_far = pixel2world_known_depth(cam_path, (left_bottom_u, left_bottom_v), far_depth)

    bbox_3d = np.concatenate((real_left_top_close, real_right_top_close, real_right_bottom_close, real_left_bottom_close, real_left_top_far, real_right_top_far, real_right_bottom_far, real_left_bottom_far), axis = 1)
    bbox_3d = np.transpose(bbox_3d)

    return bbox_3d


def crop_point_from_3d_bbox(pcd, bbox_vertices):
    """
    基于八个角点坐标的六面体(8*3)去裁剪点云
    """

    #########################################
    #### 1. 首先用axis_aligned框去初选点云 ####
    #########################################

    # 计算凸多边形的轴对齐包围盒
    box = o3d.geometry.PointCloud()
    box.points = o3d.utility.Vector3dVector(bbox_vertices)
    aabb = box.get_axis_aligned_bounding_box()
    box_points = np.asarray(aabb.get_box_points())
    x_min, x_max = box_points[:, 0].min(), box_points[:, 0].max()
    y_min, y_max = box_points[:, 1].min(), box_points[:, 1].max()
    z_min, z_max = box_points[:, 2].min(), box_points[:, 2].max()

    vol = o3d.visualization.read_selection_polygon_volume("xinyang_region.json")  # 读取cropjson模板
    vol.axis_max = z_max
    vol.axis_min = z_min
    vol.bounding_polygon = o3d.utility.Vector3dVector([[x_max, y_max, 0],
                                                       [x_max, y_min, 0],
                                                       [x_min, y_min, 0],
                                                       [x_min, y_max, 0]])
    
    init_proc_pcd = vol.crop_point_cloud(pcd)

    #########################################
    ########## 2. 创建凸包，遍历点云 ##########
    #########################################
    def in_hull(p, hull):
        """
        Test if points in `p` are in `hull`
        `p` should be a `NxK` coordinates of `N` points in `K` dimensions
        `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the
        coordinates of `M` points in `K`dimensions for which Delaunay triangulation
        will be computed
        """

        if not isinstance(hull,Delaunay):
            hull = Delaunay(hull)

        return hull.find_simplex(p)>=0
    
    init_proc_pcd_points = np.asarray(init_proc_pcd.points)
    mask = in_hull(init_proc_pcd_points, bbox_vertices)
    indices = np.where(mask)[0]
    final_proc_pcd = init_proc_pcd.select_by_index(indices)
    
    return final_proc_pcd

    # return init_proc_pcd
    


# 根据深度计算框的大小，使得实际大小接近1m
def cal_edge_bbox_length(depth_map, cam_path, left_top_point, required_real_length = 1):
    """
    计算框大小
    """
    w, h = np.shape(depth_map)[1], np.shape(depth_map)[0]

    # 初始框大小选择
    K, _, _ = get_K_Rt(cam_path)
    fx = K[0,0]  # 焦距
    depth = get_depth(left_top_point, depth_map)
    simple_length = depth / fx
    u_temp = math.ceil(required_real_length / simple_length)
    v_temp = math.ceil(required_real_length / simple_length)

    # 避免超出边界
    u_temp = min(u_temp, w - left_top_point[0] - 1)
    v_temp = min(v_temp, h - left_top_point[1] - 1)

    u_temp_vis = copy.deepcopy(u_temp)
    v_temp_vis = copy.deepcopy(v_temp)

    # 循环3次,更精确的去计算用多大的框
    for iter in range(1):
        
        real_point1 = pixel2world(cam_path, left_top_point, depth_map)
        real_point2 = pixel2world(cam_path, (left_top_point[0] + u_temp, left_top_point[1]), depth_map)
        real_point3 = pixel2world(cam_path, (left_top_point[0] , left_top_point[1] + v_temp), depth_map)
        
        ########################################
        ############### trick1.1 #################
        ########################################
        # 可能出现深度值是空的情况
        if np.isnan(real_point2).any() and np.isnan(real_point3).any():  
            complex_length_u = simple_length
            complex_length_v = simple_length
        
        elif np.isnan(real_point2).any():  
            complex_length_u = simple_length
            complex_length_v = cal_distance(real_point1, real_point3) / v_temp
        
        elif np.isnan(real_point3).any():
            complex_length_u = cal_distance(real_point1, real_point2) / u_temp
            complex_length_v = simple_length

        else:
            complex_length_u = cal_distance(real_point1, real_point2) / u_temp # x方向的单像素实际长度 
            complex_length_v = cal_distance(real_point1, real_point3) / v_temp # y方向的单像素实际长度
        
        ########################################
        ########### trick 2 ####################
        ########################################
        # 如果计算出的实际距离非常大，那么就直接设置成required_real_length（也就是说框变成5个像素*5个像素, 同时直接跳出
        if complex_length_u > required_real_length * 5:
            complex_length_u = copy.deepcopy(complex_length_v)
        
        elif complex_length_v > required_real_length * 5:
            complex_length_v = copy.deepcopy(complex_length_u)
        
        elif complex_length_u > required_real_length * 5 and complex_length_v > required_real_length * 5:
            complex_length_u, complex_length_v = required_real_length / 10, required_real_length / 10
            u_pixel = math.ceil(required_real_length / complex_length_u)
            v_pixel = math.ceil(required_real_length / complex_length_v)

            # 避免超出边界
            u_pixel = min(u_pixel, w - left_top_point[0] - 1)
            v_pixel = min(v_pixel, h - left_top_point[1] - 1)
            break
        
        u_pixel = math.ceil(required_real_length / complex_length_u)
        v_pixel = math.ceil(required_real_length / complex_length_v)

        # 避免超出边界
        u_pixel = min(u_pixel, w - left_top_point[0] - 1)
        v_pixel = min(v_pixel, h - left_top_point[1] - 1)

        u_temp, v_temp = u_pixel, v_pixel

    return u_pixel, v_pixel, u_temp_vis, v_temp_vis


# 可视化edge_bbox
def vis_edge_bbox(ori_depth_vis_img_path, img_save_path, left_top_point, u_pixel, v_pixel):
    """
    将可视化图片上画上边缘部分的边界框
    """
    depth_color = cv2.imread(ori_depth_vis_img_path)

    #### 在上面画白色框
    left_top_u, left_top_v = left_top_point
    right_top_u = left_top_u + u_pixel
    right_top_v = left_top_v
    left_bottom_u = left_top_u
    left_bottom_v = left_top_v + v_pixel
    right_bottom_u = right_top_u
    right_bottom_v = left_bottom_v

    # 使用OpenCV绘制矩形框
    color = (255, 255, 255)  # 白色
    # color = (0, 0, 0)  # 黑色
    line_width = 2
    cv2.line(depth_color, (left_top_u, left_top_v), (right_top_u, right_top_v), color, line_width)
    cv2.line(depth_color, (right_top_u, right_top_v), (right_bottom_u, right_bottom_v), color, line_width)
    cv2.line(depth_color, (right_bottom_u, right_bottom_v), (left_bottom_u, left_bottom_v), color, line_width)
    cv2.line(depth_color, (left_bottom_u, left_bottom_v), (left_top_u, left_top_v), color, line_width)

    # 保存图像
    cv2.imwrite(img_save_path, depth_color)
    
    
    # print(f"已将彩色深度图保存到 {img_save_path}")


def cal_distance(point1, point2):
    """
    计算两个(3, 1)形状的列向量之间的欧几里德距离

    参数:
    point1 (numpy.ndarray): 第一个点的坐标,形状为(3, 1)
    point2 (numpy.ndarray): 第二个点的坐标,形状为(3, 1)

    返回:
    distance (float): 两点之间的欧几里德距离
    """
    # 检查输入是否为(3, 1)形状的列向量
    if point1.shape != (3, 1) or point2.shape != (3, 1):
        raise ValueError("输入必须是(3, 1)形状的列向量")

    # 计算两点之间的向量差
    vec = point1 - point2

    # 计算欧几里德距离
    distance = np.linalg.norm(vec)

    return distance


def load_depth_map(depth_path):
    return np.load(depth_path)


def find_top_left_zero(arr):
    """
    在给定的二值(0和1)NumPy数组中找到最左上角的0元素

    参数:
    arr (numpy.ndarray): 输入的二值数组

    返回:
    top_left_zero (tuple): 最左上角的0元素的坐标,形式为(row, col)
    """
    # 找到所有0元素的坐标
    zero_indices = np.argwhere(arr == 0)

    # 如果没有0元素,返回None
    if zero_indices.size == 0:
        return None

    # 根据行和列索引排序
    zero_indices = zero_indices[np.lexsort((zero_indices[:, 1], zero_indices[:, 0]))]   # 从上往下找

    # 返回最左上角的0元素坐标
    top_left_zero = (zero_indices[0][1], zero_indices[0][0])

    return top_left_zero


    

if __name__ == "__main__":
    # 需要输入
    pcd_path = "/D/data/zt/project/colmap/xinyang/0418/tower4/simple_radial/dense/tower4_region_denoise.ply"
    pcd_save_path = "/D/data/zt/project/colmap/xinyang/0418/tower4/simple_radial/dense/tower4_region_crop.ply"
    depth_thres = 4
    required_real_length = 2


    # 默认
    bbox_info_path = "bbox_info.txt"
    depth_path = "/home/zt/Project/PCD_Registration/edge_wise_registration/temp_depth.npy"
    cam_path = "/home/zt/Project/PCD_Registration/edge_wise_registration/temp.txt"
    ori_depth_vis_img_path = "/home/zt/Project/PCD_Registration/edge_wise_registration/temp_depth.jpg"
    img_save_path = "temp_temp.png"
    # left_top_point = (1188, 927)  # 
    # left_top_point = (915, 1019)


    depth_map = load_depth_map(depth_path)
    pcd = o3d.io.read_point_cloud(pcd_path)
    # # cal_edge_bbox_length(depth_map, cam_path, left_top_point)
    # edge_status = define_edge_bbox(depth_map, cam_path, img_save_path, left_top_point, depth_thres, required_real_length)
    # print(edge_status)
    crop_pcd, bbox_info = iter_image_bbox(depth_map, pcd, cam_path, ori_depth_vis_img_path, img_save_path, depth_thres, required_real_length)

    crop_pcd.remove_duplicated_points()

    o3d.io.write_point_cloud(pcd_save_path, crop_pcd)

    with open(bbox_info_path,"w") as f:
        f.write("id, u_min, u_max, v_min, v_max, x_min, x_max, y_min, y_max, z_min, z_max\n")
        for index, bbox_info_item in enumerate(bbox_info):
            f.write(f"{index+1},"
                    f"{bbox_info_item['u_min']}, {bbox_info_item['u_max']}, "
                    f"{bbox_info_item['v_min']}, {bbox_info_item['v_max']}, "
                    f"{bbox_info_item['x_min']}, {bbox_info_item['x_max']}, "
                    f"{bbox_info_item['y_min']}, {bbox_info_item['y_max']}, "
                    f"{bbox_info_item['z_min']}, {bbox_info_item['z_max']}\n")
    
    print("over")
