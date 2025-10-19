import cv2
import numpy as np
import copy
import matplotlib.pyplot as plt
import open3d as o3d


def classify_3d_bbox(bbox_3d_centers, deviation_thres = 1, distance_thres = 5):
    """
    根据三维框的中心去区分

    TODO:同样是可以用矩阵计算的,而不需要一个个遍历的
    """
    bbox_3d_centers = np.asarray(bbox_3d_centers).reshape(-1,3)
    mask = np.ones(bbox_3d_centers.shape[0], dtype=bool)
    
    bbox_indexs_group = list()  # 最后输出的分组情况

    while np.sum(mask) >= 2:  # 至少有两个点
        # print(np.sum(mask))
        bbox_3d_centers_temp = copy.deepcopy(bbox_3d_centers)  # 重新copy一个

        init_indexs = np.where(mask)[0]
        ############## 随机的去选择当前还没考虑的点 ############
        index = np.random.choice(init_indexs)
        bbox_3d_center = bbox_3d_centers[index]

        ############## 第一步筛选两个轴方向上对齐的点，其他点都不考虑 ###########        
        bbox_3d_centers_temp[~mask] = 1e6  # 其他的都不考虑，距离认为是无穷大
        bbox_3d_deviation = np.abs(bbox_3d_center - bbox_3d_centers_temp)

        # 只能要一条轴（选择点最多的那个轴,首先默认是z轴）
        judge = bbox_3d_deviation < deviation_thres
        judge = np.concatenate((judge[:,0].reshape(-1,1),judge[:,1].reshape(-1,1)), axis = 1)
        low_devidation = np.sum(judge, axis = 1) == 2
        max_ax_num = np.sum(low_devidation)
        # max_ax_num = 0  # 不选择z轴

        for ax in ((1,2),(0,2)):  # 再看一下是否可能出现在x轴和y轴上
            judge = bbox_3d_deviation < deviation_thres
            judge = np.concatenate((judge[:,ax[0]].reshape(-1,1),judge[:,ax[1]].reshape(-1,1)), axis = 1)
            devidation = np.sum(judge, axis = 1) == 2
            ax_num = np.sum(devidation)
            if ax_num > max_ax_num:
                low_devidation = devidation

        # low_devidation = np.sum(bbox_3d_deviation < deviation_thres, axis = 1) >= 2  # 确保两个轴对齐（就会成为和x、y、z轴对齐的边缘）

        # 注意这里是肯定会包含本身的点的，要的就是这样
        inner_3d_centers = bbox_3d_centers[low_devidation]
        inner_3d_centers_indexs = np.where(low_devidation)[0]

        ############## 第二步筛选距离太远的点 ################
        if len(inner_3d_centers) >= 4:  # 至少三个点再考虑筛选距离

            sub_mask = np.ones(inner_3d_centers.shape[0], dtype=bool)
            for sub_index, inner_3d_center in enumerate(inner_3d_centers):
                min_distance = get_min_distance(inner_3d_centers, sub_index)
                
                if min_distance >= distance_thres:  # TODO：修改这个所谓的10的阈值,至少和两个点的距离近，避免两点抱团的情况
                    sub_mask[sub_index] = False
            
            if np.sum(sub_mask) >= 4:  # 至少4个点才成组
                inner_3d_centers_indexs = inner_3d_centers_indexs[sub_mask]
                bbox_indexs_group.append(inner_3d_centers_indexs)
        
        
        mask[inner_3d_centers_indexs] = False  # 把当前已经分组的点屏蔽掉，避免再选择
        
    return bbox_indexs_group


def define_edge_range(bbox_indexs_group, bbox_3ds, ori_point_cloud, croped_point_cloud, std_devs = (0.2,0.2,0.2)):
    """
    基于输入的bbox的分类和原始点云,去明确点的边界(加权得到)
    edge_range_list[i] = (x_min, x_max, y_min, y_max, z_min, z_max, num, axis)
    """

    edge_range_list = list()
    bbox_3ds_array = np.asarray(bbox_3ds)
    
    ori_point_cloud_np = np.asarray(ori_point_cloud.points)
    croped_point_cloud_np = np.asarray(croped_point_cloud.points)
    final_mask = 0

    for bbox_indexs in bbox_indexs_group:  # 每一组的bbox序号
        bbox_info = np.zeros((len(bbox_indexs), 7))
        # 读取bbox的边界信息，存到(n,6)的array中
        bbox_info[:,:6] = bbox_3ds_array[bbox_indexs]
        bounds_num = 0

        # 读取bbox中点云的数量
        for index, bbox_index in enumerate(bbox_indexs):  # 每一个的bbox序号
            x_min, x_max, y_min, y_max, z_min, z_max = bbox_3ds[bbox_index]
            # 创建掩码数组
            mask = np.logical_and.reduce((croped_point_cloud_np[:, 0] >= x_min,  # x坐标大于等于x_min
                                          croped_point_cloud_np[:, 0] <= x_max,  # x坐标小于等于x_max
                                          croped_point_cloud_np[:, 1] >= y_min,  # y坐标大于等于y_min
                                          croped_point_cloud_np[:, 1] <= y_max,  # y坐标小于等于y_max
                                          croped_point_cloud_np[:, 2] >= z_min,  # z坐标大于等于z_min
                                          croped_point_cloud_np[:, 2] <= z_max)) # z坐标小于等于z_max
            
            final_mask = final_mask + mask  # 跟之前的掩码相加
            num = np.sum(mask)
            bounds_num += num
            bbox_info[index, 6] = np.sum(mask)  # 跟点云数量有关
        
        # 计算bound的edge
        weighted_bounds, axis = cal_weighted_bounds(bbox_info)

        # # 计算bound中应该有的点云数量
        x_min, x_max, y_min, y_max, z_min, z_max = weighted_bounds
        # mask = np.logical_and.reduce((ori_point_cloud_np[:, 0] >= x_min,  # x坐标大于等于x_min
        #                               ori_point_cloud_np[:, 0] <= x_max,  # x坐标小于等于x_max
        #                               ori_point_cloud_np[:, 1] >= y_min,  # y坐标大于等于y_min
        #                               ori_point_cloud_np[:, 1] <= y_max,  # y坐标小于等于y_max
        #                               ori_point_cloud_np[:, 2] >= z_min,  # z坐标大于等于z_min
        #                               ori_point_cloud_np[:, 2] <= z_max)) # z坐标小于等于z_max
        
        # num =  np.sum(mask)  # 跟点云数量有关

        edge_range_list.append([x_min, x_max, y_min, y_max, z_min, z_max, bounds_num, axis])
    
    regenerate_pcd = create_gaussian_point_cloud(edge_range_list, std_devs=std_devs)
    rest_crop_pcd = croped_point_cloud.select_by_index(np.where(final_mask == 0)[0])

    final_pcd = regenerate_pcd + rest_crop_pcd

    return final_pcd


def create_uniform_point_cloud(edge_range_list):
    """
    在给定的三维空间范围内生成指定数量的点云数据。
    
    参数:
        edge_range_list[i] = (x_min, x_max, y_min, y_max, z_min, z_max, num)
        
    返回:
        NumPy数组,形状为(num_points, 3),表示生成的点云数据。
    """
    for index, edge_range in enumerate(edge_range_list):
        x_min, x_max, y_min, y_max, z_min, z_max, num_points = edge_range

        # 在给定范围内随机生成x, y, z坐标
        x = np.random.uniform(x_min, x_max, num_points)
        y = np.random.uniform(y_min, y_max, num_points)
        z = np.random.uniform(z_min, z_max, num_points)
        
        # 将x, y, z坐标合并为点云数据
        point_cloud = np.column_stack((x, y, z))

        # 创建Open3D点云
        pcd = o3d.geometry.PointCloud()
        points = o3d.utility.Vector3dVector(point_cloud)
        pcd.points = points
        
        if index == 0:
            merge_point_cloud = pcd
        else:
            merge_point_cloud += pcd
    
    merge_point_cloud.remove_duplicated_points()
    
    return merge_point_cloud


def create_gaussian_point_cloud(edge_range_list, std_devs=(1/3,1/3,1/3)):
    """
    在给定的三维空间范围内生成指定数量的高斯分布点云数据。
    
    参数:
        edge_range_list[i] = (x_min, x_max, y_min, y_max, z_min, z_max, num, axis)
        std_devs (tuple): x, y, z坐标的标准差,如(std_x, std_y, std_z)。
        
    返回:
        NumPy数组,形状为(num_points, 3),表示生成的点云数据。
    """
    for index, edge_range in enumerate(edge_range_list):
        x_min, x_max, y_min, y_max, z_min, z_max, num_points, axis = edge_range
        std_x, std_y, std_z = std_devs
        
        # 生成符合高斯分布的x, y, z坐标
        if axis == 0:  # 如果是在x轴上连续分布
            x = np.random.uniform(x_min, x_max, num_points)
            y = np.random.normal(loc=(y_min+y_max)/2, scale=std_y, size=num_points)
            z = np.random.normal(loc=(z_min+z_max)/2, scale=std_z, size=num_points)
        elif axis == 1:  # 如果是在y轴上连续分布
            x = np.random.normal(loc=(x_min+x_max)/2, scale=std_x, size=num_points)
            y = np.random.uniform(y_min, y_max, num_points)
            z = np.random.normal(loc=(z_min+z_max)/2, scale=std_z, size=num_points)
        elif axis == 2:  # 如果是在z轴上连续分布
            x = np.random.normal(loc=(x_min+x_max)/2, scale=std_x, size=num_points)
            y = np.random.normal(loc=(y_min+y_max)/2, scale=std_y, size=num_points)
            z = np.random.uniform(z_min, z_max, num_points)

        # 剔除超出范围的点
        x = np.clip(x, x_min, x_max)
        y = np.clip(y, y_min, y_max)
        z = np.clip(z, z_min, z_max)
        
        # 将x, y, z坐标合并为点云数据
        point_cloud = np.column_stack((x, y, z))

        # 创建Open3D点云
        pcd = o3d.geometry.PointCloud()
        points = o3d.utility.Vector3dVector(point_cloud)
        pcd.points = points
    
        if index == 0:
            merge_point_cloud = pcd
        else:
            merge_point_cloud += pcd
    
    merge_point_cloud.remove_duplicated_points()
    
    return merge_point_cloud


def vis_2d_bbox(bbox_indexs_group, bbox_2ds, ori_depth_img_path, save_depth_img_path):
    colors = get_colors_from_cmap("jet", len(bbox_indexs_group))
    line_width = 2
    depth_color = cv2.imread(ori_depth_img_path)

    for index, bbox_indexs in enumerate(bbox_indexs_group):
        color = colors[index]
        for bbox_index in bbox_indexs:
            bbox_2d = bbox_2ds[bbox_index]
            u_min, u_max, v_min, v_max = bbox_2d
            u_min_revise = int(u_min - (u_max - u_min) / 2)
            u_max_revise = int(u_max - (u_max - u_min) / 2)
            v_min_revise = int(v_min - (v_max - v_min) / 2)
            v_max_revise = int(v_max - (v_max - v_min) / 2)
            cv2.line(depth_color, (u_min_revise, v_min_revise), (u_max_revise, v_min_revise), color, line_width)
            cv2.line(depth_color, (u_max_revise, v_min_revise), (u_max_revise, v_max_revise), color, line_width)
            cv2.line(depth_color, (u_max_revise, v_max_revise), (u_min_revise, v_max_revise), color, line_width)
            cv2.line(depth_color, (u_min_revise, v_max_revise), (u_min_revise, v_min_revise), color, line_width)

    # 保存图像
    cv2.imwrite(save_depth_img_path, depth_color)


def get_colors_from_cmap(cmap_name, n):
    # 获取指定的colormap
    cmap = plt.get_cmap(cmap_name)
    
    # 从colormap中等间隔采样n个颜色
    colors = cmap(np.linspace(0, 1, n))
    
    # 将颜色值转换为RGB格式
    rgb_colors = [list(map(lambda x: int(x*255), color)) for color in colors]
    
    return rgb_colors


def get_min_distance(inner_points_3d_center, index):
    """
    计算inner_points_3d_center[index]与inner_points_3d_center中其他点的最小距离
    
    参数:
    inner_points_3d_center: (n, 3)的NumPy数组,表示n个3D点的坐标
    index: 整数,表示要计算的点的索引
    
    返回值:
    最小距离
    """
    # 获取目标点的坐标
    target_point = inner_points_3d_center[index]
    
    # 计算目标点与其他点的距离向量
    distance_vectors = inner_points_3d_center - target_point
    
    # 去除目标点与自身的距离(距离为0)
    distance_vectors = distance_vectors[np.arange(len(distance_vectors)) != index]
    
    # 计算距离向量的欧几里得范数,得到距离
    distances = np.linalg.norm(distance_vectors, axis=1)

    # 如果distances为空,返回一个很大的数值
    if len(distances) == 0:
        return 1e6
    # 如果只有一个点,返回距离
    elif len(distances) == 1:
        return distances[0]
    else:
        # 排序并返回第四小的距离
        sorted_distances = np.sort(distances)
        return sorted_distances[2]
    

def cal_weighted_bounds(bbox_info):
    """
    计算加权边界值
    
    参数:
    box_info (numpy.ndarray): 形状为(n, 7)的数组,包含边界值和权重信息

    返回是加权的平均值
    """
    # 提取边界值和权重
    # 其中有两个轴是返回平均的值，一个轴是直接返回最小值和最大值
    bounds = bbox_info[:, :6]
    weights = bbox_info[:, 6]

    # 判断是哪个轴要返回最小值和最大值（因为在这个轴上分布）
    bounds_std = np.std(bounds[:, ::2],axis=0)
    axis = np.argmax(bounds_std)
    axis_index = [2*axis, 2*axis+1]
    
    # 计算权重总和
    weight_sum = np.sum(weights)
    
    # 计算加权边界值
    weighted_bounds = np.sum(bounds * weights[:, np.newaxis], axis=0) / weight_sum

    # 调整其中两个边界，不做加权平均，而是求最小值和最大值
    weighted_bounds[axis_index[0]] = np.min(bounds[:, axis_index[0]])
    weighted_bounds[axis_index[1]] = np.max(bounds[:, axis_index[1]])
    
    return weighted_bounds, axis


def read_bbox_info(bbox_info_path):
    """
    读取自己定义的bbox,返回二维的bbox信息和三维的bbox信息
    """
    bbox_2d = list()
    bbox_2d_center = list()
    bbox_3d = list()
    bbox_3d_center = list()
    bbox_index_list = list()
    
    with open(bbox_info_path, "r") as f:
        for item in f.readlines()[1:]:
            index, u_min, u_max, v_min, v_max, x_min, x_max, y_min, y_max, z_min, z_max = item.split(",")
            if float(x_min) == 0 and float(x_max) == 0:
                continue
            else:
                bbox_index_list.append(index)
                # bbox_2d_center.append((int(float(u_min)/2 + float(u_max)/2), int(float(v_min)/2 + float(v_max)/2)))
                ######## 注意这里因为有一点偏差，因为画图的时候用的是revise之后的，但是，实际上去bbox的是非revise的，所以事实上直接用u_min, v_min就是合理的 ################
                bbox_2d.append((float(u_min), float(u_max), float(v_min), float(v_max)))
                bbox_2d_center.append((int(float(u_min)), int(float(v_min))))
                bbox_3d.append((float(x_min), float(x_max), float(y_min), float(y_max), float(z_min), float(z_max)))
                bbox_3d_center.append((float(x_min)/2 + float(x_max)/2, float(y_min)/2 + float(y_max)/2, float(z_min)/2 + float(z_max)/2))
    
    return bbox_index_list, bbox_2d, bbox_2d_center, bbox_3d, bbox_3d_center


if __name__ == "__main__":
    ##################### 输入相关路径 ######################
    tower = "tower1"  # 自己选择是tower1还是tower4


    bbox_info_path = f"data/{tower}_bbox_info.txt"
    ori_depth_img_path = f"data/{tower}_depth.jpg"
    croped_pcd_path = f"data/{tower}_region_crop.ply"
    save_depth_img_path = f"classify_bbox_temp_by_3d_{tower}.jpg"
    ori_pcd_path = f"/D/data/zt/project/colmap/xinyang/0418/{tower}/simple_radial/dense/{tower}_region_denoise.ply"
    save_pcd_path = f"data/{tower}_region_crop_regenerate.ply"

    ##################### 后续程序运行 ######################
    # 读取已有的bbox信息
    bbox_index_list, bbox_2d, bbox_2d_center, bbox_3d, bbox_3d_center = read_bbox_info(bbox_info_path)
    print("已有的bbox信息读取完成")
    # print(bbox_center_2d)

    # 读取原始点云
    ori_pcd = o3d.io.read_point_cloud(ori_pcd_path)
    croped_pcd = o3d.io.read_point_cloud(croped_pcd_path)

    print("点云读取完成")

    # 得到二维的bbox信息
    bbox_indexs_group = classify_3d_bbox(bbox_3d_center)
    print("bbox分组完成")

    # 把二维的bbox画在图上
    vis_2d_bbox(bbox_indexs_group, bbox_2d, ori_depth_img_path, save_depth_img_path)

    # 得到每一组的整体的x_min,x_max, y_min, y_max,z_min, z_max
    regenerate_pcd = define_edge_range(bbox_indexs_group, bbox_3d, ori_pcd, croped_pcd, std_devs = (0.1,0.1,0.1))
    print("边界与点云重生成完成")

    # # 根据重生成边界，重生成点云
    # # regenerate_pcd = create_uniform_point_cloud(edge_range_list)  # 平均分布
    # regenerate_pcd = create_gaussian_point_cloud(edge_range_list, std_devs=(0.5,0.5,0.5))  # 高斯分布

    # print("点云重生成完成")

    # 保存点云
    o3d.io.write_point_cloud(save_pcd_path, regenerate_pcd)

    print("点云保存完成")

    # print(bbox_indexs_group)
