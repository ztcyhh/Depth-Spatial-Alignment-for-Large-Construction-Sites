from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
# from sklearn.linear_model import TheilSenRegressor
# from scipy.linalg import lstsq
from scipy.interpolate import interp1d
import os
import math

def calculate_line_equation(x1, y1, x2, y2):
    """
    计算直线的斜率和截距。如果x坐标相同，返回垂直线的特殊情况。
    
    参数:
    - x1, y1: 第一个点的坐标
    - x2, y2: 第二个点的坐标
    
    返回:
    - slope: 斜率（如果为垂直线，返回None）
    - intercept: 截距（如果为垂直线，返回x坐标）
    """
    if x1 == x2:
        # 垂直线，斜率为None，截距为x坐标
        return None, x1
    else:
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        return slope, intercept

def calculate_distance_between_parallel_lines(slope, intercept1, intercept2):
    """
    计算两条平行直线之间的距离。
    
    参数:
    - slope: 斜率
    - intercept1: 第一条直线的截距
    - intercept2: 第二条直线的截距
    
    返回:
    - distance: 两条平行直线之间的距离
    """
    if slope is None:
        # 垂直线的情况，直接返回x坐标差的绝对值
        return abs(intercept1 - intercept2)
    else:
        return abs(intercept1 - intercept2) / np.sqrt(slope**2 + 1)
    

def calculate_distances(contour):
    """
    计算封闭轮廓上每对相邻点之间的距离。
    
    参数:
    - contour: 输入的封闭轮廓点列表，格式为 [[x1, y1], [x2, y2], ...]
    
    返回:
    - distances: 每对相邻点之间的距离列表
    """
    contour = np.array(contour)
    distances = np.sqrt(np.sum(np.diff(contour, axis=0)**2, axis=1))
    # 添加最后一个点到第一个点的距离，确保轮廓封闭
    distances = np.append(distances, np.linalg.norm(contour[-1] - contour[0]))
    return distances

def densify_contour_uniformly(contour, num_points=100):
    """
    将封闭轮廓点均匀插值为更多的点。
    
    参数:
    - contour: 输入的封闭轮廓点列表，格式为 [[x1, y1], [x2, y2], ...]
    - num_points: 输出的插值点数量
    
    返回:
    - densified_contour: 插值后的均匀分布的轮廓点列表，格式为 [[x1, y1], [x2, y2], ...]
    """
    contour = np.array(contour)
    
    # 计算相邻点之间的距离
    distances = calculate_distances(contour)
    
    # 计算轮廓的总长度
    total_length = np.sum(distances)
    
    # 在总长度上均匀生成插值点的位置
    segment_lengths = np.linspace(0, total_length, num_points)
    
    # 初始化插值后的轮廓点
    densified_contour = []
    
    # 累积长度
    cumulative_lengths = np.cumsum(distances)
    
    # 遍历每个新插值长度
    for target_length in segment_lengths:
        # 找到累积长度中第一个大于等于目标长度的索引
        idx = np.searchsorted(cumulative_lengths, target_length)
        
        # 确保 idx 不超过轮廓点的数量（闭环处理）
        idx = min(idx, len(cumulative_lengths) - 1)
        
        # 计算插值比例
        if idx == 0:
            t = target_length / distances[idx]
            new_point = (1 - t) * contour[0] + t * contour[1]
        else:
            prev_length = cumulative_lengths[idx - 1]
            t = (target_length - prev_length) / distances[idx]
            p1 = contour[idx % len(contour)]  # 使用 % 确保轮廓封闭
            p2 = contour[(idx + 1) % len(contour)]  # 环绕到第一个点
            new_point = (1 - t) * p1 + t * p2
        
        densified_contour.append(new_point)
    
    return np.array(densified_contour)

if  __name__ == "__main__":
    model_path = Path("/data/zt/project/object_seg/xinyang_GCP/0418_604/RESULT/ori_yolov8x_imgsz2560_train_batch_2/weights/best.pt")
    image_path = Path("/data/zt/project/object_seg/xinyang_GCP/0418_604/YOLODataset/images/val")
    label_path = Path("/data/zt/project/object_seg/xinyang_GCP/0418_604/YOLODataset/labels/val")
    names = ["no4", "no6", "no9", "no3", "no7", "no2", "no1"]

    distance_error_list = list()

    conf_thres = 0.5
    image_size = 2560

    # 画图相关参数
    alpha = 0.4
    mask_color_list = ((139, 0, 0),(0, 0, 139),(0, 139, 0),(155,201,239))   # 深蓝 / 深红 / 深绿 / 浅黄
    dark_color_list = ((255, 0, 0),(0, 0, 255),(0, 255, 0),(73,157,225))   # 深蓝 / 深红 / 深绿 / 深黄
    # mask_color_list = ((0, 0, 139),(139, 0, 0),(0, 139, 0),(75,0,130))   # 深蓝 / 深红 / 深绿 / 深紫
    # mask_color = (139, 0, 0)  # 深蓝色

    


    # Load a model
    model = YOLO(model_path)

    # Run inference
    results = model.predict(image_path, imgsz=image_size, conf=conf_thres)

    # 如果有n张图片
    for r in results:
        img = np.copy(r.orig_img)
        img_name = Path(r.path).stem # base name
        image_height, image_width = img.shape[:2]

        ######################################
        ########### 读取GT的内容 ##############
        ######################################
        label_file_path = label_path / (img_name + ".txt")
        labels = []
        with open(label_file_path, 'r') as file:
            for line in file:
                # 将每一行的字符串转换为浮点数数组
                label_data = list(map(float, line.strip().split()))
                labels.append(label_data)

        contours_dict = {}  # 轮廓点的dict
        center_dict = {}  # 中心点的dict
    
        for label in labels:
            class_id = int(label[0])  # 获取类别ID
            class_name = names[class_id]  # 类别名
            points = label[1:]  # 获取点坐标部分
            
            # 将归一化的坐标转换为实际图像上的像素坐标
            contour = []
            for i in range(0, len(points), 2):
                x = int(points[i] * image_width)  # x坐标
                y = int(points[i+1] * image_height)  # y坐标
                contour.append([x, y])
            
            # 写入轮廓
            contour = np.array(contour, dtype=np.int32)
            contours_dict[class_name] = [contour]


            # 计算轮廓的重心
            M = cv2.moments(contour)
            if M['m00'] != 0:
                cX = int(M['m10'] / M['m00'])  # 重心X坐标
                cY = int(M['m01'] / M['m00'])  # 重心Y坐标
            else:
                cX, cY = 0, 0  # 如果面积为0，则重心设为(0, 0)
            
            center_dict[class_name] = [(cX, cY)]

        for ci,c in enumerate(r):  # 有n个目标
            #  Get detection class name
            conf = c.boxes.conf.tolist().pop()  # bbox confidence
            label = c.names[c.boxes.cls.tolist().pop()]  # class name

            # 检查置信度是否满足阈值
            if conf < conf_thres:
                continue


            ################ segment相关 ################
            contour = c.masks.xy.pop()  # 输出的是mask的边缘轮廓

            # 将轮廓坐标转换为整数
            contour = np.array(contour, dtype=np.int32)

            if label not in contours_dict.keys():
                contours_dict[label] = [contour]
            else:
                contours_dict[label].append(contour)
            

            # 计算轮廓的重心
            M = cv2.moments(contour)
            if M['m00'] != 0:
                cX = int(M['m10'] / M['m00'])  # 重心X坐标
                cY = int(M['m01'] / M['m00'])  # 重心Y坐标
            else:
                cX, cY = 0, 0  # 如果面积为0，则重心设为(0, 0)
            
            if label not in center_dict.keys():
                center_dict[label] = [(cX, cY)]
            else:
                center_dict[label].append((cX, cY))

        
        image_gt = img.copy()
        image_pred = img.copy()

        # 开始绘画,先画掩码

        ######### 画gt图 #########
        # 创建透明图层
        overlay_gt = img.copy()

        ######### 画pred图 #########
        # 创建透明图层
        overlay_pred = img.copy()

        for ci, label in enumerate(contours_dict.keys()):
            # 读取内容
            if len(contours_dict[label]) == 1:
                contour_gt = contours_dict[label][0]
                contour_pred = contours_dict[label][0]

            else:
                contour_gt = contours_dict[label][0]
                contour_pred = contours_dict[label][1]
                       

            # 在透明图层上绘制多边形（掩膜）
            cv2.fillPoly(overlay_gt, [contour_gt], color=mask_color_list[ci % len(mask_color_list)])

            # 在透明图层上绘制多边形（掩膜）
            cv2.fillPoly(overlay_pred, [contour_pred], color=mask_color_list[ci % len(mask_color_list)])
            
        # 将透明图层与原始图像进行融合
        image_gt = cv2.addWeighted(overlay_gt, alpha, image_gt, 1 - alpha, 0)
        # 将透明图层与原始图像进行融合
        image_pred = cv2.addWeighted(overlay_pred, alpha, image_pred, 1 - alpha, 0)
        
        # 再画中心点
        for ci, label in enumerate(center_dict.keys()):
            # 读取内容
            if len(contours_dict[label]) == 1:
                center_gt = center_dict[label][0]
                center_pred = center_dict[label][0]
            else:
                center_gt = center_dict[label][0]
                center_pred = center_dict[label][1]
            
                distance = math.sqrt((center_pred[0] - center_gt[0])**2 + (center_pred[1] - center_gt[1])**2)
                distance_error_list.append(distance)


            # 绘制重心点
            cv2.circle(image_gt, center_gt, 2, mask_color_list[ci % len(dark_color_list)], -1)

            # 绘制重心点
            cv2.circle(image_pred, center_pred, 2, mask_color_list[ci % len(dark_color_list)], -1)


        cv2.imwrite(os.path.join("/home/zt/Project/Detect_Segment/ultralytics/result_2560_mask_center", img_name+"_gt.png"), image_gt)
        cv2.imwrite(os.path.join("/home/zt/Project/Detect_Segment/ultralytics/result_2560_mask_center", img_name+"_pred.png"), image_pred)
    
    print(f"error平均值是{sum(distance_error_list)/len(distance_error_list):.3f}, 共有{len(distance_error_list)}个GCP")
        