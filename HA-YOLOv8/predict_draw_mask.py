from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
# from sklearn.linear_model import TheilSenRegressor
# from scipy.linalg import lstsq
from scipy.interpolate import interp1d
import os

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
    model_path = Path("/data/zt/project/object_seg/xinyang_GCP/0418_604/RESULT/ori_yolov8x_imgsz1280_train_batch_4/weights/best.pt")
    image_path = Path("/data/zt/project/object_seg/xinyang_GCP/0418_604/YOLODataset/images/val/00000131.png")

    conf_thres = 0.5
    image_size = 1280

    # 画图相关参数
    alpha = 0.4
    mask_color_list = ((139, 0, 0),(0, 0, 139),(0, 139, 0),(75,0,130))   # 深蓝 / 深红 / 深绿 / 深紫
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

        for ci,c in enumerate(r):  # 有n个目标
            #  Get detection class name
            conf = c.boxes.conf.tolist().pop()  # bbox confidence
            label = c.names[c.boxes.cls.tolist().pop()]  # class name

            # 检查置信度是否满足阈值
            if conf < conf_thres:
                continue
            
            ################ bbox相关 ################
            # BBOX的中点
            xywh = c.boxes.xywh.tolist().pop()
            bbox_center = (xywh[0], xywh[1])
            bbox_center_int = (round(xywh[0]), round(xywh[1]))

            # BBOX的四个角点
            point_left_top = (int(xywh[0] - xywh[2]/2), int(xywh[1] - xywh[3]/2))
            point_left_down = (int(xywh[0] - xywh[2]/2), int(xywh[1] + xywh[3]/2))
            point_right_top = (int(xywh[0] + xywh[2]/2), int(xywh[1] - xywh[3]/2))
            point_right_down = (int(xywh[0] + xywh[2]/2), int(xywh[1] + xywh[3]/2))

            # cv2.line(img, point_left_top, point_left_down, mask_color_list[ci % len(mask_color_list)], 2)
            # cv2.line(img, point_left_top, point_right_top, mask_color_list[ci % len(mask_color_list)], 2)
            # cv2.line(img, point_right_top, point_right_down, mask_color_list[ci % len(mask_color_list)], 2)
            # cv2.line(img, point_left_down, point_right_down, mask_color_list[ci % len(mask_color_list)], 2)

            txt_size = np.min(np.shape(img)[:2]) * 0.0005

            cv2.putText(img, label, (point_left_top[0], int(point_left_top[1] - 8*txt_size)), cv2.FONT_HERSHEY_SIMPLEX, txt_size, mask_color_list[ci % len(mask_color_list)], 2)


            ################ segment相关 ################
            contour = c.masks.xy.pop()  # 输出的是mask的边缘轮廓

            # 将轮廓坐标转换为整数
            contour = np.array(contour, dtype=np.int32)

            # 创建透明图层
            overlay = img.copy()

            # 在透明图层上绘制多边形（掩膜）
            cv2.fillPoly(overlay, [contour], color=mask_color_list[ci % len(mask_color_list)])

            # 将透明图层与原始图像进行融合
            cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

            # 1. 找到Oriented Bounding Box
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.array(box)
            box_int = np.array(box, dtype=np.int32)
            # cv2.drawContours(img, [box_int], 0, (139, 0, 0), 1)  # 绘制Oriented Bounding Box

            # 2. 获取Oriented Bounding Box的四个顶点
            top_left = box[0]
            top_right = box[1]
            bottom_right = box[2]
            bottom_left = box[3]

            # 3. 获取上边和下边的直线方程
            top_slope, top_intercept = calculate_line_equation(top_left[0], top_left[1], top_right[0], top_right[1])
            bottom_slope, bottom_intercept = calculate_line_equation(bottom_right[0], bottom_right[1], bottom_left[0], bottom_left[1])

            # 计算上下边的距离
            dist_top_bottom = calculate_distance_between_parallel_lines(top_slope, top_intercept, bottom_intercept)

            # 4. 获取左边和右边的直线方程
            left_slope, left_intercept = calculate_line_equation(top_left[0], top_left[1], bottom_left[0], bottom_left[1])
            right_slope, right_intercept = calculate_line_equation(top_right[0], top_right[1], bottom_right[0], bottom_right[1])

            # 计算左右边的距离
            dist_left_right = calculate_distance_between_parallel_lines(left_slope, left_intercept, right_intercept)

            # 分割轮廓为上下左右四部分
            top_contour = []
            bottom_contour = []
            left_contour = []
            right_contour = []

            contour = densify_contour_uniformly(contour, num_points=100)
            print(f"contour的长度是,{len(contour)}")

            for point in contour:
                x, y = point[0], point[1]

                # 计算点到上下边的距离
                if top_slope is not None:
                    dist_top = abs(y - (top_slope * x + top_intercept)) / np.sqrt(top_slope**2 + 1)
                    dist_bottom = abs(y - (bottom_slope * x + bottom_intercept)) / np.sqrt(bottom_slope**2 + 1)
                else:  # 垂直线的情况
                    dist_top = abs(x - top_intercept)
                    dist_bottom = abs(x - bottom_intercept)

                # 计算点到左右边的距离
                if left_slope is not None:
                    dist_left = abs(y - (left_slope * x + left_intercept)) / np.sqrt(left_slope**2 + 1)
                    dist_right = abs(y - (right_slope * x + right_intercept)) / np.sqrt(right_slope**2 + 1)
                else:  # 垂直线的情况
                    dist_left = abs(x - left_intercept)
                    dist_right = abs(x - right_intercept)

                thres = 0.1  # 表示轮廓靠近旋转框的距离阈值

                if dist_top < thres * dist_top_bottom:
                    top_contour.append(point)
                if dist_bottom < thres * dist_top_bottom:
                    bottom_contour.append(point)
                if dist_left < thres * dist_left_right:
                    left_contour.append(point)
                if dist_right < thres * dist_left_right:
                    right_contour.append(point)

                # # 根据距离判断该点属于哪个边
                # min_dist = min(dist_top, dist_bottom, dist_left, dist_right)

                # if min_dist == dist_top and dist_top < thres * dist_top_bottom:
                #     top_contour.append(point)
                # elif min_dist == dist_bottom and dist_bottom < thres * dist_top_bottom:
                #     bottom_contour.append(point)
                # elif min_dist == dist_left and dist_left < thres * dist_left_right:
                #     left_contour.append(point)
                # elif min_dist == dist_right and dist_right < thres * dist_left_right:
                #     right_contour.append(point)

            # print(f"top_contour,{len(top_contour)}")
            # print(f"bottom_contour,{len(bottom_contour)}")
            # print(f"left_contour,{len(left_contour)}")
            # print(f"right_contour,{len(right_contour)}")
        
            # # 将分割结果转换为numpy数组
            # top_contour = np.array(top_contour, dtype=np.int32).reshape((-1, 1, 2))
            # bottom_contour = np.array(bottom_contour, dtype=np.int32).reshape((-1, 1, 2))
            # left_contour = np.array(left_contour, dtype=np.int32).reshape((-1, 1, 2))
            # right_contour = np.array(right_contour, dtype=np.int32).reshape((-1, 1, 2))

            # cv2.polylines(img, [top_contour], True, (0, 255, 0), 2)
            # cv2.polylines(img, [bottom_contour], True, (0, 0, 255), 2)
            # cv2.polylines(img, [left_contour], True, (255, 0, 0), 2)
            # cv2.polylines(img, [right_contour], True, (255, 255, 0), 2)

            # # 在图像上绘制点
            # for point in top_contour:
            #     x, y = point[0], point[1]
            #     cv2.circle(img, (int(x), int(y)), radius=1, color=(0, 255, 0), thickness=-1)  # 绿色点表示上边界

            # for point in bottom_contour:
            #     x, y = point[0], point[1]
            #     cv2.circle(img, (int(x), int(y)), radius=1, color=(0, 0, 255), thickness=-1)  # 红色点表示下边界

            # for point in left_contour:
            #     x, y = point[0], point[1]
            #     cv2.circle(img, (int(x), int(y)), radius=1, color=(255, 0, 0), thickness=-1)  # 蓝色点表示左边界

            # for point in right_contour:
            #     x, y = point[0], point[1]
            #     cv2.circle(img, (int(x), int(y)), radius=1, color=(255, 255, 0), thickness=-1)  # 黄色点表示右边界



        cv2.imwrite("temp.png", img)
        # cv2.imwrite(os.path.join("/home/zt/Project/Detect_Segment/ultralytics/result_HAB_2560", img_name+".png"), img)
        