"""
这个文件用于将没有任何误差的gps data，例如从ue中得到的，加入人为的高斯噪声，写成新的gps文件
"""
import random
import numpy as np

def read_ue_gps_data(gps_file_path):
    """
    :param gps_file_path: gps文件位置
    :return: tuple：pose_list(img_name,x,y,z,pitch,yaw,roll)
    事实上应当注意，对于unreal engine，xyz对应roll，pitch,yaw
    """
    with open(gps_file_path, "r") as f:
        lines = f.readlines()

    pose_list = list()  # 存储所有pose_list

    for line in lines:
        line = line.strip()
        img_name, x, y, z, pitch, yaw, roll = line.split(",")
        pose_list.append((img_name, x, y, z, pitch, yaw, roll))

    return pose_list


def noiseize_gps_data(pose_list, position_max_error=10, angle_max_error=1):
    """
    :param angle_max_error: 最大角度误差
    :param position_max_error: 最大位置误差
    :param pose_list:(img_name,x,y,z,pitch,yaw,roll)
    :return:noise_pose (x,y,z,pitch,yaw,roll)
    """

    def uniform_noise(max_noise):
        """
        加平均噪声
        """
        return random.uniform(-1*max_noise, max_noise)

    def gaussian_noise(max_noise):
        """
        加高斯噪声,scale可以再调整，是标准差
        """
        noise = np.random.normal(loc=0, scale=max_noise/3)
        noise = np.clip(noise, -1*max_noise, max_noise)  # 确保在max_noise范围内
        return noise

    noise_pose_list = list()

    for pose in pose_list:
        img_name, x, y, z, pitch, yaw, roll = pose

        # 假定GPS为厘米级定位精度和1度精度,也就是加

        noise_x = format(float(x) + gaussian_noise(position_max_error), ".3f")  # 当前单位为厘米
        noise_y = format(float(y) + gaussian_noise(position_max_error), ".3f")  # 当前单位为厘米
        noise_z = format(float(z) + gaussian_noise(position_max_error), ".3f")  # 当前单位为厘米
        noise_pitch = format(float(pitch) + gaussian_noise(angle_max_error), ".3f")  # 当前单位为度
        noise_yaw = format(float(yaw) + gaussian_noise(angle_max_error), ".3f")  # 当前单位为度
        noise_roll = format(float(roll) + gaussian_noise(angle_max_error), ".3f")  # 当前单位为度

        noise_pose = (img_name, noise_x, noise_y, noise_z, noise_pitch, noise_yaw, noise_roll)
        noise_pose_list.append(noise_pose)

    return noise_pose_list


def write_ue_gps_data(noise_pose_list, new_gps_file_path):
    """
    :param new_gps_file_path: 新gps文件存放位置
    :param noise_pose_list:(img_name,x,y,z,pitch,yaw,roll)
    :return:write a new gps file
    """
    with open(new_gps_file_path, "w") as f:
        for noise_pose in noise_pose_list:
            f.write("{0},{1},{2},{3},{4},{5},{6}\n".format(*noise_pose))
    print("new noisy gps file is written to ---{}---".format(new_gps_file_path))


if __name__ == "__main__":
    ###############################################
    ########## 自己输入参数内容 ######################
    ###############################################

    gps_file_path = "/D/data/zt/project/colmap/exp_R50_pitch40_angle360_45image_noisy/ue_pos.txt"
    position_max_error = 10  # 最大定位误差（单位为厘米）
    angle_max_error = 1  # 最大角度误差（单位为度）

    ###############################################
    ########### 后续程序 ###########################
    ##############################################

    new_gps_file_path = gps_file_path[:-4] + "_noisy.txt"  # 存放加噪声的gps文件的位置

    pose_list = read_ue_gps_data(gps_file_path)  # 读取不同形式的pose
    noise_pose_list = noiseize_gps_data(pose_list, position_max_error, angle_max_error)  # 加噪声，可以自己控制位置误差和角度误差
    write_ue_gps_data(noise_pose_list, new_gps_file_path)  # 写入新的txt文件中

    print("**************Mission Completed***********")
