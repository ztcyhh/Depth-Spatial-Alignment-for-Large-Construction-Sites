"""
这个文件是为了将colmap生成的images.txt中的相机位姿能够可视化出来，从而直观查看位姿是否正确
当前还差将R也转为欧拉角从而可视化出来
"""

import numpy as np
from scipy.spatial.transform import Rotation

def write_back_colmap_images(images_path, new_images_path):
    with open(images_path, "r") as f:
        lines = f.readlines()

    pose_list = list()
    for index, line in enumerate(lines[4::2]):
        # 读取images的外参
        param = [float(num) for num in line.split()[1:8]]
        qw, qx, qy, qz, tx, ty, tz = param
        q = np.array([qx, qy, qz, qw])  # 用scipy的形式,行向量
        t = np.array([tx, ty, tz]).reshape(-1, 1)

        # 根据colmap自己的规则，计算光心坐标
        R = Rotation.from_quat(q).as_matrix()
        t1 = -np.dot(np.transpose(R), t)

        # 读取图片名称
        img_name = line.split()[-1]
        # 写成poselist
        pose = (img_name, t1[0,0], t1[1,0], t1[2,0])
        pose_list.append(pose)

    with open(new_images_path, "w") as f:
        for pose in pose_list:
            f.write("{0},{1},{2},{3}\n".format(*pose))
    

if __name__ == "__main__":

    images_path = "/D/data/zt/project/colmap/exp_R50_pitch40_angle360_45image_noisy/sparse_align/images.txt"
    new_images_path = "/D/data/zt/project/colmap/exp_R50_pitch40_angle360_45image_noisy/sparse_align/images_readable.txt"

    write_back_colmap_images(images_path, new_images_path)

    print("over")
        