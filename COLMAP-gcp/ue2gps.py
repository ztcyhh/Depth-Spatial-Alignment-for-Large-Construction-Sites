"""
这个项目主要用于
1) 将ue4中的pos信息转换成colmap的model_aligner功能所需的txt文件（其实也没有任何转化，就是一个简单的重写的工作
具体示例形式如下：
image_name1.jpg X1 Y1 Z1
image_name2.jpg X2 Y2 Z2
image_name3.jpg X3 Y3 Z3
...

"""
import os
import argparse

def read_ue_pos(ue_pos_txt):
    """
    读取的是unrealcv插件写下来的ue_pos。
    输出的是一个dict,key就是图片名称,value就是x,y,z,pitch,yaw,roll这样的一个六个数的列表。
    """
    ue_pos_dict = dict()
    
    with open(ue_pos_txt,"r") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        img_name, x, y, z, pitch, yaw, roll = line.split(",")
        ue_pos_dict[img_name] = [float(x),float(y),float(z),float(pitch),float(yaw),float(roll)]

    return ue_pos_dict

def write_gps(ue_pos_dict, output_path):
    """
    输入的是读取的ue_pos_dict
    输出的是colmap中model aligner要求的txt格式
    """
    with open(output_path,"w") as f:
        for img_name in ue_pos_dict.keys():
            x = ue_pos_dict[img_name][0]
            y = ue_pos_dict[img_name][1]
            z = ue_pos_dict[img_name][2]
            f.write("{} {} {} {}\n".format(img_name, -x, y, z))

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--project_folder", help = "colmap重建项目的文件夹路径,示例/D/data/zt/project/colmap/test_with_gcp", required=True)
args = parser.parse_args()

project_folder = args.project_folder  # colmap的项目文件夹
pos_path = os.path.join(project_folder,"ue_pos.txt")  # 在unrealcv拍摄图片的时候就拿到的xyz数据

output_path = os.path.join(project_folder,"ref_xyz.txt")
ue_pose_dict = read_ue_pos(pos_path)
write_gps(ue_pos_dict=ue_pose_dict, output_path=output_path)
print("+++++++++++++写参考坐标系下的相机xyz位置完成++++++++++++")