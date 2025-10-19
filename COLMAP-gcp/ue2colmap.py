"""
这个项目主要用于
1) 将ue4中的pos信息转换成colmap所需的images.txt
2) 把已知的cameras.txt,写入database里面
3) 创建一个空的points3D.txt

在这个文件之前需要首先运行colmap的feature extracter,为后续提供初始的camera和db
"""
import numpy as np
import os
from scipy.spatial.transform import Rotation
import sys
import sqlite3
import copy
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


def euler2quaternion(euler):
    """
    输入:pitch yaw roll顺序的欧拉角
    输出:四元数(符合colmap)
    """
    # 注意unreal engine中pitch yaw roll对应的是yzx，所以首先转成xyz
    pitch, yaw, roll = euler
    r = Rotation.from_euler('xyz', [roll, pitch, yaw], degrees=True)
    rm = r.as_quat()
    return rm


def euler2matrix(euler):
    """
    输入:pitch yaw roll顺序的欧拉角
    输出:旋转矩阵
    """
    # 注意unreal engine中pitch yaw roll对应的是yzx，所以首先转成xyz
    pitch, yaw, roll = euler
    r = Rotation.from_euler('xyz', [roll, pitch, yaw], degrees=True)
    rm = r.as_matrix()
    return rm

def matrix2quaternion(matrix):
    """
    输入：旋转矩阵
    输出：四元数
    """
    q = Rotation.from_matrix(matrix).as_quat()
    return q

# def ue2colmap(R,t):
#     """
#     将坐标系从xyz转成zxy
#     """
#     S = np.array([[0,0,1],
#                   [1,0,0],
#                   [0,-1,0]])
#     R1 = np.dot(np.dot(S,R),S)
#     t1 = np.dot(S,t)
    
#     return R1,t1   

# def flip_axis(R,t):
#     """
#     主要用于左手坐标系下的R和t,向右手坐标系下的转换（反之也是一样的），
#     !!!!!!!!! 注：在这里是颠倒第一个轴的方向 !!!!!!!!!
#     以后也可以交换别的轴的方向
#     输入:左/右手坐标系下的R和t
#     输出:右/左手坐标系下的R1和t1
#     """
#     S = np.array([[-1,0,0],
#                   [0,1,0],
#                   [0,0,1]])
#     R1 = np.dot(np.dot(S,R),S)
#     t1 = np.dot(S,t)
    
#     return R1,t1


def ue2colmap(R,t):
    """
    将ue的w2c转换为colmap的w2c
    """
    S1 = np.array([[0,0,1],
                   [1,0,0],
                   [0,-1,0]])  # xyz to zx-y  row
    
    S2 = np.array([[0,1,0],
                   [0,0,1],
                   [1,0,0]])  # xyz to y-zx   row
    
    W = np.array([[-1,0,0],
                  [0,1,0],
                  [0,0,-1]])  
    
    W1 = np.array([[1,0,0],
                   [0,1,0],
                   [0,0,-1]])  # xyz to y-zx   row
    
    R1 = S2 @ R @ W
    t1 = S2 @ t


    return R1,t1

####################################
#############step 0 ################
####################################
parser = argparse.ArgumentParser()
parser.add_argument("-p", "--project_folder", help = "colmap重建项目的文件夹路径,示例/D/data/zt/project/colmap/test_with_gcp", required=True)
args = parser.parse_args()

project_folder = args.project_folder  # colmap的项目文件夹
pos_path = os.path.join(project_folder,"ue_pos.txt")
images_path = os.path.join(project_folder,"manual/model/images.txt")
# images_path = "images_test.txt"  # 作为测试路径
images_template_path = "images_template.txt"
cameras_path = os.path.join(project_folder,"manual/model/cameras.txt")
cameras_template_path = "cameras_template.txt"
points3D_path = os.path.join(project_folder,"manual/model/points3D.txt")
database_path = os.path.join(project_folder, "db.db")

####################################
######## step 1 读取ue的pose ########
####################################
ue_pos_dict = read_ue_pos(pos_path)
print("*****1.读取ue4中的pose信息完毕*****")
# print(ue_pos_dict)

####################################
###### step 2 转成COLMAP参数输入#####
####################################
# 改写一下pose 按着images.txt的格式 IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
IMAGE_ID = 0
IMAGE_params = list()

for NAME, values in ue_pos_dict.items():
    x, y, z, pitch, yaw, roll = values

    ######################################################################################
    ########## step2.1 euler to matrix (camera to world)##################################
    R_c2w_ue = euler2matrix([pitch, yaw, roll])
    t_c2w_ue = np.array([x,y,z]).reshape(3,1)/1000  # 以米为单位！！

    ######################################################################################
    ########## step2.2 world to camera (ue camera) ########################################
    R_w2c_ue = np.transpose(R_c2w_ue)
    t_w2c_ue = -np.transpose(R_c2w_ue) @ t_c2w_ue

    ######################################################################################
    ########## step2.3 ue camera to colmap camera ########################################
    R_w2c_colmap, t_w2c_colmap = ue2colmap(R_w2c_ue, t_w2c_ue)

    ######################################################################################
    ########## step2.4 convert to colmap coordinate ######################################
    QX, QY, QZ, QW= matrix2quaternion(R_w2c_colmap)
    t_n = t_w2c_colmap
    TX, TY, TZ = t_n[0,0], t_n[1,0], t_n[2,0]
    
    ######################################################################################
    ########## step2.4 write colmap params ################################################
    # 当前只有一个camera，所以认为cameraid就是1
    CAMERA_ID = 1 

    # 按顺序赋予IMAGE_ID
    IMAGE_ID += 1

    # 写成param形式
    param = [IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME]
    IMAGE_params.append(param)

print("*****2.转换为colmap所需的位姿输入*****")

####################################
##### step 3 写入images.txt ########
####################################
with open(images_path,"w") as f:  
    with open(images_template_path,"r") as temp:
        head = temp.readlines()[:3]
        line4 = "# Number of images: {}, mean observations per image: 1\n".format(len(IMAGE_params))
    f.writelines(head)
    f.write(line4)
    for param in IMAGE_params:
        f.write("{} {} {} {} {} {} {} {} {} {}\n".format(*param))
        f.write("\n")
print("*****3.写入文件中，文件路径{}*****".format(images_path))

####################################
##### step 4 写入cameras.txt #######
####################################
# 如果只是读取database中的cameras，写成cameras.txt，这样是非常不准的，因为feature extraction中
# 如果有标定的话，就写成自己的
# 因为当前会产生n个相机，最好还是把相机个数减为1
# 类似读取数据库，把原有的相机全部删掉，再改为1号相机

IS_PYTHON3 = sys.version_info[0] >= 3

MAX_IMAGE_ID = 2**31 - 1

CREATE_CAMERAS_TABLE = """CREATE TABLE IF NOT EXISTS cameras (
    camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    model INTEGER NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    params BLOB,
    prior_focal_length INTEGER NOT NULL)"""

CREATE_DESCRIPTORS_TABLE = """CREATE TABLE IF NOT EXISTS descriptors (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)"""

CREATE_IMAGES_TABLE = """CREATE TABLE IF NOT EXISTS images (
    image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    name TEXT NOT NULL UNIQUE,
    camera_id INTEGER NOT NULL,
    prior_qw REAL,
    prior_qx REAL,
    prior_qy REAL,
    prior_qz REAL,
    prior_tx REAL,
    prior_ty REAL,
    prior_tz REAL,
    CONSTRAINT image_id_check CHECK(image_id >= 0 and image_id < {}),
    FOREIGN KEY(camera_id) REFERENCES cameras(camera_id))
""".format(MAX_IMAGE_ID)

CREATE_TWO_VIEW_GEOMETRIES_TABLE = """
CREATE TABLE IF NOT EXISTS two_view_geometries (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    config INTEGER NOT NULL,
    F BLOB,
    E BLOB,
    H BLOB,
    qvec BLOB,
    tvec BLOB)
"""

CREATE_KEYPOINTS_TABLE = """CREATE TABLE IF NOT EXISTS keypoints (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)
"""

CREATE_MATCHES_TABLE = """CREATE TABLE IF NOT EXISTS matches (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB)"""

CREATE_NAME_INDEX = \
    "CREATE UNIQUE INDEX IF NOT EXISTS index_name ON images(name)"

CREATE_ALL = "; ".join([
    CREATE_CAMERAS_TABLE,
    CREATE_IMAGES_TABLE,
    CREATE_KEYPOINTS_TABLE,
    CREATE_DESCRIPTORS_TABLE,
    CREATE_MATCHES_TABLE,
    CREATE_TWO_VIEW_GEOMETRIES_TABLE,
    CREATE_NAME_INDEX
])


def image_ids_to_pair_id(image_id1, image_id2):
    if image_id1 > image_id2:
        image_id1, image_id2 = image_id2, image_id1
    return image_id1 * MAX_IMAGE_ID + image_id2


def pair_id_to_image_ids(pair_id):
    image_id2 = pair_id % MAX_IMAGE_ID
    image_id1 = (pair_id - image_id2) / MAX_IMAGE_ID
    return image_id1, image_id2


def array_to_blob(array):
    if IS_PYTHON3:
        return array.tostring()
    else:
        return np.getbuffer(array)


def blob_to_array(blob, dtype, shape=(-1,)):
    if IS_PYTHON3:
        return np.fromstring(blob, dtype=dtype).reshape(*shape)
    else:
        return np.frombuffer(blob, dtype=dtype).reshape(*shape)


class COLMAPDatabase(sqlite3.Connection):

    @staticmethod
    def connect(database_path):
        return sqlite3.connect(database_path, factory=COLMAPDatabase)


    def __init__(self, *args, **kwargs):
        super(COLMAPDatabase, self).__init__(*args, **kwargs)

        self.create_tables = lambda: self.executescript(CREATE_ALL)
        self.create_cameras_table = \
            lambda: self.executescript(CREATE_CAMERAS_TABLE)
        self.create_descriptors_table = \
            lambda: self.executescript(CREATE_DESCRIPTORS_TABLE)
        self.create_images_table = \
            lambda: self.executescript(CREATE_IMAGES_TABLE)
        self.create_two_view_geometries_table = \
            lambda: self.executescript(CREATE_TWO_VIEW_GEOMETRIES_TABLE)
        self.create_keypoints_table = \
            lambda: self.executescript(CREATE_KEYPOINTS_TABLE)
        self.create_matches_table = \
            lambda: self.executescript(CREATE_MATCHES_TABLE)
        self.create_name_index = lambda: self.executescript(CREATE_NAME_INDEX)

    def add_camera(self, model, width, height, params,
                   prior_focal_length=False, camera_id=None):
        """
        新增camera
        """
        params = np.asarray(params, np.float64)
        cursor = self.execute(
            "INSERT INTO cameras VALUES (?, ?, ?, ?, ?, ?)",
            (camera_id, model, width, height, array_to_blob(params),
             prior_focal_length))
        return cursor.lastrowid
    
    def update_camera(self, model, width, height, params,
                   prior_focal_length=False, camera_id=None):
        """
        更新camera
        """
        params = np.asarray(params, np.float64)
        cursor = self.execute(
            "UPDATE cameras SET model=?, width=?, height=?, params=?, prior_focal_length=1 WHERE camera_id=?",
            (model, width, height, array_to_blob(params), camera_id))
    
    def delete_cameras(self):
        """
        删除数据库中的所有camera
        """
        # 读取所有的camera_id,判断是不是需要leave,不需要的话就执行删除操作
        query = "DELETE FROM cameras"
        self.execute(query)

#######################################################################################
# Changes to the original script start here
def updateDB_camera(database_path, cameras_path, image_list=False, images_param_and_ori=False):

    if os.path.exists(database_path)==False:
        print("ERROR: database path dosen't exist -- please check db.db.")
        return

    # Open the database.
    db = COLMAPDatabase.connect(database_path)

    # For convenience, try creating all the tables upfront.
    db.create_tables()

    # Camera model dictionary
    camera_dict = {
                    "SIMPLE_PINHOLE": 0,
                    "PINHOLE" : 1,
                    "SIMPLE_RADIAL" : 2,
                    "RADIAL" : 3,
                    "OPENCV" : 4,
                    "OPENCV_FISHEYE": 5,
                    "FULL_OPENCV": 6,
                    "FOV": 7,
                    "SIMPLE_RADIAL_FISHEYE" : 8,
                    "RADIAL_FISHEYE" : 9,
                    "THIN_PRISM_FISHEYE" :10
                    }
    
    cameras = []
    with open(cameras_path, "r") as camera_models:
        lines = camera_models.readlines()[3:]
        for line in lines:
            line = line.strip()
            cameras.append(line.split(" ", 4))

    # 删除原有的所有camera。
    db.delete_cameras()

    # Import own cameras
    for camera in cameras:
        camera_id, camera_type, width, height, camera_params = camera
        camera_type = camera_dict[camera_type]
        width, height = int(width), int(height)
        camera_parameters = np.fromstring(camera_params, dtype=float, sep=' ')
        db.add_camera(camera_type, width, height, camera_parameters, camera_id=camera_id)    

    db.commit()
    db.close()

updateDB_camera(database_path, cameras_path)
print("****根据cameras.txt改写database完成****")

####################################
#############step 5 ################
####################################
# 创建一个新的points3D文件
with open(points3D_path, "w") as f:
    pass

print("****创建空的points3D.txt完成****")


