"""
在重建之前,我们把相机内参写入到database里面,以便使得相机内参是已知的。
1) 把已知的cameras.txt,写入database里面,也就是改写camera这个table
2) images.txt的camera_id都改成1

在这个文件之前需要首先运行colmap的feature extracter,为后续提供初始的camera和db
"""
import numpy as np
import os
from scipy.spatial.transform import Rotation
import sys
import sqlite3
import copy
import argparse

################################################
#############step 0 准备一些路径的东西############
################################################
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

######################################################
#############step 1 重写camera database################
######################################################
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
        删除数据库中的所有camera,准备重写
        """
        query = "DELETE FROM cameras"
        self.execute(query)
    
    def updata_images(self):
        """
        把images里面的camera_id都改成1
        """
        self.execute("UPDATE images SET camera_id = 1")


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
    print("已删除所有camera内容,准备写入新内容")

    # Import own cameras
    for camera in cameras:
        camera_id, camera_type, width, height, camera_params = camera
        camera_type = camera_dict[camera_type]
        width, height = int(width), int(height)
        camera_parameters = np.fromstring(camera_params, dtype=float, sep=' ')
        db.add_camera(camera_type, width, height, camera_parameters, camera_id=camera_id)   

    # 更新images中的camera_id
    db.updata_images()

    db.commit()
    db.close()

updateDB_camera(database_path, cameras_path)
print("****根据cameras.txt改写database完成****")

