"""
这个文件用于改写images.txt,从而对齐database中的imageID和images.txt中的ID(只是对齐ID)
同时把camera_id都改成1,因为我们只使用了一个camera
并且能够创建一个空的points3D.txt
"""
import os
import sys
import sqlite3
import numpy as np
import copy
import argparse


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
    """
    原colmap自带的python scripts内容
    """
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
        params = np.asarray(params, np.float64)
        cursor = self.execute(
            "INSERT INTO cameras VALUES (?, ?, ?, ?, ?, ?)",
            (camera_id, model, width, height, array_to_blob(params),
             prior_focal_length))
        return cursor.lastrowid

    def add_image(self, name, camera_id,
                  prior_q=np.full(4, np.NaN), prior_t=np.full(3, np.NaN), image_id=None):
        cursor = self.execute(
            "INSERT INTO images VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (image_id, name, camera_id, prior_q[0], prior_q[1], prior_q[2],
             prior_q[3], prior_t[0], prior_t[1], prior_t[2]))
        return cursor.lastrowid

    def add_keypoints(self, image_id, keypoints):
        assert(len(keypoints.shape) == 2)
        assert(keypoints.shape[1] in [2, 4, 6])

        keypoints = np.asarray(keypoints, np.float32)
        self.execute(
            "INSERT INTO keypoints VALUES (?, ?, ?, ?)",
            (image_id,) + keypoints.shape + (array_to_blob(keypoints),))

    def add_descriptors(self, image_id, descriptors):
        descriptors = np.ascontiguousarray(descriptors, np.uint8)
        self.execute(
            "INSERT INTO descriptors VALUES (?, ?, ?, ?)",
            (image_id,) + descriptors.shape + (array_to_blob(descriptors),))

    def add_matches(self, image_id1, image_id2, matches):
        assert(len(matches.shape) == 2)
        assert(matches.shape[1] == 2)

        if image_id1 > image_id2:
            matches = matches[:,::-1]

        pair_id = image_ids_to_pair_id(image_id1, image_id2)
        matches = np.asarray(matches, np.uint32)
        self.execute(
            "INSERT INTO matches VALUES (?, ?, ?, ?)",
            (pair_id,) + matches.shape + (array_to_blob(matches),))

    def add_two_view_geometry(self, image_id1, image_id2, matches,
                              F=np.eye(3), E=np.eye(3), H=np.eye(3),
                              qvec=np.array([1.0, 0.0, 0.0, 0.0]),
                              tvec=np.zeros(3), config=2):
        assert(len(matches.shape) == 2)
        assert(matches.shape[1] == 2)

        if image_id1 > image_id2:
            matches = matches[:,::-1]

        pair_id = image_ids_to_pair_id(image_id1, image_id2)
        matches = np.asarray(matches, np.uint32)
        F = np.asarray(F, dtype=np.float64)
        E = np.asarray(E, dtype=np.float64)
        H = np.asarray(H, dtype=np.float64)
        qvec = np.asarray(qvec, dtype=np.float64)
        tvec = np.asarray(tvec, dtype=np.float64)
        self.execute(
            "INSERT INTO two_view_geometries VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (pair_id,) + matches.shape + (array_to_blob(matches), config,
             array_to_blob(F), array_to_blob(E), array_to_blob(H),
             array_to_blob(qvec), array_to_blob(tvec)))


def get_db_images(database_path):
    """
    从colmap的database中读取image信息
    """
    if os.path.exists(database_path)==False:
        print("ERROR: database path dosen't exist -- please check db.db.")
        return

    # Open the database.
    db = COLMAPDatabase.connect(database_path)

    rows = db.execute("SELECT * FROM images")

    image_dict = dict()
    # Process the data
    for row in rows:
        # Access individual columns of each row
        print(row)
        image_id = row[0]
        image_name = row[1]
        image_dict[image_name] = image_id

    # # 把所有图片的camera_id都改成1，因为我们当前确实只使用了一个camera。
    # query = "UPDATE images SET camera_id = 1"
    # db.execute(query)

    # 确认修改db
    db.commit()

    # 关闭database的连接
    db.close()

    return image_dict

def rewrite_image_txt(database_path, images_path):
    """
    重写image_ID
    """
    image_dict_from_db = get_db_images(database_path)
    # Copy the camera orientation parameters leaving empty the row for the keypoint projections
    images_param_and_ori = [] # It will contain IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME for each image
    #new_file = open('{}'.format(final_images), 'w')
    with open(images_path, 'r') as f:
        lines = f.readlines()
        lines_head = lines[:4]  # 文件头里附加的一些内容
        lines_content = lines[4:]  # 每张image的具体位姿信息
        correct_lines_content = copy.deepcopy(lines_content)

        for c,line in enumerate(lines_content):
            if c%2 == 0:
                line = line.strip()
                img_params = line.split(" ", 9)
                images_param_and_ori.append(img_params)

                # 读取image_id和name,并且从db数据库中找到正确的image_id
                image_name = img_params[-1]
                false_id = img_params[0]
                correct_id = image_dict_from_db[image_name]

                # 替换正确的图片编号，并且替换lines内容
                correct_line = str(correct_id) + line[len(false_id):] + "\n"
                correct_lines_content[c] = correct_line
            
            else:
                correct_line = "\n"
                correct_lines_content[c] = correct_line

    # 确保images.txt最后一定有个空行
    if correct_lines_content[-1] != "\n":
        correct_lines_content.append("\n")

    with open(images_path, "w") as new_images_file:
        new_images_file.writelines(lines_head)
        new_images_file.writelines(correct_lines_content[:])


def create_empty_points3d(points3D_path):
    """
    创建一个完全空的points3D的txt文件,便于直接import model进行重建
    """
    with open(points3D_path, "w") as f:
        pass

    
# 传入重建项目的文件夹的路径
parser = argparse.ArgumentParser()
parser.add_argument("-p", "--project_folder", help = "colmap重建项目的文件夹路径,示例/D/data/zt/project/colmap/test_with_gcp", required=True)
args = parser.parse_args()

camera_path = os.path.join(args.project_folder,"manual/model/cameras.txt")
images_path = os.path.join(args.project_folder,"manual/model/images.txt")
points3D_path = os.path.join(args.project_folder,"manual/model/points3D.txt")
database_path = os.path.join(args.project_folder,"db.db")

rewrite_image_txt(database_path, images_path)
print("根据database重写images.txt中的imageID完成")

create_empty_points3d(points3D_path)
print("创建一个空的images3D文件完成")



