### TARGET TRIANGULATION
### 3DOM - FBK - TRENTO - ITALY
# Main
# Please, change the input directories with yours in the config.py file.

print('\nTARGET TRIANGULATION')

# Importing libraries
print('\nImporting libraries ...')
import cv2
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import subprocess

# Importing other scripts
print('Importing other scripts ...\n')
import config
from lib import RearrangeProjectionsIDXY
from lib import matching
from lib import checks
from lib import database
from lib import read_existing_db
#from lib import ExportColmapCameras

# Define the class to store the triangulated targets in COLMAP as objects
class target3D:
    def __init__(self, t3D_id, x, y, z, r, g, b, error, track):
        self.t3D_id = t3D_id
        self.x = x
        self.y = y
        self.z = z
        self.r = r
        self.g = g
        self.b = b
        self.error = error
        self.track = track

##################################################################################
# MAIN STARTS HERE

output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
else:
    shutil.rmtree(output_dir)           
    os.makedirs(output_dir)

# 打印一下输入的文件，这些文件都要在运行main.py的时候作为参数输入

print("N of images: \t\t\t{}".format(len(os.listdir(config.image_folder))))
print("Image folder: \t\t\t{}".format(config.image_folder))
print("Projections folder: \t\t{}".format(config.projection_folder))
print("Sparse model: \t\t\t{}".format(config.sparse_model_path))
print("Projection reduction factor: \t{}".format(config.image_reduction_factor))
print("Projection delimiter: \t\t'{}'".format(config.projection_delimiter))
print("Show more info: \t\t{}".format(config.INFO)) 
print("DEBUG_bool: \t\t\t{}".format(config.DEBUG)) 
print("DEBUG_level: \t\t\t{}\n".format(config.DEBUG_level))   

# Manually check if inserted directories are correct
#userIO = input("Would you continue? y/n\n")
#print('\n')
#if userIO != "y":
#    quit()

# 验证一下输入的数据是否正确，并且看一下有多少个控制点
print("Checks on input data ...")
checks.checks(config.image_folder, config.projection_folder, config.projection_delimiter)
if config.DEBUG == True and config.DEBUG_level == 0:
    quit()

# 把原有的projection的点的像素坐标调整成colmap形式的，主要是projection*ratio+0.5（大概是光心坐标）- sample里的比例改成0.25
print("\nConverting target projections in COLMAP format ...")
RearrangeProjectionsIDXY.RearrangeProjectionsIDXY(config.image_folder, config.projection_folder, config.projection_delimiter)
if config.DEBUG == True and config.DEBUG_level == 1:
    quit()

# 匹配图像中的gcp点，以output中的matches.txt存储下来，0 0 表示第一张图中的第0个gcp和第二张图中的第0个gcp是匹配的。
print("\nTargets matching ...")
all_matches = matching.Matching(config.image_folder, config.projection_folder, config.projection_delimiter)
if config.DEBUG == True and config.DEBUG_level == 2:
    quit()

# AUTOMATICALLY IMPORT THE TARGETS IN COLMAP
print("\nTargets triangulation ...")
# Rearrange the sparse model output
original_cameras = "{}/cameras.txt".format(config.sparse_model_path)
original_images = "{}/images.txt".format(config.sparse_model_path)
os.mkdir("output/temp")
final_cameras = "output/temp/cameras.txt"
final_images = "output/temp/images.txt"
final_points3D = "output/temp/points3D.txt"

# Copy the camera file cameras.txt and create an empty txt file to store in future the triangulated coordiantes
shutil.copyfile('{}'.format(original_cameras), '{}'.format(final_cameras))
new_file = open('{}'.format(final_points3D), 'w')
new_file.close()

# Copy the camera orientation parameters leaving empty the row for the keypoint projections
images_param_and_ori = [] # It will contain IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME for each image
#new_file = open('{}'.format(final_images), 'w')
with open('{}'.format(original_images), 'r') as lines:
    lines = lines.readlines()[4:]
    for c,line in enumerate(lines):
        if c%2 == 0:
            #new_file.write(line)
            line = line.strip()
            img_params = line.split(" ", 9)
            images_param_and_ori.append(img_params)
        #else:
            #new_file.write("\n")
#new_file.close()

# 导入camera模型
cameras = []
with open(final_cameras, "r") as camera_models:
    lines = camera_models.readlines()[3:]
    for line in lines:
        line = line.strip()
        cameras.append(line.split(" ", 4))

# 创建一个新的colmap database
image_list = []
new_file = open('{}'.format(final_images), 'w')
for counter, i in enumerate(images_param_and_ori):
    IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME = i
    image_list.append(NAME)
    NEW_IMAGE_ID = counter + 1
    i = NEW_IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
    new_file.write("{} {} {} {} {} {} {} {} {} {}\n\n".format(NEW_IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME))
new_file.close()


image_dict, matches_cont = database.newDB(cameras, image_list, all_matches, config.projection_delimiter, images_param_and_ori, config.image_file_extension)
if config.DEBUG == True and config.DEBUG_level == 3:
    quit()

## EXPERIMENTAL
##images_param_and_ori = []
#reverse_image_dict = {v: k for k, v in image_dict.items()}
#new_file = open('{}'.format(final_images), 'w')
#for i in images_param_and_ori:
#    IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME = i
#    if 
#        
#with open('{}'.format(original_images), 'r') as lines:
#    lines = lines.readlines()[4:]
#    for c,line in enumerate(lines):
#        if c%2 == 0:
#            new_file.write(line)
#            line = line.strip()
#            img_params = line.split(" ", 9)
#            #images_param_and_ori.append(img_params)
#        else:
#            new_file.write("\n")
#new_file.close()

# 创建一个新的project.ini
current_directory = os.getcwd()
with open("lib/project.ini", "w") as ini_file:
    ini_file.write(
f"""log_to_stderr=false
random_seed=0
log_level=2
database_path=output/db.db
image_path={config.image_folder}
input_path=output/temp
output_path=output/bin_outs""")
    ini_file.write("\n")

with open("lib/project.ini", "a") as ini_file:                    
    with open("lib/template.ini","r") as ini_options:
        ini_file.write(ini_options.read())

# gcp点的三角化（直接调用colmap程序）
os.mkdir("output/bin_outs")
print("***************")
print(r"{}".format(config.COLMAP_EXE_PATH))
print(r"{}/lib/project.ini".format(current_directory))
print("***************")

if config.AlignCC_PATH == './AlignCC_for_linux':
    # 这一步没法让camera refine，看看怎么解决，怎么提高。
    subprocess.run([r"{}/colmap".format(config.COLMAP_EXE_PATH), "point_triangulator", "--project_path", r"{}/lib/project.ini".format(current_directory)])
    subprocess.run([r"{}/colmap".format(config.COLMAP_EXE_PATH), "bundle_adjuster", "--project_path", r"{}/lib/project.ini".format(current_directory)])
    subprocess.run([r"{}/colmap".format(config.COLMAP_EXE_PATH), "point_triangulator", "--project_path", r"{}/lib/project.ini".format(current_directory)])
elif config.AlignCC_PATH == './AlignCC_for_windows':
    subprocess.run([r"{}/COLMAP.bat".format(config.COLMAP_EXE_PATH), "point_triangulator", "--project_path", r"{}/lib/project.ini".format(current_directory)])
if config.DEBUG == True and config.DEBUG_level == 4:
    quit()

# Export ply file and convert the binary output in a txt output
os.mkdir("output/txt_outs")

# bin to ply format
if config.AlignCC_PATH == './AlignCC_for_linux':
    subprocess.run(["{}/colmap".format(config.COLMAP_EXE_PATH), "model_converter", "--input_path", "{}/output/bin_outs".format(current_directory), "--output_path", "{}/output/targets.ply".format(current_directory), "--output_type", "PLY"])
elif config.AlignCC_PATH == './AlignCC_for_windows':
    subprocess.run(["{}/COLMAP.bat".format(config.COLMAP_EXE_PATH), "model_converter", "--input_path", "{}/output/bin_outs".format(current_directory), "--output_path", "{}/output/targets.ply".format(current_directory), "--output_type", "PLY"])

# bin to txt format
if config.AlignCC_PATH == './AlignCC_for_linux':
    subprocess.run(["{}/colmap".format(config.COLMAP_EXE_PATH), "model_converter", "--input_path", "{}/output/bin_outs".format(current_directory), "--output_path", "{}/output/txt_outs".format(current_directory), "--output_type", "TXT"])
elif config.AlignCC_PATH == './AlignCC_for_windows':
    subprocess.run(["{}/COLMAP.bat".format(config.COLMAP_EXE_PATH), "model_converter", "--input_path", "{}/output/bin_outs".format(current_directory), "--output_path", "{}/output/txt_outs".format(current_directory), "--output_type", "TXT"])

# Store as object the triangulated targets
os.mkdir("output/CloudCompare")
shutil.copyfile('{}'.format(config.ground_truth_path), '{}/output/CloudCompare/GroundTruth.txt'.format(current_directory))
points3D_file = "output/txt_outs/points3D.txt"

with open(points3D_file, "r") as p3D_file:
    lines = p3D_file.readlines()[3:]
    points3D = [target3D(
                                        t3D_id = None,
                                        x = None,
                                        y = None,
                                        z = None,
                                        r = None,
                                        g = None,
                                        b = None,
                                        error = None,
                                        track = None )for i in range(len(lines))]
    
    for count, line in enumerate(lines):
        line = line.strip()
        POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK = line.split(" ", 8)
        points3D[count] = target3D(
                                        t3D_id = None,
                                        x = float(X),
                                        y = float(Y),
                                        z = float(Z),
                                        r = int(R),
                                        g = int(G),
                                        b = int(B),
                                        error = float(ERROR),
                                        track = TRACK
                                    )

# Export targets with ID in COLMAP coordinates
target_COLMAP_XYZ_file = open("output/CloudCompare/colmap.txt", "w")

total_tracks = 0
reproj_error = []
for target in points3D:
    img_id, target_id, trash = target.track.split(" ", 2)
    img_id, target_id = int(img_id), int(target_id)
    
    image_dict_mirrored = {v: k for k, v in image_dict.items()}
    image_name = image_dict_mirrored[img_id]
    
    
    with open("{}/{}.txt".format(config.projection_folder, image_name)) as proj_file:
        lines = proj_file.readlines()
        line = lines[target_id]
        line = line.strip()
        target_name, trash = line.split(config.projection_delimiter, 1)
        #target.t3D_id = int(target_name)
        target.t3D_id = target_name
        if target.t3D_id[0:3] == 'gcp':
            tar = target.t3D_id[3:]
        else:
            tar = target.t3D_id
        target_COLMAP_XYZ_file.write("{},{},{},{}\n".format(tar, target.x, target.y, target.z))
    
    tracks = target.track.split(" ")
    total_tracks += len(tracks)/2
    
    reproj_error.append(target.error)

target_COLMAP_XYZ_file.close()

# Check that colmap.txt and ground_truth.txt contain the same number of points
colmap_coord = {}
with open("output/CloudCompare/colmap.txt", "r") as f1:
    lines = f1.readlines()
    for line in lines:
        id, x, y, z = line.strip().split(',', 5)
        colmap_coord[id] = (x, y, z)

gt_coord = {}
with open("{}".format(config.ground_truth_path), 'r') as f2:
    lines = f2.readlines()
    for line in lines:
        id, x, y, z = line.strip().split(',', 5)
        gt_coord[id] = (x, y, z)

common_keys = colmap_coord.keys() & gt_coord.keys()
with open("output/CloudCompare/colmap_common_keys.txt", "w") as fx, open("output/CloudCompare/gt_common_keys.txt", "w") as fy:
    for key in common_keys:
        fx.write("{},{},{},{}\n".format(key, colmap_coord[key][0], colmap_coord[key][1], colmap_coord[key][2]))
        fy.write("{},{},{},{}\n".format(key, gt_coord[key][0], gt_coord[key][1], gt_coord[key][2]))

# 直到这里才用到了gt的值，并基于投影的gcp输出了最后的R|t矩阵
output_file = open("output/outs.txt", "w")
#subprocess.run(["{}/align".format(config.AlignCC_PATH), "{}/output/CloudCompare/colmap.txt".format(current_directory), "{}".format(config.ground_truth_path), ">", "{}/output/CloudCompare/out.txt"], stdout=output_file)
subprocess.run(["{}/align".format(config.AlignCC_PATH), "{}/output/CloudCompare/colmap_common_keys.txt".format(current_directory), "{}/output/CloudCompare/gt_common_keys.txt".format(current_directory), ">", "{}/output/CloudCompare/out.txt"], stdout=output_file)
output_file.close()

# Check if all targets and projections are used
with open("output/outs.txt", "a") as output_file:
    output_file.write("\n- SUMMARY -\n")
    output_file.write("Nummber of targets: {}\n".format(len(points3D)))
    output_file.write("Number of total projections: {}\n".format(total_tracks))
    output_file.write("Mean Reprojection Error: {} pix\n".format(np.mean(reproj_error)))
    output_file.write("Standard Deviation: {} pix\n".format(np.std(reproj_error)))
    output_file.write("N of TOTAL matches: {}".format(matches_cont))

# Print results
print("\n\n")
with open("output/outs.txt", "r") as output_file:
    lines = output_file.readlines()
    for line in lines:
        line = line.strip()
        print(line)

# Export cameras
#external_cameras_path = "output/txt_outs/images.txt"
#camera_ori = ExportColmapCameras.ExportCameras(external_cameras_path)
#out_file = open("output/cameras_extr.txt", 'w')
#for element in camera_ori:
#    out_file.write(element)
#    out_file.write('\n')
#out_file.close()

### END
print('\nEND')

if config.DEBUG == True and config.DEBUG_level == 5:
    quit()