### TARGET TRIANGULATION
### 3DOM - FBK - TRENTO - ITALY
# Configuration file
# Please, change the input directories with yours.
# Note: Set 0.5 for the image_translation_vector_X and image_translation_vector_Y parameters, when you mark targets with OpenCV or other tools which have the reference system placed in the middle of the first pixel, while COLMAP has the reference system placed in the upper-left corner.
# GCPs LABEL MUST BE AN INTEGER!!!

import argparse

# image_folder 是存放原始图片的文件夹的位置
# ImgExtension 是图片文件的后缀名，一般是".jpg"
# projection_folder 应该是target_projections这个文件夹，里面存放了某几张图片中的gcp的位置（像素位置）,但是和像素位置差了4.03左右的比例。很奇怪？？？？？？？？
# ProjectionDelimeter 表示project中的每个txt文件每一行里的分隔符，当前应该是空格。
# sparse model 存放colmap稀疏重建结果的文件夹
# GroundTruth 存放gcp实际三维点坐标的文件
# ColmapExe 就是colmap可执行文件的存放文件夹位置
# AlignerExe 在ubuntu下直接选择'./AlignCC_for_linux'
# ScaleFactor colmap重建的图像分辨率 / 提取gcp特征点的图像分辨率（比例），默认是1，但是感觉这个项目中是0.25？
"""
        "--Imgs", "/home/zt/Project/COLMAP_GroundControlPoints-main/temp/imgs",
        "--ImgExtension", ".jpg",
        "--Projections", "/home/zt/Project/COLMAP_GroundControlPoints-main/temp/target_projections",
        "--ProjectionDelimeter", " ",
        "--SparseModel", "/home/zt/Project/COLMAP_GroundControlPoints-main/temp/colmap_sparse",
        "--GroundTruth", "/home/zt/Project/COLMAP_GroundControlPoints-main/temp/Ground_Truth.txt",
        "--ColmapExe", "/usr/local/bin/",
        "--AlignerExe", "./AlignCC_for_linux",
        "--ScaleFactor", 1,
"""

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--Imgs", help = "Path to the image folder", required=True)
parser.add_argument("-e", "--ImgExtension", help = "Image extension file", required=True)
parser.add_argument("-p", "--Projections", help = "Path to image projections", required=True)
parser.add_argument("-d", "--ProjectionDelimeter", help = "Delimeter used in the projections file", required=True)
parser.add_argument("-s", "--SparseModel", help = "Path to COLMAP sparse reconstruction", required=True)
parser.add_argument("-g", "--GroundTruth", help = "Path to ground truth file", required=True)
parser.add_argument("-c", "--ColmapExe", help = "Path to the COLMAP exe", required=True)
parser.add_argument("-a", "--AlignerExe", help = "Path to aligner exe", required=True, choices=['./AlignCC_for_linux', './AlignCC_for_windows'])
parser.add_argument("-r", "--ScaleFactor", help = "Ratio between the image resolution used in COLMAP and the image res targets were extracted", default=1)
args = parser.parse_args()

COLMAP_EXE_PATH = args.ColmapExe
AlignCC_PATH = args.AlignerExe
image_folder = args.Imgs
projection_folder = args.Projections
sparse_model_path = args.SparseModel
ground_truth_path = args.GroundTruth
#database_path = r"./colmap_sparse/database.db"

image_file_extension = args.ImgExtension
projection_delimiter = args.ProjectionDelimeter
image_reduction_factor = float(args.ScaleFactor)                      
image_translation_vector_X = 0.5                        # X and Y value must be the same
image_translation_vector_Y = 0.5                        # X and Y value must be the same
INFO = True                                             # Get more info printed when script is running
DEBUG = False
DEBUG_level = 5                                         # 0:CHECKS
                                                        # 1:CONVERT TARGET PROJECTIONS IN COLMAP FORMAT
                                                        # 2:TARGETS MATCHING
                                                        # 3:INITIALIZE A NEW DATABASE
                                                        # 4:TARGET TRIANGULATION
                                                        # 5:FULL PIPELINE







