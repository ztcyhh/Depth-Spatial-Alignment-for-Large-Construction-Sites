#!/bin/bash
# 基于GCP的colmap完整pipeline的命令行实现
reconstruct_project_path="/D/data/zt/project/colmap/xinyang/0418/tower1/simple_radial_beifen"
colmap_gcp_project_path="/home/zt/Project/COLMAP_GroundControlPoints-main"
yolo_project_path="/home/zt/Project/Detect_Segment/ultralytics"

# ################################
# ########## 1 确认输入 ###########
# ################################
# # 首先确认是不是准备好了images文件夹和Ground_Truth.txt文件
# read -p "你准备好了images文件夹和Ground_Truth.txt文件吗? [y/n] " ans
# if [ "$ans" == "y" ]; then
#   # 继续执行脚本后续命令
#   echo "Continuing..."
#   # ...
# else
#   # 用户输入n,退出脚本
#   echo "Exiting..."
#   exit
# fi


#######################################
######## 4 整理数据，准备gcp三角化 ######
#######################################
# 在对齐gcp的项目中创建一个temp文件夹（以后都可以放到这个文件夹里）
cd "$colmap_gcp_project_path"
if [ -d "temp" ]; then
  rm -rf temp
fi
mkdir temp

# 将images文件夹复制到imgs下面
cp -r "$reconstruct_project_path/images" "$colmap_gcp_project_path/temp/imgs"

# 将txt_output文件夹复制到target_projections
cp -r "$reconstruct_project_path/txt_output" "$colmap_gcp_project_path/temp/target_projections"

# 将Ground_Truth.txt文件移动到Ground_Truth.txt文件
cp "$reconstruct_project_path/Ground_Truth.txt" "$colmap_gcp_project_path/temp/Ground_Truth.txt"

# 创建colmap_sparse文件夹，将sparse model复制到下面
mkdir temp/colmap_sparse
cp "$reconstruct_project_path/sparse/model/0/"*.txt "$colmap_gcp_project_path/temp/colmap_sparse"

#######################################
######## 5 做gcp三角化和求变换矩阵 ######
#######################################
cd "$colmap_gcp_project_path"

# 删除空的txt文件
for file in "/home/zt/Project/COLMAP_GroundControlPoints-main/temp/target_projections/"*.txt; do
    # 检查文件是否为空
    if [ ! -s "$file" ]; then
        # 文件为空，删除文件
        echo "Deleting empty file: $file"
        rm "$file"
    fi
done

python main.py --Imgs /home/zt/Project/COLMAP_GroundControlPoints-main/temp/imgs \
               --ImgExtension .jpg \
               --Projections /home/zt/Project/COLMAP_GroundControlPoints-main/temp/target_projections \
               --ProjectionDelimeter " " \
               --SparseModel /home/zt/Project/COLMAP_GroundControlPoints-main/temp/colmap_sparse \
               --GroundTruth /home/zt/Project/COLMAP_GroundControlPoints-main/temp/Ground_Truth.txt \
               --ColmapExe /usr/local/bin/ \
               --AlignerExe ./AlignCC_for_linux \
               --ScaleFactor 1

# #######################################
# ############# 6 转换相机模型 ###########
# #######################################
# cd "$colmap_gcp_project_path"
# python rewrite_images_for_gcp.py --project_folder "$reconstruct_project_path"
# python transform_images.py --project_folder "$reconstruct_project_path" --align_output_path "$colmap_gcp_project_path/output/outs.txt"
