#!/bin/bash
# 基于GCP的colmap完整pipeline的命令行实现
reconstruct_project_path="/D/data/zt/project/colmap/scale_building/phone2"
colmap_gcp_project_path="/home/zt/Project/COLMAP_GroundControlPoints-main"
find_gcp_project_path="/home/zt/Project/Find-GCP"

################################
########## 1 确认输入 ###########
################################
# 首先确认是不是准备好了images文件夹和Ground_Truth.txt文件
read -p "你准备好了images文件夹和Ground_Truth.txt文件吗? [y/n] " ans
if [ "$ans" == "y" ]; then
  # 继续执行脚本后续命令
  echo "Continuing..."
  # ...
else
  # 用户输入n,退出脚本
  echo "Exiting..."
  exit
fi

###################################
########## 2 图像识别gcp ###########
###################################
# 在图像中找gcp二维坐标点
cd "$find_gcp_project_path"
folder="$reconstruct_project_path/images"

# 遍历文件夹下的所有图片文件
for file in "$folder"/*.png "$folder"/*.jpg "$folder"/*.jpeg; do
    # 检查文件是否存在
    if [ -f "$file" ]; then
        # 调用gcp识别并传递图片文件作为参数
        python gcp_find.py "$file" --output txt --error 0.33
    fi
done

# 做可视化
python gcp_vis.py --image_folder "$folder"

###################################
############ 3 基础sfm ############
###################################
cd "$reconstruct_project_path"
mkdir sparse
mkdir sparse/model
colmap feature_extractor --ImageReader.camera_model SIMPLE_RADIAL --database_path db.db --image_path images
colmap exhaustive_matcher --database_path db.db
colmap mapper --database_path db.db --image_path images --output_path sparse/model
# 将sfm的内外参由bin文件转成txt文件,便于后续读取处理
colmap model_converter --input_path sparse/model/0 --output_path sparse/model/0 --output_type TXT

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

#######################################
############# 6 转换相机模型 ###########
#######################################
cd "$colmap_gcp_project_path"
python transform_images.py --project_folder "$reconstruct_project_path" --align_output_path "$colmap_gcp_project_path/output/outs.txt"

#######################################
############# 7 所有点三角化 ###########
#######################################
cd "$reconstruct_project_path"
mkdir sparse/real_model
colmap point_triangulator --database_path db.db --image_path images --input_path sparse/model/manual --output_path sparse/real_model

#######################################
################ 8 mvs步骤 ############
#######################################
cd "$reconstruct_project_path"
# 图像去畸变  
# 注意如果是用原始的sfm结果应该用sparse/model/0  如果是用align之后的sfm结果，应该用sparse_align/model
mkdir dense
colmap image_undistorter \
    --image_path ./images \
    --input_path ./sparse/real_model \
    --output_path ./dense \
    --output_type COLMAP 

# 稠密重建
colmap patch_match_stereo \
    --workspace_path ./dense \
    --workspace_format COLMAP \
    --PatchMatchStereo.geom_consistency true

# 深度图融合
colmap stereo_fusion \
    --workspace_path ./dense \
    --workspace_format COLMAP \
    --input_type geometric \
    --output_path ./dense/result.ply

#######################################
################ 9 mesh步骤 ############
#######################################
colmap poisson_mesher \
    --input_path ./dense/result.ply \
    --output_path ./dense/meshed-poisson.ply