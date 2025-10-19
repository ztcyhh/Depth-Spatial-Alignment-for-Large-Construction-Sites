#!/bin/bash
reconstruct_project_path="/D/data/zt/project/colmap/cuptest"
colmap_gcp_project_path="/home/zt/Project/COLMAP_GroundControlPoints-main"

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