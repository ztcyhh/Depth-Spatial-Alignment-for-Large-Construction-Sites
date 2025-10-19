#!/bin/bash
# 将两个摄像头建的sparse_model给合起来,为了后续的mesh贴图做准备
########### 先准备基础的路径 ############
colmap_gcp_project_path="/home/zt/Project/COLMAP_GroundControlPoints-main"
project1_path="/D/data/zt/project/colmap/xiongan1019/project1"
project2_path="/D/data/zt/project/colmap/xiongan1019/project2"
project_merge_path="/D/data/zt/project/colmap/xiongan1019/merge"
# sparse_model的位置
project1_model_path="$project1_path/sparse_manual"
project2_model_path="$project2_path/sparse"
# 畸变矫正过的image的位置
project1_image_path="$project1_path/images"
project2_image_path="$project2_path/images"
# temp_model的位置
temp_model="$colmap_gcp_project_path/merge_model_temp"
temp_model_1="$temp_model/project1"
temp_model_2="$temp_model/project2"
temp_model_merge="$temp_model/merge"

############ 将模型复制到一个新的文件夹里面，作为准备 ###########
# 需要注意的是，有可能把bin文件复制过去，也有可能把txt文件复制过去
cd $temp_model
rm -rf *
mkdir $temp_model_1
mkdir $temp_model_2
cp $project1_model_path/* $temp_model_1
cp $project2_model_path/* $temp_model_2
mkdir $temp_model_merge

########### 统一删除txt文件，并且将bin文件转化为txt文件,最终文件夹中只剩txt文件 #########
colmap model_converter --input_path $temp_model_1 --output_path $temp_model_1 --output_type bin
colmap model_converter --input_path $temp_model_2 --output_path $temp_model_2 --output_type bin
rm -rf $temp_model_1/*.txt
rm -rf $temp_model_2/*.txt
colmap model_converter --input_path $temp_model_1 --output_path $temp_model_1 --output_type txt
colmap model_converter --input_path $temp_model_2 --output_path $temp_model_2 --output_type txt
rm -rf $temp_model_1/*.bin
rm -rf $temp_model_2/*.bin
rm -rf $temp_model_1/*.nvm
rm -rf $temp_model_2/*.nvm

########### 转化两个model ##########
cd $colmap_gcp_project_path
python rewrite_image_camera_to_merge_models.py --model1_path $temp_model_1 --model2_path $temp_model_2 --merge_path $temp_model_merge

########### 清除project_merge_path中的内容 #########
rm -rf $project_merge_path/*
mkdir $project_merge_path/images

########### 将model转换成nvm格式 #########
colmap model_converter --input_path $temp_model_merge --output_path $project_merge_path/images/Bundler.nvm --output_type nvm

########### 将两个数据集里的图片放到merge文件夹里面 #########
# 要注意改一下第二个数据集的图片命名，防止跟第一个数据集冲突
cp $project1_image_path/*.jpg $project_merge_path/images
for img in $project2_image_path/*.jpg; do
  filename=$(basename "$img")
  new_name="${filename%.*}_model2.${filename##*.}"
  cp "$img" "$project_merge_path/images/$new_name" 
done
