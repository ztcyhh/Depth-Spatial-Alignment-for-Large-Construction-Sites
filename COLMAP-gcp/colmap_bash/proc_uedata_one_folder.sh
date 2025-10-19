#!/bin/bash
# 主要用于将unreal采集的数据直接转化成colmap的输入，并且准备好已知的内参和xyz文件
# 当前主要是对单个文件夹进行处理
reconstruct_project_path="/D/data/zt/project/colmap/thesis_position_exp/batch_exp_three_ready_for_sfm/exp_R60_pitch0_H25"
colmap_gcp_project_path="/home/zt/Project/COLMAP_GroundControlPoints-main"
cd "$reconstruct_project_path"
mkdir images
mv *.jpg images
mkdir manual
mkdir manual/model
mkdir sparse
mkdir sparse/model
# 将已知的相机内参放到manual/model目录下面
cp /home/zt/Project/COLMAP_GroundControlPoints-main/cameras/cameras_ue_1280_960_90.txt manual/model/cameras.txt
# 创建已知的相机位姿的txt文件，便于后续使用colmap自带的model_aligner功能
cd "$colmap_gcp_project_path"
python ue2gps.py --project_folder $reconstruct_project_path
