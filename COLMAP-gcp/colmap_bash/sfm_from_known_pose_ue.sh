#!/bin/bash
# 注意这里的rewrite_images.py 和 colmap feature_extractor --ImageReader.camera_model PINHOLE --database_path db.db --image_path images --ImageReader.single_camera 1 --ImageReader.camera_params "1280,960,640.0,640.0,640.0,480.0"
reconstruct_project_path="/D/data/zt/project/colmap/exp_R50_pitch40_angle360_45image"
# reconstruct_project_beifen_path="/D/data/zt/project/colmap/exp_R50_pitch40_angle90_beifen"
colmap_gcp_project_path="/home/zt/Project/COLMAP_GroundControlPoints-main"

# ######### 为了测试，需要cp一下 ##########
# #######################################
# rm -r "$reconstruct_project_path"
# cp -r "$reconstruct_project_beifen_path" "$reconstruct_project_path"

######### 后续运行代码 #################
cd "$reconstruct_project_path"
colmap feature_extractor --ImageReader.camera_model PINHOLE --database_path db.db --image_path images
# colmap feature_extractor --ImageReader.camera_model PINHOLE --database_path db.db --image_path images --ImageReader.single_camera 1 --ImageReader.camera_params "1280,960,640.0,640.0,640.0,480.0"
cd "$colmap_gcp_project_path"
python ue2colmap.py --project_folder "$reconstruct_project_path"
python rewrite_images.py --project_folder "$reconstruct_project_path"
cd "$reconstruct_project_path"
colmap exhaustive_matcher --database_path db.db
colmap point_triangulator --database_path db.db --image_path images --input_path manual/model --output_path sparse/model

######### 当前的ue2colmap还是有点问题，还是得换成blender再说 ########
######## 退而求其次，用 model_align来对齐一下 ######################
cd "$reconstruct_project_path"
mkdir sparse_align
colmap model_aligner \
    --input_path "$reconstruct_project_path/sparse/model" \
    --output_path "$reconstruct_project_path/sparse_align" \
    --ref_images_path "$reconstruct_project_path/ref_xyz.txt" \
    --ref_is_gps 0 \
    --robust_alignment 1 \
    --robust_alignment_max_error 3.0

# 将sfm的内外参由bin文件转成txt文件,便于后续读取处理
colmap model_converter --input_path "$reconstruct_project_path/sparse_align" --output_path "$reconstruct_project_path/sparse_align" --output_type TXT
