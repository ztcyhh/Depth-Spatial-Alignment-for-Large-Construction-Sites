#!/bin/bash
reconstruct_project_path="/D/data/zt/project/colmap/xinyang/0418/two_tower"
colmap_gcp_project_path="/home/zt/Project/COLMAP_GroundControlPoints-main"
cd "$reconstruct_project_path"
mkdir sparse
mkdir sparse/model
colmap feature_extractor --ImageReader.camera_model SIMPLE_RADIAL --database_path db.db --image_path images
# colmap feature_extractor --ImageReader.camera_model RADIAL --database_path db.db --image_path images
colmap exhaustive_matcher --database_path db.db
colmap mapper --database_path db.db --image_path images --output_path sparse/model
# 将sfm的内外参由bin文件转成txt文件,便于后续读取处理
colmap model_converter --input_path sparse/model/0 --output_path sparse/model/0 --output_type TXT