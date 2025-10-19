#!/bin/bash
reconstruct_project_path="/D/data/zt/project/colmap/test8"
colmap_gcp_project_path="/home/zt/Project/COLMAP_GroundControlPoints-main"
cd "$reconstruct_project_path"
colmap feature_extractor --ImageReader.camera_model PINHOLE --database_path db.db --image_path images
colmap exhaustive_matcher --database_path db.db
cd "$colmap_gcp_project_path"
python rewrite_intrisic.py --project_folder "$reconstruct_project_path"
cd "$reconstruct_project_path"
colmap mapper --database_path db.db --image_path images --output_path sparse/model --Mapper.ba_refine_extra_params 0 --Mapper.ba_refine_focal_length 0
# 将sfm的内外参由bin文件转成txt文件,便于后续读取处理
colmap model_converter --input_path sparse/model/0 --output_path sparse/model/0 --output_type TXT