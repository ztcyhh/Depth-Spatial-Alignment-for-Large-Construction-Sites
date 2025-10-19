#!/bin/bash
reconstruct_project_path="/D/data/zt/project/colmap/cuptest1"
colmap_gcp_project_path="/home/zt/Project/COLMAP_GroundControlPoints-main"
cd "$reconstruct_project_path"
colmap feature_extractor --ImageReader.camera_model PINHOLE --database_path db.db --image_path images
cd "$colmap_gcp_project_path"
python rewrite_db.py --project_folder "$reconstruct_project_path"
cd "$reconstruct_project_path"
colmap exhaustive_matcher --database_path db.db
cd "$colmap_gcp_project_path"
python rewrite_images.py --project_folder "$reconstruct_project_path"
cd "$reconstruct_project_path"
colmap point_triangulator --database_path db.db --image_path images --input_path manual/model --output_path sparse/model