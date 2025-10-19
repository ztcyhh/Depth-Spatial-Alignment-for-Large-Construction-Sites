# 目的是先用opencv标定矫正过之后，再载入opencv标定的一些参数来做SFM和MVS
#!/bin/bash
reconstruct_project_path="/D/data/zt/project/colmap/camera_keyboard_test/OPENCV_FISHEYE_0.9"
colmap_opencv_project_path="/home/zt/Project/COLMAP_GroundControlPoints-main"
cd "$reconstruct_project_path"

# 1 特征提取（可选择是否载入已知相机内参和只使用一个camera)
# colmap feature_extractor --ImageReader.camera_model OPENCV --database_path db.db --image_path images --ImageReader.single_camera 1 --ImageReader.camera_params "955.3649,952.5471,960.2907,593.5844,4.2824,12.5274,0.0006,-0.0016,0.8745,4.5845,13.9657,4.3598"
colmap feature_extractor --ImageReader.camera_model OPENCV_FISHEYE --database_path db.db --image_path images --ImageReader.single_camera 1 --ImageReader.camera_params "1106.6311047039987,1106.126700196386,957.010951641113,595.9099327764714,-0.10404917042809313,0.030789449389170703,-0.08086034410165835,0.04201025347675814"

# 2 特征匹配
colmap exhaustive_matcher --database_path db.db

# 3 稀疏重建
mkdir sparse
mkdir sparse/model

# 3.1 调整内参（包括焦距和畸变）
colmap mapper --database_path db.db --image_path images --output_path sparse/model

# 3.2 不再调整畸变
# colmap mapper --database_path db.db --image_path images --output_path sparse/model --Mapper.ba_refine_extra_params 0

# 3.3 不再调整焦距(推荐？)
# colmap mapper --database_path db.db --image_path images --output_path sparse/model --Mapper.ba_refine_focal_length 0

# 3.4 不调整内参
# colmap mapper --database_path db.db --image_path images --output_path sparse/model --Mapper.ba_refine_extra_params 0 --Mapper.ba_refine_focal_length 0

# 将sfm的内外参由bin文件转成txt文件,便于后续读取处理
colmap model_converter --input_path sparse/model/0 --output_path sparse/model/0 --output_type TXT

# 图像去畸变
mkdir dense
colmap image_undistorter \
    --image_path ./images \
    --input_path ./sparse/model/0 \
    --min_scale 0.9\
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