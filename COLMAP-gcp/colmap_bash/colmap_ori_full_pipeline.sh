#!/bin/bash
# colmap完整pipeline的命令行实现
reconstruct_project_path="/D/data/zt/project/colmap/sanju/1212"
cd "$reconstruct_project_path"

# 特征提取
colmap feature_extractor --ImageReader.camera_model SIMPLE_RADIAL --database_path db.db --image_path images

# 特征匹配
colmap exhaustive_matcher --database_path db.db

# 稀疏重建
mkdir sparse
mkdir sparse/model
colmap mapper --database_path db.db --image_path images --output_path sparse/model

# 将sfm的内外参由bin文件转成txt文件,便于后续读取处理
colmap model_converter --input_path sparse/model/0 --output_path sparse/model/0 --output_type TXT

# 图像去畸变
mkdir dense
colmap image_undistorter \
    --image_path ./images \
    --input_path ./sparse/model/0 \
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

# ########################################
# ################ 9 mesh步骤 ############
# ########################################
# colmap poisson_mesher \
#     --input_path ./dense/result.ply \
#     --output_path ./dense/meshed-poisson.ply\
#     --PoissonMeshing.depth 14

########################################
##### 10 处理多余数据 避免多余空间占用 ####
########################################
rm -r dense/stereo/
rm -r dense/*.sh
rm -r dense/*.vis
mv *.ply dense

