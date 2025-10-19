#!/bin/bash
# 在做完colmap的sfm步骤之后（有model输出），再执行后续的mvs步骤
# 但当前只对于单个文件夹（单个图像集）
reconstruct_project_path="/D/data/zt/project/colmap/xinyang/0418/tower4/simple_radial"
cd "$reconstruct_project_path"

# # 图像去畸变  
# # 注意如果是用原始的sfm结果应该用sparse/model/0  如果是用align之后的sfm结果，应该用sparse_align/model
# mkdir dense
# # colmap image_undistorter \
# #     --image_path ./images \
# #     --input_path ./sparse/model/0 \
# #     --output_path ./dense \
# #     --output_type COLMAP 

# colmap image_undistorter \
#     --image_path ./images \
#     --input_path ./sparse/real_model \
#     --output_path ./dense \
#     --output_type COLMAP 

# # 稠密重建
# colmap patch_match_stereo \
#     --workspace_path ./dense \
#     --workspace_format COLMAP \
#     --PatchMatchStereo.geom_consistency true

# # 深度图融合
# colmap stereo_fusion \
#     --workspace_path ./dense \
#     --workspace_format COLMAP \
#     --input_type geometric \
#     --output_path ./dense/result.ply

######################################
############### 9 mesh步骤 ############
######################################
colmap poisson_mesher \
    --input_path ./dense/tower4_region.ply \
    --output_path ./dense/tower4_mesh_region_no_denoise.ply



