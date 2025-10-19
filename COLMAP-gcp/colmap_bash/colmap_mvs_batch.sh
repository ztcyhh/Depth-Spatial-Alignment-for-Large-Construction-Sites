#!/bin/bash
# 在做完colmap的sfm步骤之后（有model输出），再执行后续的mvs步骤
# 对于一个大文件夹下的所有小文件夹都分别做mvs

batch_reconstruct_project_path="/D/data/zt/project/colmap/thesis_position_exp/batch_exp_three_mvs"

# 对单个文件夹进行数据操作的函数
process_subfolder() {
    local current_folder="$1"
    reconstruct_project_path="$1"
    echo "正在处理数据集: $reconstruct_project_path"
    cd "$reconstruct_project_path"
    # 图像去畸变  
    # 注意如果是用原始的sfm结果应该用sparse/model/0  如果是用align之后的sfm结果，应该用sparse_align
    mkdir dense
    colmap image_undistorter \
        --image_path ./images \
        --input_path ./sparse_align \
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
    
    # 由于深度图占用内存太大，我们其实只关心result.ply，所以将result.ply移出来
    mv ./dense/result.ply .
    rm -r ./dense
}

# # 使用find命令找到下一层的小文件夹，并执行process_subfolder
# echo "project seconds" > time_mvs.txt

find "$batch_reconstruct_project_path" -maxdepth 1 -mindepth 1 -type d -print0 | while IFS= read -r -d '' subfolder; do
    # start_time=$(date +%s) # 开始的时刻
  
    process_subfolder "$subfolder"
    
    # end_time=$(date +%s)  # 结束的时刻

    # elapsed=$((end_time - start_time))  # 总共用时
    
    # echo "$subfolder $elapsed" >> time_mvs.txt

done

