#!/bin/bash
# 目的是批量化的进行稀疏重建，并用到已知的相机内参
batch_reconstruct_project_path="/D/data/zt/project/colmap/thesis_position_exp/batch_exp_three_sfm"
colmap_gcp_project_path="/home/zt/Project/COLMAP_GroundControlPoints-main"
txt_path="/home/zt/Project/COLMAP_GroundControlPoints-main/time_sfm.txt"

# 对单个文件夹进行数据操作的函数
process_subfolder() {
    local current_folder="$1"
    reconstruct_project_path="$1"
    echo "正在处理数据集: $reconstruct_project_path"
    cd "$reconstruct_project_path"
    # 特征提取+匹配
    colmap feature_extractor --ImageReader.camera_model PINHOLE --database_path db.db --image_path images
    colmap exhaustive_matcher --database_path db.db
    cd "$colmap_gcp_project_path"
    # 把camera内参写入
    python rewrite_intrisic.py --project_folder "$reconstruct_project_path"
    cd "$reconstruct_project_path"
    # 不更新camera内参的sfm
    colmap mapper --database_path db.db --image_path images --output_path sparse/model --Mapper.ba_refine_extra_params 0 --Mapper.ba_refine_focal_length 0
    # colmap mapper --database_path db.db --image_path images --output_path sparse/model
    # model align 对齐到真实世界坐标系
    mkdir sparse_align
    colmap model_aligner \
    --input_path sparse/model/0 \
    --output_path sparse_align \
    --ref_images_path ref_xyz.txt \
    --ref_is_gps 0 \
    --robust_alignment 1\
    --robust_alignment_max_error 3\

}

# 使用find命令找到下一层的小文件夹，并执行process_subfolder
echo "project seconds" > $txt_path
find "$batch_reconstruct_project_path" -maxdepth 1 -mindepth 1 -type d -print0 | while IFS= read -r -d '' subfolder; do
    # start_time=$(date +%s) # 开始的时刻
  
    process_subfolder "$subfolder"
    
    # end_time=$(date +%s)  # 结束的时刻

    # elapsed=$((end_time - start_time))  # 总共用时
    
    # echo "$subfolder $elapsed" >> $txt_path
    
done
