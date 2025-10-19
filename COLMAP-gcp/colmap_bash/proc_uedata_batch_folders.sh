#!/bin/bash
# 主要用于将unreal采集的数据直接转化成colmap的输入，并且准备好已知的内参和xyz文件
# 当前主要是对一个大文件夹下的所有小文件夹进行处理
batch_reconstruct_project_path="/D/data/zt/project/colmap/batch_exp_two"
colmap_gcp_project_path="/home/zt/Project/COLMAP_GroundControlPoints-main"
# 对单个文件夹进行数据操作的函数
process_subfolder() {
    local current_folder="$1"
    reconstruct_project_path="$1"
    echo "正在处理数据集: $reconstruct_project_path"
    # 创建一些简单的文件夹,符合colmap格式
    cd "$reconstruct_project_path"
    mkdir images
    mv *.jpg images
    mkdir manual
    mkdir manual/model
    mkdir sparse
    mkdir sparse/model
    # 将已知的相机内参放到manual/model目录下面
    cp /home/zt/Project/COLMAP_GroundControlPoints-main/cameras/cameras_ue_1280_960_90.txt manual/model/cameras.txt
    # 创建已知的相机位姿的txt文件,便于后续使用colmap自带的model_aligner功能
    cd "$colmap_gcp_project_path"
    python ue2gps.py --project_folder $reconstruct_project_path
}

# 使用find命令找到下一层的小文件夹，并执行process_subfolder
find "$batch_reconstruct_project_path" -maxdepth 1 -mindepth 1 -type d -print0 | while IFS= read -r -d '' subfolder; do
    process_subfolder "$subfolder"
done

