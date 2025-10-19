#!/bin/bash
set -e  # 保证一旦错误就跳出
#########################################################################
####################### TODO ############################################
# 1.由于正常采集下，使用simple radial进行模型拟合，因此，图像undistort之后会变形。因此对于depth应当应用同样的变形
# 2.应当用depth>0的部分生成mask，从而继续生成00000000_masked.jpg
# 3.应当对所有图片进行resize和crop，统一尺寸（倒不必32整除，因为dataloader的时候会再resize）

# 能够直接将unrealcv采集的文件夹，转化成符合mvsnet的输入（包括图片校正和cams生成）
# 需要输入两个位置，（1）原始的ue的大文件夹，（2）处理后保存的mvsnet的数据大文件夹
################# 原始格式 ###############
# -scan1_dir
#     -images
#         -1.jpg
#         -2.jpg
#         -···
#         -n.jpg
#         -ue_pos.txt
#     -depth_maps
#         -1.pfm
#         -2.pfm
#         -···
#         -n.pfm
#     -masked_img
#         -1_masked.jpg
#         -2_masked.jpg
#         -···
#         -n_masked.jpg
#     -mask
#         -1.jpg
#         -2.jpg
#         -···
#         -n.jpg
################# 目标格式 ###############
# 同blendedmvs即可
# -scan1_dir
#     -blended_images
#         -00000000.jpg
#         -00000000_masked.jpg
#         -00000001.jpg
#         -00000001_masked.jpg
#         ···
#         -0000000n.jpg
#         -0000000n_masked.jpg
#     -cams
#         -00000000_cam.txt
#         -00000001_cam.txt
#         ···
#         -0000000n_cam.txt
#         -pair.txt
#     -rendered_depth_maps
#         -00000000.pfm
#         -00000001.pfm
#         ···
#         -0000000n.pfm

# 需输入
ori_root_dir="/D/data/zt/3d/ori_UE_Train_data/SCAN3_240325"  # 原始的ue文件夹，里面有n个小文件夹
save_root_dir="/D/data/zt/3d/MVSNet_TRAINING/MVSNet_TRAINING_scan3_240325"  # 处理后的mvsnet文件夹，里面有n个小文件夹

################ 对每个subdir的处理代码 ############
process_subdir() {
    local subdir="$1"
    
    # 获取子目录名
    subdir_name="${subdir##*/}"

    ori_uedata_path="$ori_root_dir/$subdir_name"
    save_mvsnet_data_path="$save_root_dir/$subdir_name"
    
    echo "Processing subdir: $ori_uedata_path,save in $save_mvsnet_data_path"
    ##########################################
    ################ 步骤0 ###################
    ##########################################
    # 准备基础文件夹位置
    # 无需输入
    tmp_uedata_path="$ori_uedata_path/../tmp"
    reconstruct_project_path="$tmp_uedata_path/colmap_sfm"
    MVSNet_format_dataset_tmp_path="$tmp_uedata_path/mvsnet_tmp"
    # 代码项目位置
    colmap_gcp_project_path="/home/zt/Project/COLMAP_GroundControlPoints-main"
    MVSNet_project_path="/home/zt/Project/MVS/CasMVSNet"
    # 简单创建一些新文件夹
    if [ -d "$tmp_uedata_path" ]; then
      rm -rf "$tmp_uedata_path"
    fi
    cp -r "$ori_uedata_path" "$tmp_uedata_path"  #将uedata文件夹复制到一个tmp文件夹中处理

    if [ -d "$save_mvsnet_data_path" ]; then
      rm -rf "$save_mvsnet_data_path"
    fi
    mkdir "$save_mvsnet_data_path"  # 创建save的文件夹

    ##########################################
    ################ 步骤1 ###################
    ##########################################
    # 主要用于将unreal采集的数据直接转化成colmap的输入，并且准备好已知的内参和xyz文件
    # 当前主要是对单个文件夹进行处理
    ### 1.1 简单修改路径
    mkdir "$reconstruct_project_path"
    cd "$reconstruct_project_path"
    mkdir images
    cp ../images/*.jpg ./images
    mv ../images/*.txt .
    mv ../images ../ori_images
    mkdir manual
    mkdir manual/model
    mkdir sparse
    mkdir sparse/model

    ### 1.2 将已知的相机内参放到manual/model目录下面
    cp /home/zt/Project/COLMAP_GroundControlPoints-main/cameras/cameras_ue_1280_960_90.txt manual/model/cameras.txt

    ### 1.3 创建已知的相机位姿的txt文件，便于后续使用colmap自带的model_aligner功能
    cd "$colmap_gcp_project_path"
    python ue2gps.py --project_folder $reconstruct_project_path

    ##########################################
    ################ 步骤2 ###################
    ##########################################
    # 主要希望能够输出一个dense文件夹即可
    # 基于gps的colmap三维重建
    ### 2.1 初始重建
    cd "$reconstruct_project_path"
    colmap feature_extractor --ImageReader.camera_model PINHOLE --database_path db.db --image_path images
    # colmap feature_extractor --ImageReader.camera_model PINHOLE --database_path db.db --image_path images --ImageReader.single_camera 1 --ImageReader.camera_params "1280,960,640.0,640.0,640.0,480.0"
    cd "$colmap_gcp_project_path"
    python ue2colmap.py --project_folder "$reconstruct_project_path"
    python rewrite_images.py --project_folder "$reconstruct_project_path"
    cd "$reconstruct_project_path"
    colmap exhaustive_matcher --database_path db.db
    colmap point_triangulator --database_path db.db --image_path images --input_path manual/model --output_path sparse/model

    ### 2.2 坐标对齐（当前不得已的解决方案，model_aligner)
    cd "$reconstruct_project_path"
    mkdir sparse_align
    colmap model_aligner \
        --input_path "$reconstruct_project_path/sparse/model" \
        --output_path "$reconstruct_project_path/sparse_align" \
        --ref_images_path "$reconstruct_project_path/ref_xyz.txt" \
        --ref_is_gps 0 \
        --robust_alignment 1 \
        --robust_alignment_max_error 3.0

    ### 2.3 将sfm的内外参由bin文件转成txt文件
    colmap model_converter --input_path "$reconstruct_project_path/sparse_align" --output_path "$reconstruct_project_path/sparse_align" --output_type TXT

    ## 2.4 图像去畸变
    mkdir dense
    colmap image_undistorter \
        --image_path ./images \
        --input_path ./sparse_align \
        --output_path ./dense \
        --output_type COLMAP

    ##########################################
    ################ 步骤3 ###################
    ##########################################
    # 将colmap的sparse转成mvsnet的cams文件
    # 输出：
    # -cams
    #   -000000001.txt
    #   -000000002.txt
    #   ···
    #   -00000000n.txt
    # -image_post
    #   -000000001.jpg
    #   -000000002.jpg
    #   ···
    #   -00000000n.jpg
    # -pair.txt
    cd "$MVSNet_project_path"
    mkdir "$MVSNet_format_dataset_tmp_path"  # 作为临时存储colmap2mvsnet的output的位置
    python colmap2mvsnet.py --dense_folder "$reconstruct_project_path/dense" --save_folder "$MVSNet_format_dataset_tmp_path" --max_d 256

    ##########################################
    ################ 步骤4 ###################
    ##########################################
    # 最后调整一下文件夹格式，直接从tmp文件夹向save文件夹中移动即可

    ### 4.1 创建save文件夹中的子文件夹
    mkdir "$save_mvsnet_data_path/blended_images"
    mkdir "$save_mvsnet_data_path/cams"
    mkdir "$save_mvsnet_data_path/rendered_depth_maps"

    ### 4.2 移动图片
    # 移动masked图片 (TODO：事实上应该是根据depth重新生成masked图片)
    cp "$reconstruct_project_path/masked_img"/*.jpg "$save_mvsnet_data_path/blended_images"
    # 移动原始图片
    cp "$MVSNet_format_dataset_tmp_path/images_post"/*.jpg "$save_mvsnet_data_path/blended_images"

    ### 4.2 移动cams
    # 移动cams.txt
    cp "$MVSNet_format_dataset_tmp_path/cams"/*.txt "$save_mvsnet_data_path/cams"
    # 移动pair.txt
    cp "$MVSNet_format_dataset_tmp_path/pair.txt" "$save_mvsnet_data_path/cams"

    ### 4.3 移动depth (TODO:事实上应该是根据undistort的内容重新生成depth)
    cp "$tmp_uedata_path/depth_maps"/*.pfm "$save_mvsnet_data_path/rendered_depth_maps"

    ### 4.4 最后删除tmp文件夹
    rm -r "$tmp_uedata_path"
}

############## 批量处理 #############
# 遍历一级子目录
traverse() {
    local dir="$1"
    for entry in "$dir"/*
    do
        if [ -d "$entry" ]; then
            process_subdir "$entry"
        fi
    done
}

# 调用递归函数开始遍历
traverse "$ori_root_dir"


