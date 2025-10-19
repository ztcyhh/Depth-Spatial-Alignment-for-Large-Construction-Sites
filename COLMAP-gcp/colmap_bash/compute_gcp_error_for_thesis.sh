#!/bin/bash
# 基于GCP的colmap完整pipeline的命令行实现
reconstruct_project_path="/data/zt/project/colmap/xinyang/0418/tower4/simple_radial"
colmap_gcp_project_path="/home/zt/Project/COLMAP_GroundControlPoints-main"

# #######################################
# ######## 整理数据，准备gcp三角化 ######
# #######################################
# # 在对齐gcp的项目中创建一个temp文件夹（以后都可以放到这个文件夹里）
# cd "$colmap_gcp_project_path"
# if [ -d "temp" ]; then
#   rm -rf temp
# fi
# mkdir temp

# # 将images文件夹复制到imgs下面
# cp -r "$reconstruct_project_path/images" "$colmap_gcp_project_path/temp/imgs"

# # 将txt_output文件夹复制到target_projections
# cp -r "$reconstruct_project_path/txt_output" "$colmap_gcp_project_path/temp/target_projections"

# # 创建colmap_sparse文件夹，将sparse model复制到下面
# mkdir temp/colmap_sparse
# cp "$reconstruct_project_path/sparse/model/0/"*.txt "$colmap_gcp_project_path/temp/colmap_sparse"

# # 将Ground_Truth.txt文件移动到Ground_Truth.txt文件
# cp "$reconstruct_project_path/Ground_Truth.txt" "$colmap_gcp_project_path/temp/Ground_Truth.txt"

# #######################################
# ######## 5 做gcp三角化和求变换矩阵 ######
# #######################################
# cd "$colmap_gcp_project_path"

# # 删除空的txt文件
# for file in "/home/zt/Project/COLMAP_GroundControlPoints-main/temp/target_projections/"*.txt; do
#     # 检查文件是否为空
#     if [ ! -s "$file" ]; then
#         # 文件为空，删除文件
#         echo "Deleting empty file: $file"
#         rm "$file"
#     fi
# done

#######################################
######## 6 开始for循环计算误差 ######
#######################################
# --- 配置 ---
# 假设 colmap_gcp_project_path 已经设置
# colmap_gcp_project_path="/home/zt/Project/COLMAP_GroundControlPoints-main"
if [ -z "$colmap_gcp_project_path" ]; then
    echo "错误: 环境变量 colmap_gcp_project_path 未设置。"
    exit 1
fi

# 辅助脚本路径
combination_script="$colmap_gcp_project_path/compute_gcp_projection_error_random_combination.py"
if [ ! -f "$combination_script" ]; then
    echo "错误: 组合脚本未找到: $combination_script"
    echo "请确保您已创建新版本的 compute_gcp_projection_error_random_combination.py。"
    exit 1
fi

# 文件路径定义
original_file="$colmap_gcp_project_path/temp/Ground_Truth.txt"
random_file="$colmap_gcp_project_path/temp/Ground_Truth_random.txt"
rest_file="$colmap_gcp_project_path/temp/Ground_Truth_rest.txt"


# --- 准备工作 ---
# 1. 获取 Ground_Truth.txt 的总行数 (N)
#    使用 < 重定向，wc -l 只输出数字
N=$(wc -l < "$original_file")

if [ "$N" -lt 5 ]; then
    echo "错误: Ground_Truth.txt 至少需要 5 行才能运行 k=5 (当前行数: $N)"
    exit 1
fi
echo "总GCP行数 (N) = $N"


# 修正1：Bash 变量赋值不能有空格
# --- 主循环 (外层: 循环 k) ---
for k in 3 4 5
do
    echo "====================================================="
    echo "         开始处理 k=$k (训练集大小)                "
    echo "====================================================="
    echo "--- 详尽验证 (k=$k) ---" >> "$all_errors_file"
    
    i=0 # 重置每个k的组合计数器

    # --- 主循环 (内层: 循环 C(N, k) 组合) ---
    # 1. 调用Python脚本生成 C(N, k) 个组合
    #    输入: $N (总行数), $k (训练行数)
    #    输出格式: "1,3,4;2,5,6" (行号)
    python3 "$combination_script" "$N" "$k" | while IFS=';' read -r train_line_nums rest_line_nums
    do
        i=$((i+1))
        echo "-----------------------------------------------------"
        echo "k=$k, 组合 $i: 训练行 = { $train_line_nums }, 测试行 = { $rest_line_nums }"
        echo "-----------------------------------------------------"

        # 2. 修正：使用 AWK 按行号选择
        #    'NR==FNR {lines[$1]=1; next} FNR in lines' 是一个标准模式
        #    它从标准输入(-)读取行号，然后在 $original_file 中打印匹配的行
        
        # 创建训练文件
        echo "$train_line_nums" | tr ',' '\n' | \
            awk 'NR==FNR {lines[$1]=1; next} FNR in lines' - "$original_file" > "$random_file"
        
        # 创建测试文件
        echo "$rest_line_nums" | tr ',' '\n' | \
            awk 'NR==FNR {lines[$1]=1; next} FNR in lines' - "$original_file" > "$rest_file"

        echo "已创建 $random_file 和 $rest_file。"

        # 3. 运行您的主程序 (计算变换 T)
        #    (使用 $random_file 作为 --GroundTruth)
        cd "$colmap_gcp_project_path"

        python main.py --Imgs /home/zt/Project/COLMAP_GroundControlPoints-main/temp/imgs \
               --ImgExtension .jpg \
               --Projections /home/zt/Project/COLMAP_GroundControlPoints-main/temp/target_projections \
               --ProjectionDelimeter " " \
               --SparseModel /home/zt/Project/COLMAP_GroundControlPoints-main/temp/colmap_sparse \
               --GroundTruth /home/zt/Project/COLMAP_GroundControlPoints-main/temp/Ground_Truth_random.txt \
               --ColmapExe /usr/local/bin/ \
               --AlignerExe ./AlignCC_for_linux \
               --ScaleFactor 1
            
        echo "main.py 完成。正在计算验证误差..."
        python /home/zt/Project/COLMAP_GroundControlPoints-main/compute_gcp_projection_error.py
        echo "验证误差完成。"

    done
done

echo "====================================================="
echo "所有 k 值的组合均已处理完毕。"
echo "====================================================="

