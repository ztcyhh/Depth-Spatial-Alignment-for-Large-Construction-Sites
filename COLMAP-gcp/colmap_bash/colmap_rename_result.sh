#!/bin/bash
# 目的是把result.ply改名字，然后把改完名字的result.ply统一存放在一个目录下面，便于统一计算误差

# 定义文件夹路径
DIR=/D/data/zt/project/colmap/batch_exp_one_has_mvs_result_left
PCD_SAVE_DIR=/D/data/zt/project/colmap/batch_exp_one_dense_pcd_left

for d in $DIR/*/; do

  subdir=${d%/}
  subdir_name=$(basename $subdir)

  if [ -f $subdir/result.ply ]; then
    # 如果存在result.ply，就把result.ply重命名一下子，并且统一存放起来
    
    mv $subdir/result.ply $subdir/$subdir_name\_result.ply
    cp $subdir/$subdir_name\_result.ply $PCD_SAVE_DIR/$subdir_name\_result.ply

  elif [ -f $subdir/$subdir_name\_result.ply ]; then
    # 如果存在重命名后的result.ply，就直接统一存放
    echo $subdir/$subdir_name\_result.ply
    cp $subdir/$subdir_name\_result.ply $PCD_SAVE_DIR/$subdir_name\_result.ply
    
  fi
done