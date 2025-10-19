#!/bin/bash
# 目的是只把result.ply挪出来，因为发现是深度图-stero最占空间

# 定义文件夹路径
DIR=/D/data/zt/project/colmap/batch_exp_one_good_result

# 遍历一级子目录
for d in $DIR/*/; do

  # 获取子目录名 
  subdir=${d%/}
  
  # 检查是否有dense文件夹
  if [ -d "$d/dense" ]; then

    mv $d/dense/result.ply $subdir
    rm -r $d/dense

  # 如果没有则跳过    
  else
    continue
  fi
  
done