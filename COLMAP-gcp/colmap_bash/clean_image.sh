#!/bin/bash

# 定义图片文件夹
image_dir="/D/data/zt/project/colmap/xiongan1111/no1/images"

# 获取图片列表
images=$(ls $image_dir)

# 计数器
count=1

# 遍历图片    
for image in $images; do
  # 如果计数器除以2余1,则删除该图片    
  if [ $((count % 2)) -eq 1 ]; then
    rm "$image_dir/$image"
  fi
  
  # 计数器自增1
  let count=count+1 
done