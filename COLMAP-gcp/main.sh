#!/bin/bash

python main.py --Imgs /home/zt/Project/COLMAP_GroundControlPoints-main/temp/imgs \
               --ImgExtension .jpg \
               --Projections /home/zt/Project/COLMAP_GroundControlPoints-main/temp/target_projections \
               --ProjectionDelimeter " " \
               --SparseModel /home/zt/Project/COLMAP_GroundControlPoints-main/temp/colmap_sparse \
               --GroundTruth /home/zt/Project/COLMAP_GroundControlPoints-main/temp/Ground_Truth.txt \
               --ColmapExe /usr/local/bin/ \
               --AlignerExe ./AlignCC_for_linux \
               --ScaleFactor 1