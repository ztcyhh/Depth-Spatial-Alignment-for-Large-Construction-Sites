colmap model_aligner \
    --input_path /D/data/zt/project/colmap/exp_R50_pitch40_angle90/sparse/model \
    --output_path /D/data/zt/project/colmap/exp_R50_pitch40_angle90/sparse_align \
    --ref_images_path /D/data/zt/project/colmap/exp_R50_pitch40_angle90/ref_xyz.txt \
    --ref_is_gps 0 \
    --robust_alignment 1 \
    --robust_alignment_max_error 3.0