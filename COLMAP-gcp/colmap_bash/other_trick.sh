colmap_project_path="/D/data/zt/project/colmap/test_no_gcp"
cd "$colmap_project_path"
input_path="sparse/model/0"
colmap model_converter --input_path "$input_path" --output_path "$input_path" --output_type TXT