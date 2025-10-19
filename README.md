# Depth-Spatial-Alignment for Large Construction Sites

## Project Overview

This project is a comprehensive 3D reconstruction and spatial alignment system designed for large construction sites. It addresses the challenges of precise 3D reconstruction and multi-model alignment in complex construction environments. The system integrates deep learning object detection, Ground Control Point (GCP) based 3D reconstruction, and edge-aware point cloud registration technologies.

## Main Functional Modules

### 1. HA-YOLOv8 - GCP Detection Module
- **Function**: Uses an improved YOLOv8 model to detect Ground Control Points (GCPs)
- **Features**: Integrates HAB (Hierarchical Attention Block) module, specifically optimized for long-distance GCP detection
- **Input**: Construction site images
- **Output**: GCP pixel coordinates and segmentation masks
- **Run Command**: `python predict_draw_mask_and_center.py`

### 2. COLMAP-GCP - Real-Scale 3D Reconstruction Based on GCPs
- **Function**: Utilizes detected GCPs for precise 3D reconstruction and scale recovery
- **Core Workflow**:
  - Basic SfM reconstruction
  - GCP point matching and triangulation
  - Camera parameter optimization
  - Real-scale recovery
  - Dense point cloud reconstruction
- **Run Command**: `bash colmap_bash/colmap_gcp_full_pipeline_with_yolo.sh`

### 3. Edge-Aware-Registration - Edge-Aware Point Cloud Registration
- **Function**: Edge detection based on depth projection and 3D bounding box classification
- **Core Algorithms**:
  - Edge feature extraction from depth maps
  - 3D bounding box generation and classification
  - Precise alignment between multiple models
- **Run Commands**: 
  1. `python get_edge_from_depth_projection.py`
  2. `python classify_bbox_by_3d.py`

## System Workflow

1. **Image Acquisition**: Capture multi-view images at construction sites
2. **GCP Detection**: Use HA-YOLOv8 to detect ground control points in images
3. **Basic Reconstruction**: Perform initial sparse 3D reconstruction using COLMAP
4. **GCP Triangulation**: Precise triangulation positioning based on detected GCPs
5. **Scale Recovery**: Restore real scale of reconstruction models using real GCP coordinates
6. **Dense Reconstruction**: Generate high-precision dense point cloud models
7. **Edge Registration**: Use edge-aware algorithms for precise alignment between multiple models

## Technical Features

- **High Precision**: Real-scale recovery based on actual GCPs ensures accurate reconstruction results
- **Robustness**: Specialized optimization for complex construction environments
- **Automation**: Complete end-to-end automated workflow
- **Scalability**: Modular design for easy feature extension and customization

## Application Scenarios

- 3D digitization of large construction sites
- Construction progress monitoring and quality inspection
- Precise alignment between BIM models and on-site reality
- Construction surveying and mapping applications

## System Requirements

- Python 3.8+
- COLMAP
- OpenCV
- Open3D
- PyTorch
- Ultralytics YOLO

## Quick Start

1. Prepare construction site images and GCP real coordinate files
2. Run GCP detection: `python predict_draw_mask_and_center.py`
3. Execute complete reconstruction pipeline: `bash colmap_bash/colmap_gcp_full_pipeline_with_yolo.sh`
4. Perform edge-aware registration: Run the two registration scripts sequentially

## Project Structure

```
Depth-Spatial-Alignment-for-Large-Construction-Sites/
├── HA-YOLOv8/                    # GCP detection module
├── COLMAP-gcp/                   # GCP-based 3D reconstruction
├── edge-aware-registration/      # Edge-aware registration
└── README.md                     # Project documentation
```

