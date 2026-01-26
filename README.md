# Forest Point Cloud Segmentation & Instance Extraction

This repository contains a pipeline for semantic segmentation, instance extraction, and post-processing of forest point clouds (PLY/LAZ). It supports parallel inference, ablation studies, and utility tools for class extraction. It is destined to help with the re-annotation of the DALES benchmark.

## Core Scripts

### 1. Training & Inference
*   **`scripts/train.py`**: Training loop for the PointNet++ based model.
*   **`scripts/model.py`**: Definition of the `ForestPointNetPP` architecture.
*   **`scripts/pipeline_inference.py`**: Main inference script.
    *   **Features**: Batch processing, TTA (Test-Time Augmentation), parallel execution (sharding), and specific post-processing rules (Z-coordinate filtering, non-tree masking).
    *   **Post-processing**:
        *   Merges predicted instances.
        *   Relabels non-tree points located above tree points based on vertical and distance thresholds.
        *   Assigns instance ID 0 to non-tree points.
*   **`run_inference_parallel.sh`**: Bash script to orchestrate multi-GPU inference.
    *   **Usage**: `./run_inference_parallel.sh -i <input_dir> -o <output_dir> -c <checkpoint>`
    *   **Features**: Automatically detects GPUs, launches multiple workers per GPU, and handles `Ctrl+C` for clean termination.

### 2. Utilities
*   **`run_dump_ground_and_fence.py`**: Extracts 'Ground' (class 1) and 'Fence' (class 6) points from PLY files and converts them to LAZ.
    *   **Output**: Preserves original attributes and adds `new_sem_class` (copy) and `new_ins_class` (set to -1).
    *   **Usage**: `python run_dump_ground_and_fence.py --input_dir <path> --output_dir <path>`
*   **`scripts/ablation.py`**: Configuration and logic for running ablation studies on model components.
*   **`run_ablation_parallel.sh`**: Parallel runner for ablation experiments across multiple GPUs.

## Multi-Dataset Support & Configuration

This repository now supports training on multiple datasets (Forestry, Tree, Truck, Building, Pole) via configuration files located in `configs/`.

### Supported Datasets & Classes
*   **Forestry** (Default): Tree vs Undergrowth.
*   **Tree**: Broadleaf vs Conifer.
*   **Truck**: Pick-up, Van, Truck, Construction Vehicle.
*   **Building**: House, Flat, Complex.
*   **Pole**: Light Pole, Utility Pole, Traffic Pole.

### Usage

**1. Preprocessing:**
Use `scripts/preprocess.py` with a config file to process LAZ files into training data.
```bash
python scripts/preprocess.py --config configs/truck.json
```

**2. Training:**
Use `scripts/train.py` with the same config file to launch training with the correct parameters.
```bash
python scripts/train.py --config configs/truck.json
```

**Configuration Files:**
Config files (JSON) specify:
*   `dataset_name`: Name of the dataset (used for checkpoints).
*   `data_dir`: Input directory containing LAZ files.
*   `output_dir`: Output directory for processed .pt files.
*   `num_points`: Number of points to sample.
*   `task_type`: `binary_one_vs_rest` or `multiclass`.
*   `class_mapping`: Mapping from LAZ class IDs to training labels (0, 1, ...).

## Installation

Dependencies include:
*   `torch`, `torch_geometric`
*   `laspy` (for LAZ I/O)
*   `plyfile` (for PLY I/O)
*   `scipy`, `sklearn`, `numpy`, `tqdm`

## Usage Examples

**Run Parallel Inference:**
```bash
# Run on all available GPUs with default batch size
./run_inference_parallel.sh -i ./data/test -o ./data/preds -c ./checkpoints/best_model.pt --batch_size 16
```

**Extract Ground & Fence:**
```bash
python run_dump_ground_and_fence.py --input_dir ./data/raw_ply --output_dir ./data/ground_fence_laz
```
