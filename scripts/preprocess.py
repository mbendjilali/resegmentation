import os
import glob
import numpy as np
import laspy
import torch
from tqdm import tqdm
import argparse
import json


def normalize_pc(points):
    """
    Center the cloud at (0,0,0) and scale to fit within a unit sphere.
    """
    centroid = np.mean(points, axis=0)
    points -= centroid
    max_dist = np.max(np.sqrt(np.sum(points**2, axis=1)))
    points /= max_dist
    return points

def process_file(file_path, output_path, config):
    try:
        # 1. Read LAZ file
        las = laspy.read(file_path)
        
        # 2. Extract Coordinates (N, 3)
        coords = np.vstack((las.x, las.y, las.z)).T.astype(np.float32)
        
        # Extract Intensity
        intensity = las.intensity.astype(np.float32)
        intensity /= 65535.0 
        
        # 3. Extract SemClass and Labels based on Config
        raw_classes = las.sub_class
        
        task_type = config.get("task_type", "binary_one_vs_rest")
        
        if task_type == "multiclass":
            class_map = config.get("class_mapping", {})
            # Ensure keys are ints
            class_map = {int(k): int(v) for k, v in class_map.items()}
            
            # Filter points to only include those in class_map
            valid_mask = np.isin(raw_classes, list(class_map.keys()))
            
            if np.sum(valid_mask) == 0:
                # print(f"Skipping {file_path}: No valid classes found.")
                return False
                
            coords = coords[valid_mask]
            raw_classes = raw_classes[valid_mask]
            intensity = intensity[valid_mask]
            
            # Map labels
            labels = np.zeros_like(raw_classes, dtype=np.int64)
            for k, v in class_map.items():
                labels[raw_classes == k] = v
                
        else:
            # Binary One vs Rest (Default Forestry behavior)
            target_id = config.get("target_class_id", 9)
            labels = (raw_classes == target_id).astype(np.int64)
            # We keep all points in this mode, just label them 0 or 1
            
        # --- FEATURE ENGINEERING: Z-NORM ---
        # Compute on the (potentially filtered) points
        z_col = coords[:, 2]
        z_min = np.min(z_col)
        z_max = np.max(z_col)
        z_range = z_max - z_min
        if z_range == 0: z_range = 1.0 
        z_norm = (z_col - z_min) / z_range

        # 5. Handle Variable Point Counts
        N = coords.shape[0]
        num_points = config.get("num_points", 2048)
        
        if N >= num_points:
            choice_idx = np.random.choice(N, num_points, replace=False)
        else:
            choice_idx = np.random.choice(N, num_points, replace=True)
            
        sampled_coords = coords[choice_idx]
        sampled_labels = labels[choice_idx]
        sampled_intensity = intensity[choice_idx]
        sampled_z_norm = z_norm[choice_idx]
        
        # 6. Normalize Geometry
        normalized_coords = normalize_pc(sampled_coords)
        
        # 7. Create Features Vector
        # Structure: [intensity, z_norm] -> Shape (N, 2)
        features = np.column_stack((
            sampled_intensity, 
            sampled_z_norm, 
        ))
        
        # 8. Save
        torch_data = {
            'pos': torch.from_numpy(normalized_coords), 
            'x': torch.from_numpy(features),            
            'y': torch.from_numpy(sampled_labels)       
        }
        torch.save(torch_data, output_path)
        return True
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Preprocess Point Cloud Data")
    parser.add_argument("--config", type=str, help="Path to configuration JSON file")
    args = parser.parse_args()

    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
            
    data_dir = config.get("data_dir")
    output_dir = config.get("output_dir")
    
    print(f"Using Config: {config}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    split = ["train", "val", "test"]
    for s in split:
        if os.path.exists(os.path.join(data_dir, s)):
            print(f"Processing {s} split...")
            target_split_dir = os.path.join(output_dir, s)
            if not os.path.exists(target_split_dir):
                os.makedirs(target_split_dir)

            files = glob.glob(os.path.join(data_dir, s, "*.laz"))
            print(f"Found {len(files)} LAZ files in {os.path.join(data_dir, s)}")
            
            count = 0
            for f in tqdm(files):
                filename = os.path.basename(f).replace('.laz', '.pt')
                out_path = os.path.join(target_split_dir, filename)
                if process_file(f, out_path, config):
                    count += 1
                    
            print(f"Successfully processed {count} files.")
        else:
            print(f"Split directory not found: {os.path.join(data_dir, s)}")

if __name__ == "__main__":
    main()