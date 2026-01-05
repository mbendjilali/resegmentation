import os
import glob
import numpy as np
import laspy
import torch
from tqdm import tqdm

# --- Configuration ---
DATA_DIR = "./treeseg_plus_handmade"        
OUTPUT_DIR = "./processed_data"
NUM_POINTS = 2048             # As requested in your snippet
TREE_CLASS_ID = 9             

def normalize_pc(points):
    """
    Center the cloud at (0,0,0) and scale to fit within a unit sphere.
    """
    centroid = np.mean(points, axis=0)
    points -= centroid
    max_dist = np.max(np.sqrt(np.sum(points**2, axis=1)))
    points /= max_dist
    return points

def process_file(file_path, output_path):
    try:
        # 1. Read LAZ file
        las = laspy.read(file_path)
        
        # 2. Extract Coordinates (N, 3)
        coords = np.vstack((las.x, las.y, las.z)).T.astype(np.float32)

        # --- FEATURE ENGINEERING: Z-NORM ---
        z_col = coords[:, 2]
        z_min = np.min(z_col)
        z_max = np.max(z_col)
        z_range = z_max - z_min
        if z_range == 0: z_range = 1.0 
        z_norm = (z_col - z_min) / z_range

        # 3. Extract SemClass and Labels
        raw_classes = las.SemClass
        labels = (raw_classes == TREE_CLASS_ID).astype(np.int64)
        
        # 4. Extract Intensity
        intensity = las.intensity.astype(np.float32)
        intensity /= 65535.0 
        
        # 5. Handle Variable Point Counts
        N = coords.shape[0]
        
        if N >= NUM_POINTS:
            choice_idx = np.random.choice(N, NUM_POINTS, replace=False)
        else:
            choice_idx = np.random.choice(N, NUM_POINTS, replace=True)
            
        sampled_coords = coords[choice_idx]
        sampled_labels = labels[choice_idx]
        sampled_intensity = intensity[choice_idx]
        sampled_z_norm = z_norm[choice_idx]
        
        # 6. Normalize Geometry
        normalized_coords = normalize_pc(sampled_coords)
        
        # 7. Create Features Vector
        # Structure: [intensity, z_norm] -> Shape (N, 2)
        # We DO NOT include coords here to allow cleaner augmentation (rotation) later.
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
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    split = ["train", "val", "test"]
    for s in split:
        if os.path.exists(os.path.join(DATA_DIR, s)):
            print(f"Processing {s} split...")
            os.makedirs(os.path.join(OUTPUT_DIR, s))

            files = glob.glob(os.path.join(DATA_DIR, s, "*.laz"))
            print(f"Found {len(files)} LAZ files.")
            
            count = 0
            for f in tqdm(files):
                filename = os.path.basename(f).replace('.laz', '.pt')
                out_path = os.path.join(OUTPUT_DIR, s, filename)
                if process_file(f, out_path):
                    count += 1
                    
            print(f"Successfully processed {count} files.")

if __name__ == "__main__":
    main()