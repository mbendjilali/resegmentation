import os
import argparse
import glob
import numpy as np
import laspy
from tqdm import tqdm

try:
    from plyfile import PlyData, PlyElement
except ImportError:
    print("Error: plyfile library is required. Please install it using 'pip install plyfile'")
    sys.exit(1)

def process_file(file_path, output_dir):
    filename = os.path.basename(file_path)
    output_filename = filename.replace('.ply', '.laz')
    output_path = os.path.join(output_dir, output_filename)
    
    try:
        # Read PLY
        plydata = PlyData.read(file_path)
        vertex = plydata['testing']
        
        # Extract fields
        # Assuming standard names, but user said x, y, z, intensity, sem_class, ins_class
        # Check available properties
        prop_names = [p.name for p in vertex.properties]
        
        # Mapping to numpy arrays
        x = vertex['x']
        y = vertex['y']
        z = vertex['z']
        
        # Handle intensity (might be float or int)
        if 'intensity' in prop_names:
            intensity = vertex['intensity']
        else:
            intensity = np.zeros_like(x, dtype=np.uint16)
            
        if 'sem_class' in prop_names:
            sem_class = vertex['sem_class']
        else:
            print(f"Warning: 'sem_class' not found in {filename}. Skipping.")
            return

        if 'ins_class' in prop_names:
            ins_class = vertex['ins_class']
        else:
            # If not present, maybe default to 0? Or skip? User said "keep ... ins_class".
            # Let's assume it exists or default to 0.
            ins_class = np.zeros_like(x, dtype=np.int32)

        # Filter: Keep only Ground (1) and Fence (6)
        # sem_class might be float or int
        mask = np.isin(sem_class, [1, 6])
        
        if not np.any(mask):
            print(f"No ground/fence points in {filename}. Skipping.")
            return

        # Apply mask
        x_filtered = x[mask]
        y_filtered = y[mask]
        z_filtered = z[mask]
        intensity_filtered = intensity[mask]
        sem_class_filtered = sem_class[mask]
        ins_class_filtered = ins_class[mask]
        
        # Create new fields
        new_sem_class = sem_class_filtered.copy()
        new_ins_class = np.full_like(ins_class_filtered, -1)
        
        # Create LAZ
        # 1. Create Header
        header = laspy.LasHeader(point_format=3, version="1.2")
        header.scales = [0.001, 0.001, 0.001] # Standard scale
        # Offset to min to preserve precision
        header.offsets = [np.min(x_filtered), np.min(y_filtered), np.min(z_filtered)]
        
        las = laspy.LasData(header)
        
        las.x = x_filtered
        las.y = y_filtered
        las.z = z_filtered
        las.intensity = intensity_filtered
        
        # Add custom dimensions
        # sem_class, ins_class, new_sem_class, new_ins_class
        # sem_class usually fits in 'classification' but user asked for specific fields?
        # "keep x, y, z, intensity, sem_class and ins_class"
        # Standard LAS has 'classification'. 
        # But if we want to preserve exactly "sem_class" as a scalar field, we should add it as extra dim.
        # Or map sem_class to standard LAS classification?
        # User said: "In the LAZ files, keep x, y, z, intensity, sem_class and ins_class."
        # This implies custom dimensions or mapping. 
        # I'll add them as ExtraBytes to be safe and explicit.
        
        # Add Extra Dimensions
        las.add_extra_dim(laspy.ExtraBytesParams(name='sem_class', type=np.int32, description="Semantic Class"))
        las.sem_class = sem_class_filtered.astype(np.int32)
        
        las.add_extra_dim(laspy.ExtraBytesParams(name='ins_class', type=np.int32, description="Instance Class"))
        las.ins_class = ins_class_filtered.astype(np.int32)
        
        las.add_extra_dim(laspy.ExtraBytesParams(name='new_sem_class', type=np.int32, description="New Semantic Class"))
        las.new_sem_class = new_sem_class.astype(np.int32)
        
        las.add_extra_dim(laspy.ExtraBytesParams(name='new_ins_class', type=np.int32, description="New Instance Class"))
        las.new_ins_class = new_ins_class.astype(np.int32)
        
        # Write
        las.write(output_path)
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Extract Ground (1) and Fence (6) from PLY and save as LAZ")
    parser.add_argument("--input_dir", required=True, help="Input directory containing .ply files")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    files = glob.glob(os.path.join(args.input_dir, "*.ply"))
    
    if not files:
        print(f"No .ply files found in {args.input_dir}")
        return
        
    print(f"Found {len(files)} files. Processing...")
    
    for f in tqdm(files):
        process_file(f, args.output_dir)
        
    print("Done.")

if __name__ == "__main__":
    main()

