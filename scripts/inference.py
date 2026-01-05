import os
import sys
import glob
import argparse
import numpy as np
import laspy
import torch
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

# Add scripts directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test import load_model_from_checkpoint
from preprocess import normalize_pc, NUM_POINTS

# --- CONFIGURATION ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUTPUT_DIR = "./prediction"

def rotate_point_cloud_z(points, angle_deg):
    """ 
    Rotates points around the Z-axis by a specific angle (degrees).
    Tree structure is vertical, so we only rotate on Z.
    """
    if angle_deg == 0:
        return points
        
    angle_rad = np.deg2rad(angle_deg)
    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)
    
    # Standard 3D Rotation Matrix for Z-axis
    rotation_matrix = np.array([
        [cos_theta, -sin_theta, 0],
        [sin_theta, cos_theta, 0],
        [0, 0, 1]
    ])
    
    # Apply rotation
    return np.dot(points, rotation_matrix.T)


def predict_file(model, file_path, output_dir, use_tta=True):
    filename = os.path.basename(file_path)
    save_path = os.path.join(output_dir, filename)
    
    try:
        las = laspy.read(file_path)
    except Exception as e:
        print(f"Failed to read {file_path}: {e}")
        return

    coords = np.vstack((las.x, las.y, las.z)).T.astype(np.float32)
    intensity = las.intensity.astype(np.float32)
    intensity /= 65535.0 
    
    original_count = coords.shape[0]

    z_col = coords[:, 2]
    z_min, z_max = np.min(z_col), np.max(z_col)
    z_range = z_max - z_min
    if z_range == 0: z_range = 1.0
    z_norm = (z_col - z_min) / z_range

    norm_coords = normalize_pc(coords)

    if original_count >= NUM_POINTS:
        choice_idx = np.random.choice(original_count, NUM_POINTS, replace=False)
    else:
        choice_idx = np.random.choice(original_count, NUM_POINTS, replace=True)
        
    input_coords = norm_coords[choice_idx]
    input_intensity = intensity[choice_idx]
    input_z_norm = z_norm[choice_idx]
    
    # If TTA is on, we rotate 4 times. If off, just 0 degrees.
    rotations = [0, 90, 180, 270] if use_tta else [0]
    accumulated_probs = None
    
    tensor_batch = torch.zeros(NUM_POINTS, dtype=torch.long).to(DEVICE) 
    
    with torch.no_grad():
        for angle in rotations:
            rot_coords = rotate_point_cloud_z(input_coords, angle).astype(np.float32)
            features = np.column_stack((rot_coords, input_intensity, input_z_norm))
            tensor_pos = torch.from_numpy(rot_coords).to(DEVICE)
            tensor_x = torch.from_numpy(features).to(DEVICE)
            
            data = type('Data', (object,), {
                'pos': tensor_pos,
                'x': tensor_x,
                'batch': tensor_batch
            })
            
            out = model(data) # Log Softmax
            probs = torch.exp(out) # Probabilities
            
            if accumulated_probs is None:
                accumulated_probs = probs
            else:
                accumulated_probs += probs
                
    avg_probs = accumulated_probs / len(rotations)
    pred_labels = avg_probs.argmax(dim=1).cpu().numpy()

    # Map predictions back to original points
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(input_coords)
    distances, indices = nbrs.kneighbors(norm_coords)
    final_labels = pred_labels[indices.flatten()]
    
    # 8. Save
    new_las = laspy.LasData(las.header)
    new_las.x = las.x
    new_las.y = las.y
    new_las.z = las.z
    new_las.intensity = las.intensity
    new_las.classification = final_labels.astype(np.uint8)
    
    new_las.write(save_path)

def main():
    parser = argparse.ArgumentParser(
        description="Inference for Tree Segmentation with TTA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard model inference
  python inference.py --input ./data/test.laz
  
  # SWA model inference
  python inference.py --input ./data/test.laz --model ./checkpoints/swa_model_final.pth --swa
  
  # Custom dropout
  python inference.py --input ./data/test.laz --model ./checkpoints/model.pth --dropout 0.7
        """
    )
    parser.add_argument("-input", "--i", required=True, help="Path to a single .laz file or a folder containing .laz files")
    parser.add_argument("--model", "-m", help="Path to model checkpoint")
    parser.add_argument("--dropout", "-d", type=float, default=0.5, help="Dropout value used in model (default: 0.5)")
    parser.add_argument("--swa", action="store_true", help="Treat checkpoint as SWA model (AveragedModel)")
    parser.add_argument("--no-tta", action="store_true", help="Disable Test-Time Augmentation (faster but less accurate)")
    parser.add_argument("--output", "-o", default=OUTPUT_DIR, help=f"Output directory for predictions (default: {OUTPUT_DIR})")
    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)
        
    model = load_model_from_checkpoint(args.model, dropout=args.dropout, is_swa=args.swa)
    use_tta = not args.no_tta
    
    files = []
    if os.path.isdir(args.i):
        files = glob.glob(os.path.join(args.i, "*.laz"))
    elif os.path.isfile(args.i):
        files = [args.i]
    else:
        print(f"Invalid input: {args.i}")
        return

    print(f"Found {len(files)} files.")
    print(f"Model: {args.model} {'(SWA)' if args.swa else '(Standard)'}")
    print(f"Mode: {'TTA Enabled (4x Rotations)' if use_tta else 'Single Pass'}")
    
    for f in tqdm(files):
        predict_file(model, f, args.output, use_tta=use_tta)
        
    print(f"Done. Results saved to {args.output}")

if __name__ == "__main__":
    main()