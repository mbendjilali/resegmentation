import os
import sys
import glob
import argparse
import numpy as np
import laspy
import torch
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
from torch.optim.swa_utils import AveragedModel
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

# Add scripts directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import ForestPointNetPP

# --- CONFIGURATION ---
NUM_POINTS = 2048
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def normalize_pc(points):
    """
    Center the cloud at (0,0,0) and scale to fit within a unit sphere.
    """
    centroid = np.mean(points, axis=0)
    points = points - centroid
    max_dist = np.max(np.sqrt(np.sum(points**2, axis=1)))
    if max_dist == 0:
        max_dist = 1.0
    points = points / max_dist
    return points

def rotate_point_cloud_z(points, angle_deg):
    """ 
    Rotates points around the Z-axis by a specific angle (degrees).
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

def load_model_from_checkpoint(checkpoint_path, dropout=0.5, is_swa=False):
    """
    Load model from checkpoint.
    """
    model = ForestPointNetPP(num_classes=2, in_channels=5, dropout=dropout).to(DEVICE)
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    try:
        state_dict = torch.load(checkpoint_path, map_location=DEVICE)
        
        if is_swa:
            # For SWA models, wrap in AveragedModel
            swa_model = AveragedModel(model)
            swa_model.load_state_dict(state_dict)
            swa_model.eval()
            return swa_model
        else:
            model.load_state_dict(state_dict)
            model.eval()
            return model
            
    except RuntimeError as e:
        print(f"Error loading checkpoint {checkpoint_path}: {e}")
        raise e

def prepare_instance_data(points, intensity, use_tta=True):
    """
    Prepare data for a single instance: normalize, resample, create TTA variations.
    Returns:
        data_list: List of PyG Data objects (one per TTA rotation)
        meta: Dictionary containing info for reconstruction
    """
    N = points.shape[0]
    if N == 0:
        return [], None

    # 1. Feature Engineering: Z-Norm (on the original instance)
    z_col = points[:, 2]
    z_min, z_max = np.min(z_col), np.max(z_col)
    z_range = z_max - z_min
    if z_range == 0: z_range = 1.0
    z_norm = (z_col - z_min) / z_range

    # 2. Normalize Geometry (Center & Scale)
    norm_coords = normalize_pc(points.copy())

    # 3. Resampling to NUM_POINTS
    if N >= NUM_POINTS:
        choice_idx = np.random.choice(N, NUM_POINTS, replace=False)
    else:
        choice_idx = np.random.choice(N, NUM_POINTS, replace=True)
        
    input_coords = norm_coords[choice_idx]
    input_intensity = intensity[choice_idx]
    input_z_norm = z_norm[choice_idx]
    
    # 4. Create TTA variations
    rotations = [0, 90, 180, 270] if use_tta else [0]
    data_list = []
    
    for angle in rotations:
        # Rotate
        rot_coords = rotate_point_cloud_z(input_coords, angle).astype(np.float32)
        
        # Features: [x, y, z, intensity, z_norm]
        features = np.column_stack((rot_coords, input_intensity, input_z_norm))
        
        # Create Tensors
        tensor_pos = torch.from_numpy(rot_coords)
        tensor_x = torch.from_numpy(features)
        
        # Create PyG Data object
        # Note: model expects data.x, data.pos. DataLoader will add data.batch.
        data = Data(pos=tensor_pos, x=tensor_x)
        data_list.append(data)
        
    meta = {
        'norm_coords': norm_coords,   # Original normalized points (N, 3)
        'input_coords': input_coords, # Resampled normalized points (NUM_POINTS, 3)
        'num_augs': len(rotations)
    }
    
    return data_list, meta

def post_process_instance(inst_labels, inst_z, inst_coords, ball_radius=0.5, tree_label=1):
    """
    Refines predictions for an instance using vectorized operations:
    If a non-tree point's Z is above the lowest tree point Z_min, check its distance R
    to the closest tree point below it. If R < ball_radius, relabel the point as tree.
    
    Args:
        inst_labels: (N,) array of predicted labels
        inst_z: (N,) array of Z coordinates
        inst_coords: (N, 3) array of XYZ coordinates
        ball_radius: Distance threshold for relabeling (default: 0.5)
        tree_label: Label value for tree class (default: 1)
    Returns:
        inst_labels: Updated labels
    """
    tree_mask = (inst_labels == tree_label)
    
    # If no tree points predicted, nothing to do
    if not np.any(tree_mask):
        return inst_labels
        
    min_z_tree = np.min(inst_z[tree_mask])
    
    # Candidate non-tree points above the lowest tree point
    candidate_mask = (inst_labels != tree_label) & (inst_z > min_z_tree)
    
    if not np.any(candidate_mask):
        return inst_labels
    
    # Extract arrays
    tree_coords = inst_coords[tree_mask]  # (Nt, 3)
    tree_z = inst_z[tree_mask]            # (Nt,)
    cand_coords = inst_coords[candidate_mask] # (Nc, 3)
    cand_z = inst_z[candidate_mask]           # (Nc,)

    # 1. Compute pairwise distances between all candidates and all tree points
    # Result is (Nc, Nt) matrix
    dists = cdist(cand_coords, tree_coords)

    # 2. Compute pairwise Z differences (Cand_Z - Tree_Z)
    # We want Tree_Z < Cand_Z, so (Cand_Z - Tree_Z) > 0
    # Shape broadcasting: (Nc, 1) - (1, Nt) -> (Nc, Nt)
    z_diff = cand_z[:, None] - tree_z[None, :]

    # 3. Filter: We only care about tree points BELOW the candidate.
    # Set distance to infinity where tree point is NOT below (z_diff <= 0)
    dists[z_diff <= 0] = np.inf

    # 4. Find min distance for each candidate
    min_dists = np.min(dists, axis=1)

    # 5. Relabel candidates where min_dist < ball_radius
    to_relabel = min_dists < ball_radius
    
    if np.any(to_relabel):
        # Map back to global indices
        cand_indices = np.where(candidate_mask)[0]
        relabel_indices = cand_indices[to_relabel]
        inst_labels[relabel_indices] = tree_label

    return inst_labels

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline Inference: Extract Instances -> Predict -> Merge -> Save (Parallelized)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input_dir", "-i", required=True, help="Input directory containing .laz files")
    parser.add_argument("--output_dir", "-o", required=True, help="Output directory")
    parser.add_argument("--checkpoint", "-c", required=True, help="Path to model checkpoint")
    parser.add_argument("--swa", action="store_true", help="Treat checkpoint as SWA model")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout value used in model (default: 0.5)")
    parser.add_argument("--no-tta", action="store_true", help="Disable Test-Time Augmentation")
    parser.add_argument("--batch_size", "-b", type=int, default=16, help="Batch size for inference")
    parser.add_argument("--ball_radius", type=float, default=0.5, help="Ball radius threshold for Z-coordinate filtering (default: 0.5)")
    
    # Sharding arguments for parallel execution
    parser.add_argument("--num_shards", type=int, default=1, help="Total number of shards (workers)")
    parser.add_argument("--shard_id", type=int, default=0, help="ID of this shard (0 to num_shards-1)")
    
    args = parser.parse_args()

    # 1. Setup
    if not os.path.exists(args.output_dir):
        # Avoid race condition in parallel creation
        try:
            os.makedirs(args.output_dir, exist_ok=True)
            print(f"Created output directory: {args.output_dir}")
        except OSError:
            pass

    print(f"Loading model from {args.checkpoint}...")
    model = load_model_from_checkpoint(args.checkpoint, dropout=args.dropout, is_swa=args.swa)
    use_tta = not args.no_tta
    print(f"Model loaded. TTA: {'Enabled' if use_tta else 'Disabled'}")
    print(f"Batch size: {args.batch_size}")

    # 2. Get Files
    files = sorted(glob.glob(os.path.join(args.input_dir, "*.laz")))
    if not files:
        print(f"No .laz files found in {args.input_dir}")
        return

    # Sharding
    if args.num_shards > 1:
        # Simple striding
        files = files[args.shard_id::args.num_shards]
        print(f"Shard {args.shard_id}/{args.num_shards}: Processing {len(files)} files.")
    else:
        print(f"Found {len(files)} files to process.")

    # 3. Process Files
    # Use position to avoid progress bar collision if running in parallel terminals (though typically hidden in bg)
    pbar_pos = args.shard_id if args.num_shards > 1 else 0
    
    for file_path in tqdm(files, desc=f"Processing Files (Shard {args.shard_id})", position=pbar_pos, leave=True):
        try:
            filename = os.path.basename(file_path)
            output_path = os.path.join(args.output_dir, filename)
            
            # Read LAZ
            las = laspy.read(file_path)
            
            # Check for PredInstance
            dim_names = list(las.point_format.dimension_names)
            if 'PredInstance' not in dim_names:
                print(f"Warning: 'PredInstance' not found in {filename}. Processing as single cloud.")
                instance_ids = [0]
                instances = np.zeros(len(las.x), dtype=np.int32)
            else:
                instances = las.PredInstance
                instance_ids = np.unique(instances)
            
            # Prepare result array
            final_classification = np.zeros(len(las.x), dtype=np.uint8)
            # Make a copy of instances to modify into new_instances
            new_instances = instances.copy()
            
            # Extract basic data
            coords = np.vstack((las.x, las.y, las.z)).T.astype(np.float32)
            intensity = las.intensity.astype(np.float32)
            intensity /= 65535.0 # Normalize 16-bit intensity
            
            # Collect data for all instances
            all_data_list = []
            all_meta_list = []
            
            # We need to map back to original indices.
            # Let's iterate and prepare
            
            for inst_id in instance_ids:
                if 'PredInstance' in dim_names:
                    mask = (instances == inst_id)
                else:
                    mask = np.ones(len(las.x), dtype=bool)
                
                if not np.any(mask):
                    continue

                inst_coords = coords[mask]
                inst_intensity = intensity[mask]
                
                data_list, meta = prepare_instance_data(inst_coords, inst_intensity, use_tta=use_tta)
                
                if meta is None:
                    continue
                    
                meta['mask'] = mask # Store mask to assign back later
                # Store original coordinates for post-processing logic
                meta['original_z'] = inst_coords[:, 2]
                meta['original_coords'] = inst_coords  # Store full XYZ coordinates 
                
                all_data_list.extend(data_list)
                all_meta_list.append(meta)
                
            if not all_data_list:
                print(f"No valid instances in {filename}. Skipping.")
                # Save empty/zeros
                new_las = laspy.LasData(las.header)
                new_las.points = las.points.copy()
                new_las.classification = final_classification
                # Even if we skip processing, we might want to save new_instances (which are just copies of original)
                # But here we didn't predict anything, so classification is 0.
                if 'PredInstance' not in dim_names:
                     new_las.add_extra_dim(laspy.ExtraBytes(name='PredInstance', type=np.int32))
                new_las.PredInstance = new_instances
                new_las.write(output_path)
                continue
                
            # Batch Inference
            loader = DataLoader(all_data_list, batch_size=args.batch_size, shuffle=False)
            
            all_probs = []
            
            with torch.no_grad():
                for batch in loader:
                    batch = batch.to(DEVICE)
                    out = model(batch) # [Batch_Nodes, Num_Classes]
                    
                    # Output is flattened points. Reshape to [Batch_Size, NUM_POINTS, Num_Classes]
                    # Note: This relies on every sample having exactly NUM_POINTS.
                    # prepare_instance_data guarantees this.
                    probs = torch.exp(out)
                    probs = probs.view(-1, NUM_POINTS, 2)
                    
                    all_probs.append(probs.cpu())
                    
            if not all_probs:
                 continue
                 
            all_probs = torch.cat(all_probs, dim=0) # [Total_Items, NUM_POINTS, 2]
            
            # Post-process and Assign
            current_idx = 0
            
            for meta in all_meta_list:
                n_augs = meta['num_augs']
                
                # Get probs for this instance (across all TTA variations)
                inst_probs_tta = all_probs[current_idx : current_idx + n_augs] # [n_augs, NUM_POINTS, 2]
                current_idx += n_augs
                
                # Average over TTA
                avg_probs = inst_probs_tta.mean(dim=0) # [NUM_POINTS, 2]
                pred_labels_downsampled = avg_probs.argmax(dim=1).numpy() # [NUM_POINTS]
                
                # Map back
                input_coords = meta['input_coords']
                norm_coords = meta['norm_coords']
                mask = meta['mask']
                inst_z = meta['original_z']
                inst_coords = meta['original_coords']
                
                nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(input_coords)
                distances, indices = nbrs.kneighbors(norm_coords)
                pred_labels = pred_labels_downsampled[indices.flatten()]
                
                # --- Post-Processing Rule 2: Non-tree above tree -> Tree (distance-based) ---
                pred_labels = post_process_instance(pred_labels, inst_z, inst_coords, 
                                                    ball_radius=args.ball_radius, tree_label=1)

                final_classification[mask] = pred_labels
                
                # --- Post-Processing Rule 1: Non-trees to PredInstance 0 ---
                # Points where pred_labels == 0 (non-tree) should be assigned instance ID 0
                # But only within this instance mask
                
                # If a point is 0 (non-tree), we set its instance ID to 0
                # If a point is 1 (tree), we keep its original instance ID (which is already in new_instances)
                
                # We need to act on new_instances at indices where 'mask' is True AND pred_labels is 0
                
                # Create a global boolean mask for the whole file:
                # We can't do it globally easily because we are iterating locally.
                # But 'mask' is the global boolean mask for this instance.
                # So we can index into new_instances[mask]
                
                current_inst_vals = new_instances[mask]
                current_inst_vals[pred_labels == 0] = 0
                new_instances[mask] = current_inst_vals

                
            # Save Result
            new_las = laspy.LasData(las.header)
            new_las.points = las.points.copy()
            new_las.classification = final_classification
            
            # Ensure PredInstance dimension exists in output
            # If input didn't have it, we created a dummy one. 
            # If input had it, we overwrote it.
            # laspy handles add_extra_dim logic.
            
            # Check if PredInstance exists in original header/points we copied
            found_pi = False
            for dim in new_las.point_format.dimension_names:
                if dim == 'PredInstance':
                    found_pi = True
                    break
            
            if not found_pi:
                new_las.add_extra_dim(laspy.ExtraBytes(name='PredInstance', type=np.int32))
                
            new_las.PredInstance = new_instances
            
            new_las.write(output_path)
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            import traceback
            traceback.print_exc()

    print(f"Shard {args.shard_id} processing complete.")

if __name__ == "__main__":
    main()
