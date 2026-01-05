import os
import sys
import argparse
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import laspy
import numpy as np
# Add scripts directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import ForestPointNetPP
from loss import LovaszSoftmaxLoss
from train import ForestryDataset, val_transform
from torch.optim.swa_utils import AveragedModel

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")

# --- CONFIGURATION ---
DATA_ROOT = "./processed_data"
OUTPUT_DIR = "./visualization_results"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 16

# --- EVALUATION FUNCTIONS ---
def evaluate_detailed(model, loader, export_visualization=False, desc="Evaluation"):
    """
    Comprehensive evaluation with IoU, loss, and per-class metrics.
    """
    model.eval()
    intersection = 0
    union = 0
    total_loss = 0
    num_batches = 0
    
    # Per-class metrics
    true_positives = {0: 0, 1: 0}  # TP for each class
    false_positives = {0: 0, 1: 0}  # FP for each class
    false_negatives = {0: 0, 1: 0}  # FN for each class
    total_points = {0: 0, 1: 0}    # Total points per class
    
    lovasz_loss_fn = LovaszSoftmaxLoss()
    
    with torch.no_grad():
        for data in tqdm(loader, desc=desc, leave=False):
            data = data.to(DEVICE)
            out = model(data)
            
            # Loss computation
            loss = F.nll_loss(out, data.y) + 0.8 * lovasz_loss_fn(out, data.y)
            total_loss += loss.item()
            num_batches += 1

            # Predictions
            pred = out.argmax(dim=1)
            
            # Tree class IoU (class 1)
            tree_mask = (data.y == 1)
            pred_tree_mask = (pred == 1)
            intersection += (pred_tree_mask & tree_mask).sum().item()
            union += (pred_tree_mask | tree_mask).sum().item()
            
            # Per-class metrics
            for class_id in [0, 1]:
                true_mask = (data.y == class_id)
                pred_mask = (pred == class_id)
                
                tp = (pred_mask & true_mask).sum().item()
                fp = (pred_mask & ~true_mask).sum().item()
                fn = (~pred_mask & true_mask).sum().item()
                
                true_positives[class_id] += tp
                false_positives[class_id] += fp
                false_negatives[class_id] += fn
                total_points[class_id] += true_mask.sum().item()
            
            if export_visualization:
                # Get predictions and ground truth for this batch
                batch_pred = pred.cpu().numpy()  # Shape: [N_total_points]
                batch_gt = data.y.cpu().numpy()  # Shape: [N_total_points]
                batch_pos = data.pos.cpu().numpy()  # Shape: [N_total_points, 3]
                
                # Initialize RGB arrays (Gray = 32768 for correct predictions)
                N = len(batch_pred)
                red = np.full(N, 32768, dtype=np.uint16)
                green = np.full(N, 32768, dtype=np.uint16)
                blue = np.full(N, 32768, dtype=np.uint16)
                
                # False Positives: Predicted Tree (1), Actual Non-Tree (0) -> Red
                fp_mask = (batch_pred == 1) & (batch_gt == 0)
                red[fp_mask] = 65535
                green[fp_mask] = 0
                blue[fp_mask] = 0
                
                # False Negatives: Predicted Non-Tree (0), Actual Tree (1) -> Blue
                fn_mask = (batch_pred == 0) & (batch_gt == 1)
                red[fn_mask] = 0
                green[fn_mask] = 0
                blue[fn_mask] = 65535
                
                # Create LAS file structure
                # Note: We need to handle batching - each batch may contain multiple point clouds
                batch_idx = data.batch.cpu().numpy()
                unique_batches = np.unique(batch_idx)
                
                for batch_id in unique_batches:
                    mask = (batch_idx == batch_id)
                    batch_points = batch_pos[mask]
                    batch_pred_subset = batch_pred[mask]
                    batch_red = red[mask]
                    batch_green = green[mask]
                    batch_blue = blue[mask]
                    
                    # Create a minimal LAS file
                    new_las = laspy.create(point_format=2)  # LAS 1.2 format
                    new_las.x = batch_points[:, 0]
                    new_las.y = batch_points[:, 1]
                    new_las.z = batch_points[:, 2]
                    new_las.classification = batch_pred_subset.astype(np.uint8)
                    new_las.red = batch_red
                    new_las.green = batch_green
                    new_las.blue = batch_blue
                    
                    # Generate filename from batch index
                    filename = f"visualization_batch_{batch_id}.laz"
                    output_path = os.path.join(OUTPUT_DIR, filename)
                    new_las.write(output_path)


    # Overall metrics
    iou = intersection / (union + 1e-6)
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    
    # Per-class metrics
    metrics = {}
    for class_id in [0, 1]:
        class_name = "Non-Tree" if class_id == 0 else "Tree"
        tp = true_positives[class_id]
        fp = false_positives[class_id]
        fn = false_negatives[class_id]
        
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
        iou_class = tp / (tp + fp + fn + 1e-6)
        
        metrics[class_name] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'iou': iou_class,
            'total_points': total_points[class_id]
        }
    
    return {
        'iou': iou,
        'loss': avg_loss,
        'per_class': metrics
    }

def load_model_from_checkpoint(checkpoint_path, dropout=0.5, is_swa=False):
    """
    Load model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        dropout: Dropout value (default 0.5)
        is_swa: Whether this is an SWA model (requires AveragedModel wrapper)
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

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate model checkpoints on test dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test a single checkpoint
  python test.py --checkpoint ./checkpoints/best_std_model.pth
  
  # Test multiple checkpoints
  python test.py --checkpoint ./checkpoints/best_std_model.pth ./checkpoints/swa_model_final.pth
  
  # Test with custom dropout
  python test.py --checkpoint ./checkpoints/model.pth --dropout 0.7
  
  # Test SWA model
  python test.py --checkpoint ./checkpoints/swa_model_final.pth --swa
        """
    )
    parser.add_argument(
        "--checkpoint", "-c",
        nargs="+",
        required=True,
        help="Path(s) to model checkpoint(s) to evaluate"
    )
    parser.add_argument(
        "--dropout", "-d",
        type=float,
        default=0.5,
        help="Dropout value used in model (default: 0.5)"
    )
    parser.add_argument(
        "--swa",
        action="store_true",
        help="Treat checkpoint(s) as SWA model(s) (AveragedModel)"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=DATA_ROOT,
        help=f"Root directory containing test dataset (default: {DATA_ROOT})"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=BATCH_SIZE,
        help=f"Batch size for evaluation (default: {BATCH_SIZE})"
    )
    parser.add_argument(
        "--export-visualization",
        action="store_true",
        help="Export visualization of the test dataset"
    )
    
    args = parser.parse_args()
    
    # Validate test dataset exists
    test_path = os.path.join(args.data_root, "test")
    if not os.path.exists(test_path):
        print(f"Error: Test dataset not found at {test_path}")
        print("Please ensure the test dataset exists in the processed_data directory.")
        return
    
    # Load test dataset
    print(f"Loading test dataset from {test_path}...")
    try:
        test_dataset = ForestryDataset(test_path, transform=val_transform)
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4
        )
        print(f"Test dataset loaded: {len(test_dataset)} samples")
    except Exception as e:
        print(f"Error loading test dataset: {e}")
        return
    
    # Evaluate each checkpoint
    print(f"\n{'='*70}")
    print(f"EVALUATING {len(args.checkpoint)} CHECKPOINT(S) ON TEST SET")
    print(f"{'='*70}\n")
    
    results = []
    
    for i, checkpoint_path in enumerate(args.checkpoint, 1):
        print(f"\n[{i}/{len(args.checkpoint)}] Evaluating: {checkpoint_path}")
        print("-" * 70)
        
        try:
            # Load model
            model = load_model_from_checkpoint(
                checkpoint_path,
                dropout=args.dropout,
                is_swa=args.swa
            )
            
            # Evaluate
            metrics = evaluate_detailed(model, test_loader, export_visualization=args.export_visualization, desc=f"Evaluating {os.path.basename(checkpoint_path)}")
            
            # Store results
            results.append({
                'checkpoint': checkpoint_path,
                'metrics': metrics
            })
            
            # Print results
            print(f"\nResults for {os.path.basename(checkpoint_path)}:")
            print(f"  Overall mIoU: {metrics['iou']:.4f} ({metrics['iou']:.2%})")
            print(f"  Average Loss: {metrics['loss']:.4f}")
            print(f"\n  Per-Class Metrics:")
            for class_name, class_metrics in metrics['per_class'].items():
                print(f"    {class_name}:")
                print(f"      Precision: {class_metrics['precision']:.4f}")
                print(f"      Recall:    {class_metrics['recall']:.4f}")
                print(f"      F1-Score:  {class_metrics['f1']:.4f}")
                print(f"      IoU:       {class_metrics['iou']:.4f}")
                print(f"      Total Points: {class_metrics['total_points']:,}")
            
        except Exception as e:
            print(f"  ERROR: Failed to evaluate {checkpoint_path}: {e}")
            continue
    
    # Summary comparison if multiple checkpoints
    if len(results) > 1:
        print(f"\n{'='*70}")
        print("SUMMARY COMPARISON")
        print(f"{'='*70}\n")
        print(f"{'Checkpoint':<50} {'mIoU':<10} {'Loss':<10}")
        print("-" * 70)
        for result in results:
            checkpoint_name = os.path.basename(result['checkpoint'])
            if len(checkpoint_name) > 47:
                checkpoint_name = checkpoint_name[:44] + "..."
            print(f"{checkpoint_name:<50} {result['metrics']['iou']:.4f}     {result['metrics']['loss']:.4f}")
        
        # Find best
        best_result = max(results, key=lambda x: x['metrics']['iou'])
        print(f"\nBest checkpoint: {os.path.basename(best_result['checkpoint'])}")
        print(f"  mIoU: {best_result['metrics']['iou']:.4f} ({best_result['metrics']['iou']:.2%})")
    
    print(f"\n{'='*70}")
    print("Evaluation complete!")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()

