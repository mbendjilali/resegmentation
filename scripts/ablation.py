"""
Ablation Study Script

This script progressively adds performance improvement methods to assess their impact:
1. Vanilla: Baseline training (no enhancements)
2. + Data Augmentation
3. + Class Weighting
4. + Lovasz Loss
5. + Learning Rate Scheduling
6. + SWA (Stochastic Weight Averaging)
7. Full: All methods combined

Reuses code from train.py to avoid duplication.
"""

import os
import sys
import argparse
import json
import glob
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset, Data
from tqdm import tqdm

# Add scripts directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import ForestPointNetPP
from loss import LovaszSoftmaxLoss
from train import train_transform, val_transform, DATA_ROOT, CHECKPOINT_DIR, BATCH_SIZE, LEARNING_RATE, LOVASZ_START_EPOCH, EPOCHS, SWA_START, SWA_LR, DEVICE, ForestryDataset, evaluate_single, compute_weights
import torch_geometric.transforms as T
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")

# --- CONFIGURATION ---
# Override checkpoint directory for ablation study
ABLATION_CHECKPOINT_DIR = os.path.join(CHECKPOINT_DIR, "ablation")

# Seed will be set per trial to ensure variability

# --- ABLATION CONFIGURATIONS ---
ABLATION_CONFIGS = {
    # "vanilla": {
    #     "name": "Vanilla (Baseline)",
    #     "description": "Basic training with no enhancements",
    #     "dropout": 0.5,
    #     "use_augmentation": False,
    #     "use_class_weights": False,
    #     "use_lovasz": False,
    #     "use_lr_scheduler": False,
    #     "use_swa": False,
    # },
    # "augmentation": {
    #     "name": "+ Data Augmentation",
    #     "description": "Add random rotations, scaling, flips, and jitter",
    #     "dropout": 0.5,
    #     "use_augmentation": True,
    #     "use_class_weights": False,
    #     "use_lovasz": False,
    #     "use_lr_scheduler": False,
    #     "use_swa": False,
    # },
    # "class_weights": {
    #     "name": "+ Class Weighting",
    #     "description": "Add class-weighted cross-entropy loss",
    #     "dropout": 0.5,
    #     "use_augmentation": False,
    #     "use_class_weights": True,
    #     "use_lovasz": False,
    #     "use_lr_scheduler": False,
    #     "use_swa": False,
    # },
    # "lovasz": {
    #     "name": "+ Lovasz Loss",
    #     "description": "Add Lovasz-Softmax loss (IoU-aware)",
    #     "dropout": 0.5,
    #     "use_augmentation": False,
    #     "use_class_weights": False,
    #     "use_lovasz": True,
    #     "use_lr_scheduler": False,
    #     "use_swa": False,
    # },
    # "lr_scheduler": {
    #     "name": "+ LR Scheduling",
    #     "description": "Add cosine annealing learning rate scheduler",
    #     "dropout": 0.5,
    #     "use_augmentation": True,
    #     "use_class_weights": True,
    #     "use_lovasz": True,
    #     "use_lr_scheduler": True,
    #     "use_swa": False,
    # },
    "swa": {
        "name": "+ SWA",
        "description": "Add Stochastic Weight Averaging",
        "dropout": 0.5,
        "use_augmentation": False,
        "use_class_weights": False,
        "use_lovasz": True,
        "use_lr_scheduler": False,
        "use_swa": True,
    },
}

def train_ablation(config_name, config, train_loader, val_loader, test_loader, output_dir, trial_num=None, seed=None):
    """
    Train model with specific ablation configuration.
    Reuses training logic from train.py.
    
    Args:
        config_name: Name of the ablation configuration
        config: Configuration dictionary
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        output_dir: Directory to save checkpoints and results
        trial_num: Trial number (for multiple runs of same config)
        seed: Random seed for this trial (if None, uses 42 + trial_num)
    """
    # Set seed for this trial
    if seed is None:
        seed = 42 if trial_num is None else 42 + trial_num
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    print("\n" + "="*70)
    print(f"ABLATION: {config['name']}")
    if trial_num is not None:
        print(f"Trial: {trial_num} (seed: {seed})")
    print(f"Description: {config['description']}")
    print("="*70)
    
    # Setup Model & Optimizer
    if config["dropout"]:
        model = ForestPointNetPP(num_classes=2, in_channels=5, dropout=config["dropout"]).to(DEVICE)
    else:
        raise ValueError("Dropout must be specified for the model")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    
    # Setup components based on config
    scheduler = None
    swa_model = None
    swa_scheduler = None
    
    if config["use_lr_scheduler"]:
        scheduler = CosineAnnealingLR(optimizer, T_max=SWA_START)
    
    if config["use_swa"]:
        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(optimizer, swa_lr=SWA_LR)
    
    # Setup Loss Functions
    ce_loss_fn = F.nll_loss
    lovasz_loss_fn = LovaszSoftmaxLoss() if config["use_lovasz"] else None
    class_weights = None
    
    if config["use_class_weights"]:
        class_weights = compute_weights(train_loader)
    else:
        class_weights = torch.tensor([1.0, 1.0], dtype=torch.float32).to(DEVICE)
    
    # Training Loop
    best_iou = 0.0
    best_epoch = 0
    swa_n_models = 0
    train_losses = []
    val_losses = []
    val_ious = []
    
    print(f"Starting training for {EPOCHS} epochs...")
    if config["use_swa"]:
        print(f"SWA will start at epoch {SWA_START}.")
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        
        for data in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", leave=False):
            data = data.to(DEVICE)
            optimizer.zero_grad()
            out = model(data)
            
            # Loss computation based on config
            loss_ce = ce_loss_fn(out, data.y, weight=class_weights)
            
            if config["use_lovasz"] and epoch >= LOVASZ_START_EPOCH:
                loss_lovasz = lovasz_loss_fn(out, data.y)
                loss = loss_ce + 0.8 * loss_lovasz
            else:
                loss = loss_ce
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation Step
        val_iou, val_loss = evaluate_single(model, val_loader, desc="[Val]")
        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss)
        val_ious.append(val_iou)
        
        # Save Best Model
        if val_iou > best_iou:
            best_iou = val_iou
            best_epoch = epoch + 1
            if trial_num is not None:
                checkpoint_path = os.path.join(output_dir, f"{config_name}_trial{trial_num}_best.pth")
            else:
                checkpoint_path = os.path.join(output_dir, f"{config_name}_best.pth")
            torch.save(model.state_dict(), checkpoint_path)
        
        # SWA Logic (if enabled)
        if config["use_swa"] and epoch >= SWA_START:
            threshold = best_iou * 0.95
            if val_iou >= threshold:
                swa_model.update_parameters(model)
                swa_n_models += 1
                status = f"Included (IoU {val_iou:.4f} >= {threshold:.4f})"
            else:
                status = f"Skipped (IoU {val_iou:.4f} < {threshold:.4f})"
            
            swa_scheduler.step()
            lr_display = swa_scheduler.get_last_lr()[0]
            mode_display = f"SWA Phase [{swa_n_models} models]"
        else:
            if config["use_lr_scheduler"]:
                scheduler.step()
                lr_display = scheduler.get_last_lr()[0]
            else:
                lr_display = LEARNING_RATE
            mode_display = "Standard Phase"
            status = "N/A"
        
        print(f"Epoch {epoch+1} | {mode_display} {status} | Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f} | Val mIoU: {val_iou:.4f} | LR: {lr_display:.6f}")
    
    # SWA Finalization (if enabled)
    if config["use_swa"] and swa_n_models > 0:
        print("\nUpdating SWA batch norm statistics...")
        update_bn(train_loader, swa_model, device=DEVICE)
        if trial_num is not None:
            swa_path = os.path.join(output_dir, f"{config_name}_trial{trial_num}_swa.pth")
        else:
            swa_path = os.path.join(output_dir, f"{config_name}_swa.pth")
        torch.save(swa_model.state_dict(), swa_path)
        print(f"SWA Model saved to {swa_path}")
        
        # Use SWA model for final evaluation
        final_model = swa_model
    else:
        # Load best model for final evaluation
        final_model = model
        if trial_num is not None:
            checkpoint_path = os.path.join(output_dir, f"{config_name}_trial{trial_num}_best.pth")
        else:
            checkpoint_path = os.path.join(output_dir, f"{config_name}_best.pth")
        if os.path.exists(checkpoint_path):
            final_model.load_state_dict(torch.load(checkpoint_path))
    
    # Final Test Evaluation
    print("\nFinal Evaluation on Test Set...")
    test_iou, test_loss = evaluate_single(final_model, test_loader, desc="[Test]")
    
    result = {
        "config": config_name,
        "trial": trial_num,
        "seed": seed,
        "test_iou": test_iou,
        "test_loss": test_loss,
        "best_val_iou": best_iou,
        "best_epoch": best_epoch
    }
    
    print(f"Final Test IoU: {test_iou:.2%}")
    return result


def aggregate_trial_results(config_name, output_dir):
    """
    Aggregate results from multiple trials of the same configuration.
    Returns mean and variance statistics.
    """
    
    # Find all result files for this configuration
    pattern = os.path.join(output_dir, f"result_{config_name}_trial*.json")
    result_files = glob.glob(pattern)
    
    if len(result_files) == 0:
        return None
    
    results = []
    for result_file in result_files:
        try:
            with open(result_file, "r") as f:
                results.append(json.load(f))
        except Exception as e:
            print(f"Warning: Could not load {result_file}: {e}")
            continue
    
    if len(results) == 0:
        return None
    
    # Extract metrics
    test_ious = [r["test_iou"] for r in results]
    test_losses = [r["test_loss"] for r in results]
    val_ious = [r["best_val_iou"] for r in results]
    
    # Compute statistics
    stats = {
        "config": config_name,
        "num_trials": len(results),
        "test_iou": {
            "mean": float(np.mean(test_ious)),
            "std": float(np.std(test_ious)),
            "min": float(np.min(test_ious)),
            "max": float(np.max(test_ious)),
            "values": test_ious
        },
        "test_loss": {
            "mean": float(np.mean(test_losses)),
            "std": float(np.std(test_losses)),
            "min": float(np.min(test_losses)),
            "max": float(np.max(test_losses)),
            "values": test_losses
        },
        "best_val_iou": {
            "mean": float(np.mean(val_ious)),
            "std": float(np.std(val_ious)),
            "min": float(np.min(val_ious)),
            "max": float(np.max(val_ious)),
            "values": val_ious
        },
        "trials": results
    }
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Run ablation study for model improvements")
    parser.add_argument("--mode", type=str, choices=list(ABLATION_CONFIGS.keys()) + ["all"], default="all",
                        help="Specific ablation mode to run, or 'all' to run sequence")
    parser.add_argument("--trial", type=int, default=None,
                        help="Trial number for this run (for multiple trials of same config)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (if not provided, uses 42 + trial number)")
    parser.add_argument("--aggregate", action="store_true",
                        help="Aggregate results from all trials and compute statistics")
    args = parser.parse_args()
    
    if not os.path.exists(ABLATION_CHECKPOINT_DIR):
        os.makedirs(ABLATION_CHECKPOINT_DIR)
        
    # Load Datasets
    print("Loading datasets...")
    train_path = os.path.join(DATA_ROOT, "train")
    val_path = os.path.join(DATA_ROOT, "val")
    test_path = os.path.join(DATA_ROOT, "test")
    
    # Check if splits exist
    if not (os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path)):
        print("Error: Train/Val/Test splits not found in processed_data.")
        return

    # Base datasets
    train_dataset_base = ForestryDataset(train_path, transform=val_transform)
    val_dataset = ForestryDataset(val_path, transform=val_transform)
    test_dataset = ForestryDataset(test_path, transform=val_transform)
    
    # Loaders
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    modes_to_run = []
    if args.mode == "all":
        # Order matters: progressive improvement
        modes_to_run = list(ABLATION_CONFIGS.keys())
    else:
        modes_to_run = [args.mode]
    
    # If only aggregating, skip training
    if args.aggregate and args.trial is None:
        # Aggregation only mode
        aggregated_stats = {}
        for mode in modes_to_run:
            stats = aggregate_trial_results(mode, ABLATION_CHECKPOINT_DIR)
            if stats:
                aggregated_stats[mode] = stats
                # Save aggregated stats
                stats_file = os.path.join(ABLATION_CHECKPOINT_DIR, f"stats_{mode}.json")
                with open(stats_file, "w") as f:
                    json.dump(stats, f, indent=4)
        
        # Print aggregated summary
        print("\n" + "="*70)
        print("ABLATION STUDY RESULTS (Aggregated)")
        print("="*70)
        print("{:25} | {:8} | {:12} | {:12} | {:12}".format("Method", "Trials", "Test IoU (μ±σ)", "Test Loss (μ±σ)", "Val IoU (μ±σ)"))
        print("-" * 80)
        
        baseline_iou_mean = None
        for mode in modes_to_run:
            if mode in aggregated_stats:
                stats = aggregated_stats[mode]
                iou_mean = stats["test_iou"]["mean"]
                iou_std = stats["test_iou"]["std"]
                loss_mean = stats["test_loss"]["mean"]
                loss_std = stats["test_loss"]["std"]
                val_iou_mean = stats["best_val_iou"]["mean"]
                val_iou_std = stats["best_val_iou"]["std"]
                
                if baseline_iou_mean is None:
                    baseline_iou_mean = iou_mean
                
                print(f"{mode:<25} | {stats['num_trials']:8} | {iou_mean:.4f}±{iou_std:.4f} | {loss_mean:.4f}±{loss_std:.4f} | {val_iou_mean:.4f}±{val_iou_std:.4f}")
        
        print("="*70)
        
        # Save overall aggregated results
        all_stats_file = os.path.join(ABLATION_CHECKPOINT_DIR, "aggregated_stats.json")
        with open(all_stats_file, "w") as f:
            json.dump(aggregated_stats, f, indent=4)
        print(f"\nAggregated statistics saved to {all_stats_file}")
        return
    
    results = []
    
    for mode in modes_to_run:
        config = ABLATION_CONFIGS[mode]
        
        # Configure Training Dataset based on augmentation flag
        if config["use_augmentation"]:
            train_dataset = ForestryDataset(train_path, transform=train_transform)
        else:
            train_dataset = ForestryDataset(train_path, transform=val_transform)
            
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
        
        # Run Training
        result = train_ablation(mode, config, train_loader, val_loader, test_loader, ABLATION_CHECKPOINT_DIR, 
                               trial_num=args.trial, seed=args.seed)

        # Save individual result file to avoid parallel write conflicts
        if args.trial is not None:
            result_file = os.path.join(ABLATION_CHECKPOINT_DIR, f"result_{mode}_trial{args.trial}.json")
        else:
            result_file = os.path.join(ABLATION_CHECKPOINT_DIR, f"result_{mode}.json")
        with open(result_file, "w") as f:
            json.dump(result, f, indent=4)
        
        results.append(result)
        
        # Try to update the shared results file (best effort)
        try:
            full_results_path = os.path.join(ABLATION_CHECKPOINT_DIR, "ablation_results.json")
            current_results = []
            if os.path.exists(full_results_path):
                with open(full_results_path, "r") as f:
                    try:
                        current_results = json.load(f)
                    except json.JSONDecodeError:
                        pass
            
            # Remove existing entry for this mode+trial if present
            current_results = [r for r in current_results if not (r.get("config") == mode and r.get("trial") == args.trial)]
            current_results.append(result)
            
            # Sort by order in ABLATION_CONFIGS, then by trial
            ordered_keys = list(ABLATION_CONFIGS.keys())
            current_results.sort(key=lambda x: (
                ordered_keys.index(x["config"]) if x["config"] in ordered_keys else 999,
                x.get("trial", -1)
            ))
            
            with open(full_results_path, "w") as f:
                json.dump(current_results, f, indent=4)
        except Exception as e:
            print(f"Warning: Could not update shared results file: {e}")
            
    # Aggregate results if requested
    if args.aggregate:
        print("\n" + "="*70)
        print("AGGREGATING TRIAL RESULTS")
        print("="*70)
        
        aggregated_stats = {}
        for mode in modes_to_run:
            stats = aggregate_trial_results(mode, ABLATION_CHECKPOINT_DIR)
            if stats:
                aggregated_stats[mode] = stats
                # Save aggregated stats
                stats_file = os.path.join(ABLATION_CHECKPOINT_DIR, f"stats_{mode}.json")
                with open(stats_file, "w") as f:
                    json.dump(stats, f, indent=4)
        
        # Print aggregated summary
        print("\n" + "="*70)
        print("ABLATION STUDY RESULTS (Aggregated)")
        print("="*70)
        print("{:25} | {:8} | {:12} | {:12} | {:12}".format("Method", "Trials", "Test IoU (μ±σ)", "Test Loss (μ±σ)", "Val IoU (μ±σ)"))
        print("-" * 80)
        
        baseline_iou_mean = None
        for mode in modes_to_run:
            if mode in aggregated_stats:
                stats = aggregated_stats[mode]
                iou_mean = stats["test_iou"]["mean"]
                iou_std = stats["test_iou"]["std"]
                loss_mean = stats["test_loss"]["mean"]
                loss_std = stats["test_loss"]["std"]
                val_iou_mean = stats["best_val_iou"]["mean"]
                val_iou_std = stats["best_val_iou"]["std"]
                
                if baseline_iou_mean is None:
                    baseline_iou_mean = iou_mean
                
                print(f"{mode:<25} | {stats['num_trials']:8} | {iou_mean:.4f}±{iou_std:.4f} | {loss_mean:.4f}±{loss_std:.4f} | {val_iou_mean:.4f}±{val_iou_std:.4f}")
        
        print("="*70)
        
        # Save overall aggregated results
        all_stats_file = os.path.join(ABLATION_CHECKPOINT_DIR, "aggregated_stats.json")
        with open(all_stats_file, "w") as f:
            json.dump(aggregated_stats, f, indent=4)
        print(f"\nAggregated statistics saved to {all_stats_file}")
    else:
        # Print Final Summary (single trial or non-aggregated)
        print("\n" + "="*70)
        print("ABLATION STUDY RESULTS")
        print("="*70)
        print("{:25} | {:10} | {:10}".format("Method", "Test IoU", "Gain"))
        print("-" * 50)
        
        baseline_iou = 0
        for i, res in enumerate(results):
            iou = res["test_iou"]
            if i == 0:
                baseline_iou = iou
                gain = "0.00%"
            else:
                gain = f"{iou - baseline_iou:+.2%}"
            
            trial_info = f" (trial {res.get('trial', 'N/A')})" if res.get('trial') is not None else ""
            print(f"{res['config']:<25} | {iou:.2%}   | {gain}{trial_info}")
        
        print("="*70)

if __name__ == "__main__":
    main()
