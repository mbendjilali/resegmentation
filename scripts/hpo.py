import os
import sys
import glob
import torch
import torch.nn.functional as F
import optuna
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset, Data
from tqdm import tqdm

# Add scripts directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import ForestPointNetPP
from loss import LovaszSoftmaxLoss
import torch_geometric.transforms as T
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
import logging
import sys

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")

# --- CONFIGURATION ---
DATA_ROOT = "./processed_data"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 50 # Keep it short for optimization
N_TRIALS = 20 

# Configure Optuna logging to stdout
optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

# --- DATASET (With Safety Checks) ---
class ForestryDataset(Dataset):
    def __init__(self, root_dir, transform=None, pre_transform=None):
        # Explicitly check path
        self.root_dir = os.path.abspath(root_dir)
        self.file_list = glob.glob(os.path.join(self.root_dir, "*.pt"))
        
        if len(self.file_list) == 0:
            if not os.path.exists(self.root_dir):
                raise FileNotFoundError(f"CRITICAL ERROR: Directory does not exist -> {self.root_dir}")
            else:
                raise FileNotFoundError(f"CRITICAL ERROR: No .pt files found in -> {self.root_dir}. \nDid you run 'preprocess.py' successfully?")
                
        super().__init__(root_dir, transform, pre_transform)

    def len(self):
        return len(self.file_list)

    def get(self, idx):
        try:
            raw_data = torch.load(self.file_list[idx])
            # Ensure we return a Data object with expected attributes
            return Data(pos=raw_data['pos'], x=raw_data['x'], y=raw_data['y'])
        except Exception as e:
            print(f"Error loading file {self.file_list[idx]}: {e}")
            raise e

def compute_weights(loader):
    print("Computing class weights from training set...")
    total_tree = 0
    total_non_tree = 0
    
    # Estimate from first 100 clouds to save time
    limit = 100
    for i, data in enumerate(loader):
        if i >= limit: break
        y = data.y
        total_tree += (y == 1).sum().item()
        total_non_tree += (y == 0).sum().item()
    
    if total_tree == 0 or total_non_tree == 0:
        print("Warning: One class is missing in the subset. Defaulting to 1:1 weights.")
        return torch.tensor([1.0, 1.0], dtype=torch.float32).to(DEVICE)

    total = total_tree + total_non_tree
    w0 = total / (2.0 * total_non_tree)
    w1 = total / (2.0 * total_tree)
    
    print(f"Weights -> Non-Tree: {w0:.2f}, Tree: {w1:.2f}")
    return torch.tensor([w0, w1], dtype=torch.float32).to(DEVICE)

# --- OBJECTIVE FUNCTION ---
def objective(trial):
    # 1. SAMPLE HYPERPARAMETERS
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    dropout = trial.suggest_float("dropout", 0.6, 0.95)
    swa_lr = trial.suggest_float("swa_lr", 1e-4, 1e-2, log=True)
    swa_start = 40
    lovasz_start_epoch = 10
    
    # VERBOSE: Print start of trial
    print(f"\n{'='*20} Trial {trial.number} Started {'='*20}")
    print(f"Params: LR={lr:.2e}, WD={weight_decay:.2e}, Drop={dropout:.2f}, SWA_LR={swa_lr:.2e}, SWA_Start={swa_start}, Lovasz_Start={lovasz_start_epoch}")

    # 2. SETUP DATA
    train_transform = T.Compose([
        T.RandomRotate(degrees=180, axis=2),
        T.RandomScale((0.7, 1.3)),
        T.RandomFlip(axis=0, p=0.5),
        T.RandomFlip(axis=1, p=0.5),
        T.RandomJitter(0.03)
    ])
    
    train_path = os.path.join(DATA_ROOT, "train")
    val_path = os.path.join(DATA_ROOT, "val")
    
    train_dataset = ForestryDataset(train_path, transform=train_transform)
    val_dataset = ForestryDataset(val_path)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)
    
    # 3. SETUP MODEL & LOSS
    model = ForestPointNetPP(num_classes=2, in_channels=5, dropout=dropout).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    swa_model = AveragedModel(model)
    scheduler = CosineAnnealingLR(optimizer, T_max=swa_start)
    swa_scheduler = SWALR(optimizer, swa_lr=swa_lr)
    
    ce_loss_fn = F.nll_loss
    lovasz_loss_fn = LovaszSoftmaxLoss()
    class_weights = compute_weights(train_loader)

    # 4. TRAINING LOOP
    best_iou = 0.0
    
    # Use tqdm for epoch progress
    epoch_pbar = tqdm(range(EPOCHS), desc=f"Trial {trial.number}", unit="ep")
    
    for epoch in epoch_pbar:
        model.train()
        train_loss = 0
        num_batches = 0
        
        for data in train_loader:
            data = data.to(DEVICE)
            optimizer.zero_grad()
            out = model(data)
            
            # Use fixed Warmup of 5 epochs for HPO stability (since EPOCHS=20)
            loss_ce = ce_loss_fn(out, data.y, weight=class_weights)
            if epoch >= lovasz_start_epoch:
                loss = loss_ce + 0.5 * lovasz_loss_fn(out, data.y)
            else:
                loss = loss_ce
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1
            
        avg_train_loss = train_loss / num_batches
            
        # SWA Logic
        if epoch >= swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()
            
        # Validation
        eval_model = swa_model if epoch >= swa_start else model
        eval_model.eval()
        
        intersection = 0
        union = 0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(DEVICE)
                out = eval_model(data)
                pred = out.argmax(dim=1)
                tree_mask = (data.y == 1)
                pred_tree_mask = (pred == 1)
                intersection += (pred_tree_mask & tree_mask).sum().item()
                union += (pred_tree_mask | tree_mask).sum().item()
        
        val_iou = intersection / (union + 1e-6)
        
        # Track BEST IoU for Stability
        if val_iou > best_iou:
            best_iou = val_iou
        
        epoch_pbar.set_postfix({'loss': f"{avg_train_loss:.4f}", 'iou': f"{val_iou:.4f}", 'best': f"{best_iou:.4f}"})
        
        # 5. REPORT TO OPTUNA (Robust Mode)
        # We report 'best_iou' instead of 'val_iou'. 
        # This makes the curve monotonic and prevents pruning due to temporary dips in unstable models.
        trial.report(best_iou, epoch)
        
        if trial.should_prune():
            print(f"\n[Trial {trial.number}] Pruned at Epoch {epoch} (Best IoU so far: {best_iou:.4f})")
            raise optuna.TrialPruned()

    if swa_start < EPOCHS:
        update_bn(train_loader, swa_model, device=DEVICE)
    
    print(f"\n[Trial {trial.number}] Finished. Best IoU: {best_iou:.4f}")
    return best_iou

# --- RUN OPTIMIZATION ---
if __name__ == "__main__":
    if not os.path.exists(DATA_ROOT):
        print(f"Error: Data root '{DATA_ROOT}' not found.")
        print("Please run 'preprocess.py' first.")
        exit(1)
    
    pruner = optuna.pruners.PercentilePruner(
        percentile=25.0, 
        n_startup_trials=5, 
        n_warmup_steps=10
    )
    
    study = optuna.create_study(direction="maximize", pruner=pruner)
    print(f"Starting optimization with {N_TRIALS} trials...")
    
    try:
        study.optimize(objective, n_trials=N_TRIALS)
        
        print("\n" + "="*50)
        print("BEST HYPERPARAMETERS FOUND:")
        print("="*50)
        print(study.best_params)
        print(f"Best Validation IoU: {study.best_value:.4f}")
        
        with open("best_hyperparams.txt", "w") as f:
            for k, v in study.best_params.items():
                f.write(f"{k}: {v}\n")
                
    except KeyboardInterrupt:
        print("Optimization interrupted.")
    except Exception as e:
        print(f"Optimization failed: {e}")