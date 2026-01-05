import os
import sys
import glob
import torch
import torch.nn.functional as F
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

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")

# --- CONFIGURATION ---
DATA_ROOT = "./processed_data"  # Root directory containing train/val/test subfolders
CHECKPOINT_DIR = "./checkpoints"
BATCH_SIZE = 16                 
LEARNING_RATE = 3e-4
LOVASZ_START_EPOCH = 10
EPOCHS = 50
SWA_START = 35         # Start averaging weights from this epoch
SWA_LR = 8e-4         # Constant LR for SWA phase
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(42)


class AddCoordsToFeatures(object):
    """
    Concatenates pos (x,y,z) to the feature vector x.
    This ensures that geometric augmentations (on pos) are reflected in the input features.
    """
    def __call__(self, data):
        if data.x is not None:
            # Check if x already has 5 channels (legacy data case handled in Dataset, but safety check here)
            # We want to prepend pos to x
            data.x = torch.cat([data.pos, data.x], dim=1)
        else:
            data.x = data.pos
        return data

train_transform = T.Compose([
    T.RandomRotate(degrees=180, axis=2),
    # T.RandomScale((0.7, 1.3)),
    T.RandomFlip(axis=0, p=0.5),
    T.RandomFlip(axis=1, p=0.5),
    # T.RandomJitter(0.03),
    AddCoordsToFeatures()
])

val_transform = T.Compose([
    AddCoordsToFeatures()
])


# --- DATASET DEFINITION ---
class ForestryDataset(Dataset):
    def __init__(self, root_dir, transform=None, pre_transform=None):
        self.file_list = glob.glob(os.path.join(root_dir, "*.pt"))
        if len(self.file_list) == 0:
            print(f"Warning: No .pt files found in {root_dir}")
        super().__init__(root_dir, transform, pre_transform)

    def len(self):
        return len(self.file_list)

    def get(self, idx):
        raw_data = torch.load(self.file_list[idx])
        
        # Handle Legacy Data: If x has 5 channels (x,y,z,int,norm), strip x,y,z
        x = raw_data['x']
        if x.shape[1] == 5:
            x = x[:, 3:] # Keep only intensity and z_norm
            
        data = Data(
            pos=raw_data['pos'],
            x=x,
            y=raw_data['y']
        )
        return data

# --- UTILS ---
def compute_weights(loader):
    print("Computing class weights from training set...")
    total_tree = 0
    total_non_tree = 0
    limit = 100
    for i, data in enumerate(loader):
        if i >= limit: break
        y = data.y
        total_tree += (y == 1).sum().item()
        total_non_tree += (y == 0).sum().item()
    
    if total_tree == 0 or total_non_tree == 0:
        return torch.tensor([1.0, 1.0], dtype=torch.float32).to(DEVICE)

    total = total_tree + total_non_tree
    w0 = total / (2.0 * total_non_tree)
    w1 = total / (2.0 * total_tree)
    print(f"Weights -> Non-Tree: {w0:.2f}, Tree: {w1:.2f}")
    return torch.tensor([w0, w1], dtype=torch.float32).to(DEVICE)

def evaluate_single(model, loader, desc="Validation"):
    model.eval()
    intersection = 0
    union = 0
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for data in tqdm(loader, desc=desc, leave=False):
            data = data.to(DEVICE)
            out = model(data)
            
            # Simple NLL loss for tracking
            loss = F.nll_loss(out, data.y) + 0.8 * LovaszSoftmaxLoss()(out, data.y)
            total_loss += loss.item()
            num_batches += 1

            tree_mask = (data.y == 1)
            pred_tree_mask = (out.argmax(dim=1) == 1)
            intersection += (pred_tree_mask & tree_mask).sum().item()
            union += (pred_tree_mask | tree_mask).sum().item()
            
    iou = intersection / (union + 1e-6)
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    return iou, avg_loss

# --- MAIN LOOP ---
def main():
    print(f"Initializing SWA Training on {DEVICE}...")
    
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)

    # 1. Prepare Data
    train_path = os.path.join(DATA_ROOT, "train")
    val_path = os.path.join(DATA_ROOT, "val")
    test_path = os.path.join(DATA_ROOT, "test")

    # Handle dataset loading with fallback if split not found
    if os.path.exists(val_path):
        val_dataset = ForestryDataset(val_path, transform=val_transform)
        train_dataset = ForestryDataset(train_path, transform=train_transform)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    else:
        # Fallback to random split if 'val' folder missing
        # Note: This applies train_transform to validation set too, which is suboptimal but keeps logic simple
        train_dataset = ForestryDataset(train_path, transform=train_transform)
        val_size = int(len(train_dataset) * 0.20)
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [len(train_dataset) - val_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 2. Setup Model & Optimizer
    # CHANGED: Dropout lowered to 0.5 (safe default). 0.92 is likely too high.
    model = ForestPointNetPP(num_classes=2, in_channels=5, dropout=0.7).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    
    # SWA Setup
    swa_model = AveragedModel(model)
    scheduler = CosineAnnealingLR(optimizer, T_max=SWA_START) 
    swa_scheduler = SWALR(optimizer, swa_lr=SWA_LR)           
    
    # 3. Define Losses
    ce_loss_fn = F.nll_loss
    lovasz_loss_fn = LovaszSoftmaxLoss() 
    class_weights = compute_weights(train_loader)
    
    # 4. Training Loop
    best_iou = 0.0
    best_epoch = 0
    swa_n_models = 0  # <--- CRITICAL FIX: Initialized variable here
    
    print(f"Starting training for {EPOCHS} epochs. SWA starts at epoch {SWA_START}.")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        
        for data in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
            data = data.to(DEVICE)
            optimizer.zero_grad()
            out = model(data)
            
            # Hybrid Loss
            loss_ce = ce_loss_fn(out, data.y, weight=class_weights)
            
            if epoch >= LOVASZ_START_EPOCH:
                loss_lovasz = lovasz_loss_fn(out, data.y)
                loss = loss_ce + 0.8 * loss_lovasz
            else:
                loss = loss_ce
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation Step
        val_iou, val_loss = evaluate_single(model, val_loader, desc="[Val]")
        
        # Save Best Standard Model
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"best_std_model.pth"))
        
        # --- SWA LOGIC WITH GATING ---
        if epoch >= SWA_START:
            # GATING: Only include if within 95% of best performance
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
            scheduler.step()
            lr_display = scheduler.get_last_lr()[0]
            mode_display = "Standard Phase"
            status = "N/A"
            
        print(f"Epoch {epoch+1} | {mode_display} {status} | Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f} | Val mIoU: {val_iou:.4f} | LR: {lr_display:.6f}")

    # 5. SWA FINALIZATION
    print("\n" + "="*50)
    print("UPDATING SWA BATCH NORM STATISTICS...")
    print("="*50)
    
    update_bn(train_loader, swa_model, device=DEVICE)
    
    # Save SWA Model
    swa_path = os.path.join(CHECKPOINT_DIR, "swa_model_final.pth")
    if swa_n_models > 0:
        torch.save(swa_model.state_dict(), swa_path)
        print(f"SWA Model saved to {swa_path}")
    else:
        print("No models were included in the SWA average. SWA not saved.")

    # 6. FINAL TEST EVALUATION
    print("\n" + "="*50)
    print("FINAL TEST EVALUATION")
    print("="*50)
    
    if os.path.exists(test_path):
        test_dataset = ForestryDataset(test_path, transform=val_transform)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        
        # Evaluate SWA Model if available
        if swa_n_models > 0:
            print("Evaluating SWA Model...")
            test_iou, test_loss = evaluate_single(swa_model, test_loader, desc="[Test SWA]")
            print(f"Final Test mIoU (SWA): {test_iou:.2%}")
            print(f"Final Test Loss: {test_loss:.4f}")
        
        # Also Evaluate Best Standard Model for comparison
        print("Evaluating Best Standard Model...")
        best_checkpoint_path = os.path.join(CHECKPOINT_DIR, f"best_std_model.pth")
        if os.path.exists(best_checkpoint_path):
            model.load_state_dict(torch.load(best_checkpoint_path))
            test_iou, test_loss = evaluate_single(model, test_loader, desc="[Test Best]")
            print(f"Final Test mIoU (Best Std): {test_iou:.2%}")
            print(f"Final Test Loss: {test_loss:.4f}")

if __name__ == "__main__":
    main()