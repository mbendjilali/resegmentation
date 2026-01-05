#!/usr/bin/env python3
"""
Extract training metrics from ablation log files and save as CSV.

This script parses log files to extract:
- Epoch number
- Training Loss
- Validation Loss
- Validation mIoU
- Learning Rate (optional)

Outputs CSV files suitable for Google Sheets.
"""

import os
import re
import glob
import csv
import argparse
from pathlib import Path


def parse_log_file(log_path):
    """
    Parse a log file and extract training metrics.
    
    Returns a list of dictionaries with epoch metrics.
    """
    metrics = []
    
    # Pattern to match epoch summary lines like:
    # Epoch 1 | Standard Phase N/A | Loss: 0.3711 | Val Loss: 0.5511 | Val mIoU: 0.8464 | LR: 0.000300
    pattern = re.compile(
        r'Epoch\s+(\d+)\s+\|\s+.*?\|\s+Loss:\s+([\d.]+)\s+\|\s+Val Loss:\s+([\d.]+)\s+\|\s+Val mIoU:\s+([\d.]+)\s+\|\s+LR:\s+([\d.e-]+)'
    )
    
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    epoch = int(match.group(1))
                    train_loss = float(match.group(2))
                    val_loss = float(match.group(3))
                    val_miou = float(match.group(4))
                    lr = float(match.group(5))
                    
                    metrics.append({
                        'epoch': epoch,
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'val_miou': val_miou,
                        'lr': lr
                    })
    except Exception as e:
        print(f"Error reading {log_path}: {e}")
        return []
    
    return sorted(metrics, key=lambda x: x['epoch'])


def write_csv(metrics, output_path, include_lr=True):
    """
    Write metrics to a CSV file.
    """
    if not metrics:
        print(f"Warning: No metrics found, skipping {output_path}")
        return
    
    fieldnames = ['epoch', 'train_loss', 'val_loss', 'val_miou']
    if include_lr:
        fieldnames.append('lr')
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics)
    
    print(f"Saved {len(metrics)} epochs to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract training metrics from ablation log files"
    )
    parser.add_argument(
        '--log-dir',
        type=str,
        default='logs',
        help='Directory containing log files (default: logs)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='csv_metrics',
        help='Directory to save CSV files (default: csv_metrics)'
    )
    parser.add_argument(
        '--pattern',
        type=str,
        default='ablation_*.log',
        help='Glob pattern to match log files (default: ablation_*.log)'
    )
    parser.add_argument(
        '--no-lr',
        action='store_true',
        help='Exclude learning rate from CSV output'
    )
    parser.add_argument(
        '--single-file',
        action='store_true',
        help='Combine all trials into a single CSV file per configuration'
    )
    args = parser.parse_args()
    
    log_dir = Path(args.log_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all log files
    log_files = sorted(glob.glob(str(log_dir / args.pattern)))
    
    if not log_files:
        print(f"No log files found matching pattern: {args.pattern} in {log_dir}")
        return
    
    print(f"Found {len(log_files)} log file(s)")
    
    if args.single_file:
        # Group by configuration (extract mode name from filename)
        configs = {}
        for log_file in log_files:
            # Extract config and trial from filename like: ablation_vanilla_trial0.log
            filename = Path(log_file).stem
            match = re.match(r'ablation_(\w+)_trial(\d+)', filename)
            if match:
                config = match.group(1)
                trial = int(match.group(2))
                if config not in configs:
                    configs[config] = []
                configs[config].append((trial, log_file))
            else:
                # Fallback: try without trial number
                match = re.match(r'ablation_(\w+)', filename)
                if match:
                    config = match.group(1)
                    if config not in configs:
                        configs[config] = []
                    configs[config].append((0, log_file))
        
        # Write combined CSV for each configuration
        for config, trials in configs.items():
            all_metrics = []
            for trial, log_file in sorted(trials):
                metrics = parse_log_file(log_file)
                for m in metrics:
                    m['trial'] = trial
                    all_metrics.append(m)
            
            if all_metrics:
                # Sort by trial, then epoch
                all_metrics.sort(key=lambda x: (x['trial'], x['epoch']))
                
                # Write CSV with trial column
                output_path = output_dir / f"{config}_all_trials.csv"
                fieldnames = ['trial', 'epoch', 'train_loss', 'val_loss', 'val_miou']
                if not args.no_lr:
                    fieldnames.append('lr')
                
                with open(output_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(all_metrics)
                
                print(f"Saved {len(all_metrics)} epochs from {len(trials)} trial(s) to {output_path}")
    else:
        # Write individual CSV for each log file
        for log_file in log_files:
            metrics = parse_log_file(log_file)
            
            if metrics:
                # Extract config and trial from filename
                filename = Path(log_file).stem
                match = re.match(r'ablation_(\w+)_trial(\d+)', filename)
                if match:
                    config = match.group(1)
                    trial = match.group(2)
                    output_filename = f"{config}_trial{trial}.csv"
                else:
                    # Fallback: use full filename
                    output_filename = f"{filename}.csv"
                
                output_path = output_dir / output_filename
                write_csv(metrics, output_path, include_lr=not args.no_lr)
    
    print(f"\nAll CSV files saved to {output_dir}")


if __name__ == "__main__":
    main()

