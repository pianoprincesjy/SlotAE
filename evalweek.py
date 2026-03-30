"""
Weekend Experiments Analysis
학습된 30개 실험의 loss history를 수집하고 CSV 및 비교 차트 생성
"""
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# Configuration
BASE_DIR = Path("/home/jaey00ns/MetaSlot-main/slotae")
OUTPUT_DIR = BASE_DIR / "evalweek"
LOSS_DIR = OUTPUT_DIR / "loss_curves"

MODELS = ['linear', 'nonlinear_simple', 'nonlinear_medium', 'nonlinear_deep', 'nonlinear_gelu']
BATCH_SIZES = [64, 256, 512]
PRETRAINS = {
    'coco': BASE_DIR / 'pth_coco',
    'clevrtex': BASE_DIR / 'pth_clevrtex'
}

# Create output directories
OUTPUT_DIR.mkdir(exist_ok=True)
LOSS_DIR.mkdir(exist_ok=True)

# Color palette
COLORS = {
    'linear': '#1f77b4',
    'nonlinear_simple': '#ff7f0e',
    'nonlinear_medium': '#2ca02c',
    'nonlinear_deep': '#d62728',
    'nonlinear_gelu': '#9467bd'
}

BATCH_COLORS = {
    64: '#e74c3c',
    256: '#3498db',
    512: '#2ecc71'
}


def find_latest_run(model_dir):
    """Find the most recent training run directory"""
    if not model_dir.exists():
        return None
    
    # Get all timestamped directories
    runs = [d for d in model_dir.iterdir() if d.is_dir() and d.name.startswith('202')]
    if not runs:
        return None
    
    # Return the most recent one
    return max(runs, key=lambda p: p.name)


def collect_loss_data():
    """Collect all loss history data into a structured format"""
    data = []
    
    for pretrain_name, pretrain_dir in PRETRAINS.items():
        print(f"\n{'='*60}")
        print(f"Collecting {pretrain_name.upper()} experiments...")
        print(f"{'='*60}")
        
        for model in MODELS:
            for batch_size in BATCH_SIZES:
                model_dir = pretrain_dir / model
                run_dir = find_latest_run(model_dir)
                
                if run_dir is None:
                    print(f"  ⚠️  Missing: {model} / batch {batch_size}")
                    continue
                
                # Find loss history file
                loss_file = list(run_dir.glob(f"{model}_batch{batch_size}_loss_history.json"))
                if not loss_file:
                    print(f"  ⚠️  No loss history: {model} / batch {batch_size}")
                    continue
                
                # Load loss history
                with open(loss_file[0], 'r') as f:
                    loss_history = json.load(f)
                
                # Add to dataframe
                for epoch, loss in zip(loss_history['epoch'], loss_history['train_loss']):
                    data.append({
                        'pretrain': pretrain_name,
                        'model': model,
                        'batch_size': batch_size,
                        'epoch': epoch,
                        'train_loss': loss,
                        'run_dir': str(run_dir)
                    })
                
                final_loss = loss_history['train_loss'][-1]
                print(f"  ✓ {model:20s} | batch {batch_size:3d} | final loss: {final_loss:.6f}")
    
    return pd.DataFrame(data)


def create_summary_csv(df):
    """Create summary CSV with final losses"""
    print(f"\n{'='*60}")
    print("Creating summary CSV...")
    print(f"{'='*60}")
    
    # Get final epoch losses
    summary = df[df['epoch'] == 30].copy()
    
    # Pivot table
    pivot = summary.pivot_table(
        index=['pretrain', 'model'],
        columns='batch_size',
        values='train_loss'
    )
    
    # Save to CSV
    csv_path = OUTPUT_DIR / "loss_comparison.csv"
    pivot.to_csv(csv_path)
    print(f"✓ Summary CSV saved: {csv_path}")
    
    # Also save full history
    full_csv_path = OUTPUT_DIR / "loss_history_full.csv"
    df.to_csv(full_csv_path, index=False)
    print(f"✓ Full history CSV saved: {full_csv_path}")
    
    return pivot


def plot_pretrain_comparison(df, pretrain_name):
    """Plot all models for one pretrain dataset"""
    pretrain_df = df[df['pretrain'] == pretrain_name]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'{pretrain_name.upper()} - Loss Curves by Batch Size', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    for idx, batch_size in enumerate(BATCH_SIZES):
        ax = axes[idx]
        batch_df = pretrain_df[pretrain_df['batch_size'] == batch_size]
        
        for model in MODELS:
            model_df = batch_df[batch_df['model'] == model]
            if not model_df.empty:
                ax.plot(model_df['epoch'], model_df['train_loss'], 
                       label=model, color=COLORS[model], linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Train Loss', fontsize=12)
        ax.set_title(f'Batch Size: {batch_size}', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = LOSS_DIR / f"{pretrain_name}_all_models.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path.name}")


def plot_batch_comparison(df):
    """Plot batch size comparison for each model"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Batch Size Comparison by Model', fontsize=16, fontweight='bold', y=0.995)
    axes = axes.flatten()
    
    for idx, model in enumerate(MODELS):
        ax = axes[idx]
        model_df = df[df['model'] == model]
        
        for pretrain_name in ['coco', 'clevrtex']:
            pretrain_df = model_df[model_df['pretrain'] == pretrain_name]
            
            for batch_size in BATCH_SIZES:
                batch_df = pretrain_df[pretrain_df['batch_size'] == batch_size]
                if not batch_df.empty:
                    linestyle = '-' if pretrain_name == 'coco' else '--'
                    label = f"{pretrain_name} (batch {batch_size})"
                    ax.plot(batch_df['epoch'], batch_df['train_loss'],
                           label=label, color=BATCH_COLORS[batch_size],
                           linestyle=linestyle, linewidth=2, alpha=0.7)
        
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Train Loss', fontsize=11)
        ax.set_title(model, fontsize=12, fontweight='bold')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)
    
    # Hide extra subplot
    axes[5].axis('off')
    
    plt.tight_layout()
    save_path = LOSS_DIR / "batch_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path.name}")


def plot_model_comparison_by_pretrain(df):
    """Compare models side-by-side for each pretrain"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Model Comparison (Batch 512)', fontsize=16, fontweight='bold', y=0.98)
    
    for idx, pretrain_name in enumerate(['coco', 'clevrtex']):
        ax = axes[idx]
        pretrain_df = df[(df['pretrain'] == pretrain_name) & (df['batch_size'] == 512)]
        
        for model in MODELS:
            model_df = pretrain_df[pretrain_df['model'] == model]
            if not model_df.empty:
                ax.plot(model_df['epoch'], model_df['train_loss'],
                       label=model, color=COLORS[model], linewidth=2.5, alpha=0.8)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Train Loss', fontsize=12)
        ax.set_title(pretrain_name.upper(), fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='upper right')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = LOSS_DIR / "model_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path.name}")


def main():
    print("\n" + "="*60)
    print("Weekend Experiments Analysis")
    print("="*60)
    
    # Collect data
    print("\n[1/4] Collecting loss data from all experiments...")
    df = collect_loss_data()
    
    if df.empty:
        print("\n❌ No data found! Make sure weekend.sh has completed.")
        return
    
    print(f"\n✓ Collected {len(df)} data points from {len(df.groupby(['pretrain', 'model', 'batch_size']))} experiments")
    
    # Create summary
    print("\n[2/4] Creating summary CSV...")
    summary = create_summary_csv(df)
    print("\nFinal Loss Summary:")
    print(summary.to_string())
    
    # Generate plots
    print(f"\n[3/4] Generating comparison plots...")
    
    plot_pretrain_comparison(df, 'coco')
    plot_pretrain_comparison(df, 'clevrtex')
    plot_batch_comparison(df)
    plot_model_comparison_by_pretrain(df)
    
    print(f"\n✓ All plots saved to: {LOSS_DIR}")
    
    # Summary statistics
    print(f"\n[4/4] Summary statistics...")
    print("\nBest performing configurations (final loss):")
    final_df = df[df['epoch'] == 30].sort_values('train_loss')
    print(final_df[['pretrain', 'model', 'batch_size', 'train_loss']].head(10).to_string(index=False))
    
    print("\n" + "="*60)
    print("Analysis completed successfully!")
    print("="*60)
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print(f"  - loss_comparison.csv")
    print(f"  - loss_history_full.csv")
    print(f"  - loss_curves/")


if __name__ == "__main__":
    main()
