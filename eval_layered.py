"""
Layered Linear Models Analysis
학습된 레이어별 선형 모델들의 loss history를 수집하고 CSV 및 비교 차트 생성
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
OUTPUT_DIR = BASE_DIR / "eval_layered"
LOSS_DIR = OUTPUT_DIR / "loss_curves"

MODELS = [f'linear_layered_{i}' for i in range(2, 11)]  # 2 to 10 layers
BATCH_SIZE = 512  # Fixed batch size

PRETRAINS = {
    'coco': BASE_DIR / 'pth_layered_coco',
    'clevrtex': BASE_DIR / 'pth_layered_clevrtex'
}

# Create output directories
OUTPUT_DIR.mkdir(exist_ok=True)
LOSS_DIR.mkdir(exist_ok=True)

# Color palette - gradient from light to dark
COLORS = {
    'linear_layered_2': '#fee5d9',
    'linear_layered_3': '#fcbba1',
    'linear_layered_4': '#fc9272',
    'linear_layered_5': '#fb6a4a',
    'linear_layered_6': '#ef3b2c',
    'linear_layered_7': '#cb181d',
    'linear_layered_8': '#a50f15',
    'linear_layered_9': '#67000d',
    'linear_layered_10': '#000000',
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
            model_dir = pretrain_dir / model
            run_dir = find_latest_run(model_dir)
            
            if run_dir is None:
                print(f"  ⚠️  Missing: {model}")
                continue
            
            # Find loss history file
            loss_file = list(run_dir.glob(f"{model}_batch{BATCH_SIZE}_loss_history.json"))
            if not loss_file:
                print(f"  ⚠️  No loss history: {model}")
                continue
            
            # Load loss history
            with open(loss_file[0], 'r') as f:
                loss_history = json.load(f)
            
            # Add to dataframe
            for epoch, total_loss, recon_loss, feature_loss in zip(
                loss_history['epoch'], 
                loss_history['train_loss'],
                loss_history['recon_loss'],
                loss_history['feature_loss']
            ):
                data.append({
                    'pretrain': pretrain_name,
                    'model': model,
                    'num_layers': int(model.split('_')[-1]),
                    'batch_size': BATCH_SIZE,
                    'epoch': epoch,
                    'total_loss': total_loss,
                    'recon_loss': recon_loss,
                    'feature_loss': feature_loss,
                    'run_dir': str(run_dir)
                })
            
            final_total = loss_history['train_loss'][-1]
            final_recon = loss_history['recon_loss'][-1]
            final_feature = loss_history['feature_loss'][-1]
            print(f"  ✓ {model:20s} | Total: {final_total:.6f} | Recon: {final_recon:.6f} | Feature: {final_feature:.6f}")
    
    df = pd.DataFrame(data)
    return df


def save_full_csv(df):
    """Save complete loss history to CSV"""
    csv_path = OUTPUT_DIR / "loss_history_layered_full.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Full loss history saved: {csv_path}")
    print(f"  Total records: {len(df)}")


def create_comparison_csv(df):
    """Create summary comparison CSV (final losses only)"""
    # Get final epoch for each experiment
    final_df = df.loc[df.groupby(['pretrain', 'model'])['epoch'].idxmax()]
    
    # Select relevant columns
    comparison = final_df[['pretrain', 'model', 'num_layers', 'total_loss', 'recon_loss', 'feature_loss']].copy()
    comparison = comparison.sort_values(['pretrain', 'num_layers'])
    
    csv_path = OUTPUT_DIR / "loss_comparison_layered.csv"
    comparison.to_csv(csv_path, index=False)
    print(f"✓ Comparison CSV saved: {csv_path}")
    
    # Print summary table
    print("\n" + "="*80)
    print("Final Loss Comparison (by number of layers)")
    print("="*80)
    for pretrain in ['coco', 'clevrtex']:
        print(f"\n{pretrain.upper()}:")
        pretrain_data = comparison[comparison['pretrain'] == pretrain]
        print(f"{'Layers':>8} | {'Total Loss':>12} | {'Recon Loss':>12} | {'Feature Loss':>12}")
        print("-" * 60)
        for _, row in pretrain_data.iterrows():
            print(f"{row['num_layers']:>8} | {row['total_loss']:>12.6f} | {row['recon_loss']:>12.6f} | {row['feature_loss']:>12.6f}")


def plot_loss_curves_by_pretrain(df):
    """Plot loss curves for each pretrain dataset (all models together)"""
    loss_types = ['total_loss', 'recon_loss', 'feature_loss']
    loss_names = ['Total Loss', 'Reconstruction Loss', 'Feature Matching Loss']
    
    for pretrain in ['coco', 'clevrtex']:
        pretrain_df = df[df['pretrain'] == pretrain]
        
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle(f'{pretrain.upper()} - Layered Linear Models', fontsize=16, fontweight='bold')
        
        for ax, loss_type, loss_name in zip(axes, loss_types, loss_names):
            for model in MODELS:
                model_df = pretrain_df[pretrain_df['model'] == model]
                if len(model_df) > 0:
                    num_layers = int(model.split('_')[-1])
                    ax.plot(model_df['epoch'], model_df[loss_type], 
                           label=f'{num_layers} layers', 
                           color=COLORS[model], linewidth=2)
            
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Loss', fontsize=12)
            ax.set_title(loss_name, fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9, loc='best')
        
        plt.tight_layout()
        save_path = LOSS_DIR / f"{pretrain}_all_models.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Loss curves saved: {save_path}")


def plot_final_loss_comparison(df):
    """Plot bar chart comparing final losses"""
    # Get final losses
    final_df = df.loc[df.groupby(['pretrain', 'model'])['epoch'].idxmax()]
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('Final Loss Comparison (All Models)', fontsize=16, fontweight='bold')
    
    loss_types = ['total_loss', 'recon_loss', 'feature_loss']
    loss_names = ['Total Loss', 'Reconstruction Loss', 'Feature Matching Loss']
    
    for ax, loss_type, loss_name in zip(axes, loss_types, loss_names):
        # Prepare data
        coco_data = final_df[final_df['pretrain'] == 'coco'].sort_values('num_layers')
        clevrtex_data = final_df[final_df['pretrain'] == 'clevrtex'].sort_values('num_layers')
        
        x = np.arange(len(MODELS))
        width = 0.35
        
        ax.bar(x - width/2, coco_data[loss_type], width, label='COCO', alpha=0.8, color='#3498db')
        ax.bar(x + width/2, clevrtex_data[loss_type], width, label='ClevrTex', alpha=0.8, color='#e74c3c')
        
        ax.set_xlabel('Number of Layers', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title(loss_name, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(range(2, 11))
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_path = OUTPUT_DIR / "final_loss_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Final loss comparison saved: {save_path}")


def plot_loss_vs_layers(df):
    """Plot how loss changes with number of layers"""
    final_df = df.loc[df.groupby(['pretrain', 'model'])['epoch'].idxmax()]
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('Loss vs Number of Layers', fontsize=16, fontweight='bold')
    
    loss_types = ['total_loss', 'recon_loss', 'feature_loss']
    loss_names = ['Total Loss', 'Reconstruction Loss', 'Feature Matching Loss']
    
    for ax, loss_type, loss_name in zip(axes, loss_types, loss_names):
        for pretrain, color, marker in [('coco', '#3498db', 'o'), ('clevrtex', '#e74c3c', 's')]:
            pretrain_df = final_df[final_df['pretrain'] == pretrain].sort_values('num_layers')
            ax.plot(pretrain_df['num_layers'], pretrain_df[loss_type], 
                   marker=marker, markersize=8, linewidth=2, 
                   label=pretrain.upper(), color=color)
        
        ax.set_xlabel('Number of Layers', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title(loss_name, fontsize=14, fontweight='bold')
        ax.set_xticks(range(2, 11))
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = OUTPUT_DIR / "loss_vs_layers.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Loss vs layers plot saved: {save_path}")


def main():
    print("="*80)
    print("Layered Linear Models Analysis")
    print("="*80)
    
    # Collect data
    print("\n[1/5] Collecting loss data...")
    df = collect_loss_data()
    
    if df.empty:
        print("\n⚠️  No data collected. Please check if training has been completed.")
        return
    
    # Save full CSV
    print("\n[2/5] Saving full loss history...")
    save_full_csv(df)
    
    # Create comparison CSV
    print("\n[3/5] Creating comparison summary...")
    create_comparison_csv(df)
    
    # Plot loss curves
    print("\n[4/5] Plotting loss curves...")
    plot_loss_curves_by_pretrain(df)
    
    # Plot comparisons
    print("\n[5/5] Creating comparison plots...")
    plot_final_loss_comparison(df)
    plot_loss_vs_layers(df)
    
    # Summary
    print("\n" + "="*80)
    print("Analysis Complete!")
    print("="*80)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Loss curves: {LOSS_DIR}")
    print("\nGenerated files:")
    print(f"  - loss_history_layered_full.csv")
    print(f"  - loss_comparison_layered.csv")
    print(f"  - coco_all_models.png")
    print(f"  - clevrtex_all_models.png")
    print(f"  - final_loss_comparison.png")
    print(f"  - loss_vs_layers.png")
    print("="*80)


if __name__ == "__main__":
    main()
