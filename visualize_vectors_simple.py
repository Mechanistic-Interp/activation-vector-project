#!/usr/bin/env python3
"""
Simple visualization tool for activation vectors from CSV files.
Works with minimal dependencies (just matplotlib and numpy).

Usage:
    python3 visualize_vectors_simple.py vectors_long_mode_centered.csv
    python3 visualize_vectors_simple.py vectors_long_mode_centered.csv long_1
"""

import csv
import sys
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional


def load_vectors_from_csv(filepath: str) -> Dict[str, np.ndarray]:
    """
    Load vectors from CSV where columns are samples and rows are dimensions.
    
    Returns dict mapping sample names to vectors.
    """
    vectors = {}
    
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        
        # First row contains sample names
        header = next(reader)
        
        # Initialize arrays for each sample
        for sample_name in header:
            vectors[sample_name] = []
        
        # Read dimension values
        for row in reader:
            for i, value in enumerate(row):
                vectors[header[i]].append(float(value))
    
    # Convert to numpy arrays
    for sample_name in vectors:
        vectors[sample_name] = np.array(vectors[sample_name])
    
    print(f"✅ Loaded {len(vectors)} samples with {len(vectors[header[0]])} dimensions")
    return vectors


def plot_vector_analysis(vector: np.ndarray, sample_name: str):
    """Create comprehensive visualization for a single vector."""
    
    fig = plt.figure(figsize=(24, 14))
    fig.suptitle(f'Activation Vector Analysis: {sample_name}', fontsize=14, fontweight='bold')
    
    # 1. Distribution histogram
    ax1 = plt.subplot(2, 3, 1)
    ax1.hist(vector, bins=100, alpha=0.7, color='blue', edgecolor='black', linewidth=0.5)
    ax1.axvline(np.mean(vector), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(vector):.4f}')
    ax1.axvline(np.median(vector), color='green', linestyle='--', linewidth=2,
                label=f'Median: {np.median(vector):.4f}')
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Value Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. All dimensions as bar chart - exactly like your reference image
    ax2 = plt.subplot(2, 3, 2)
    total_dims = len(vector)
    
    # Create bar chart for ALL dimensions
    indices = np.arange(total_dims)
    colors = ['blue' if v >= 0 else 'red' for v in vector]
    
    # Use bar plot for better visibility of individual dimensions
    ax2.bar(indices, vector, color=colors, width=1.0, edgecolor='none', alpha=0.7)
    
    ax2.set_xlabel('Dimension Index')
    ax2.set_ylabel('Value')
    ax2.set_title(f'Full {total_dims}-Dimensional Vector')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add vertical lines to show 5120-dim boundaries for long mode
    if total_dims == 20480:  # Long mode
        for i in range(1, 4):
            ax2.axvline(x=i*5120, color='green', linestyle='--', alpha=0.5, linewidth=1)
        # Add labels at the bottom
        ax2.text(2560, ax2.get_ylim()[0]*1.1, 'Last Token', ha='center', fontsize=7, color='green')
        ax2.text(7680, ax2.get_ylim()[0]*1.1, 'Exp 97.7%', ha='center', fontsize=7, color='green')
        ax2.text(12800, ax2.get_ylim()[0]*1.1, 'Exp 93.3%', ha='center', fontsize=7, color='green')
        ax2.text(17920, ax2.get_ylim()[0]*1.1, 'Exp 84.1%', ha='center', fontsize=7, color='green')
    
    # Set x-axis to show full range
    ax2.set_xlim(-100, total_dims + 100)
    
    # 3. Box plot
    ax3 = plt.subplot(2, 3, 3)
    bp = ax3.boxplot(vector, vert=True, patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    ax3.set_ylabel('Value')
    ax3.set_title('Box Plot')
    ax3.grid(True, alpha=0.3)
    
    # 4. Cumulative distribution
    ax4 = plt.subplot(2, 3, 4)
    sorted_values = np.sort(vector)
    cumulative = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
    ax4.plot(sorted_values, cumulative, linewidth=2, color='purple')
    ax4.set_xlabel('Value')
    ax4.set_ylabel('Cumulative Probability')
    ax4.set_title('Cumulative Distribution')
    ax4.grid(True, alpha=0.3)
    
    # 5. Heatmap view for long vectors or log histogram for short
    ax5 = plt.subplot(2, 3, 5)
    if len(vector) == 20480:  # Long mode - show as 4 x 5120 heatmap
        # Reshape to 4 rows (one per pooling strategy) x 5120 columns
        reshaped = vector.reshape(4, 5120)
        im = ax5.imshow(reshaped, aspect='auto', cmap='RdBu_r', vmin=-np.percentile(np.abs(vector), 95), 
                       vmax=np.percentile(np.abs(vector), 95))
        ax5.set_ylabel('Pooling Strategy')
        ax5.set_yticks([0, 1, 2, 3])
        ax5.set_yticklabels(['Last Token', 'Exp 97.7%', 'Exp 93.3%', 'Exp 84.1%'])
        ax5.set_xlabel('Dimension within 5120')
        ax5.set_title('Heatmap View (4 strategies × 5120 dims)')
        plt.colorbar(im, ax=ax5, fraction=0.046, pad=0.04)
    else:
        # For short vectors, show log histogram
        abs_values = np.abs(vector[vector != 0])  # Exclude zeros for log scale
        if len(abs_values) > 0:
            ax5.hist(abs_values, bins=50, alpha=0.7, color='orange', edgecolor='black', linewidth=0.5)
            ax5.set_yscale('log')
            ax5.set_xlabel('|Value|')
            ax5.set_ylabel('Log Frequency')
            ax5.set_title('Absolute Value Distribution (Log Scale)')
            ax5.grid(True, alpha=0.3)
    
    # 6. Statistics
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    # Calculate statistics
    vector_type = "LONG (4×5120)" if len(vector) == 20480 else f"SHORT ({len(vector)})"
    
    stats_text = f"""Statistics Summary - {vector_type}:
    
Dimensions:     {len(vector):,}
Mean:           {np.mean(vector):.6f}
Std Dev:        {np.std(vector):.6f}
Min:            {np.min(vector):.6f}
Max:            {np.max(vector):.6f}
Median:         {np.median(vector):.6f}
25th %ile:      {np.percentile(vector, 25):.6f}
75th %ile:      {np.percentile(vector, 75):.6f}

Zero values:    {np.sum(vector == 0):,} ({100*np.sum(vector == 0)/len(vector):.1f}%)
Positive:       {np.sum(vector > 0):,} ({100*np.sum(vector > 0)/len(vector):.1f}%)
Negative:       {np.sum(vector < 0):,} ({100*np.sum(vector < 0)/len(vector):.1f}%)

L2 Magnitude:   {np.linalg.norm(vector):.2f}
L1 Magnitude:   {np.sum(np.abs(vector)):.2f}
"""
    
    # Add per-strategy stats for long mode
    if len(vector) == 20480:
        stats_text += "\nPer-Strategy Magnitudes:\n"
        for i, name in enumerate(['Last Token', 'Exp 97.7%', 'Exp 93.3%', 'Exp 84.1%']):
            segment = vector[i*5120:(i+1)*5120]
            stats_text += f"  {name:12}: {np.linalg.norm(segment):.2f}\n"
    
    ax6.text(0.1, 0.5, stats_text, transform=ax6.transAxes, fontsize=9,
             verticalalignment='center', family='monospace')
    
    plt.tight_layout()
    plt.show()


def compare_vectors(vectors: Dict[str, np.ndarray], sample_names: List[str]):
    """Compare multiple vectors side by side."""
    
    fig = plt.figure(figsize=(20, 10))
    fig.suptitle(f'Vector Comparison: {", ".join(sample_names)}', fontsize=14, fontweight='bold')
    
    # 1. Distribution overlay
    ax1 = plt.subplot(2, 3, 1)
    colors = plt.cm.Set2(np.linspace(0, 1, len(sample_names)))
    for sample_name, color in zip(sample_names, colors):
        vector = vectors[sample_name]
        ax1.hist(vector, bins=50, alpha=0.5, label=sample_name, color=color, density=True)
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Density')
    ax1.set_title('Distribution Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Box plots
    ax2 = plt.subplot(2, 3, 2)
    data = [vectors[name] for name in sample_names]
    bp = ax2.boxplot(data, labels=sample_names, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax2.set_ylabel('Value')
    ax2.set_title('Box Plot Comparison')
    ax2.grid(True, alpha=0.3)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 3. Magnitude comparison
    ax3 = plt.subplot(2, 3, 3)
    magnitudes = [np.linalg.norm(vectors[name]) for name in sample_names]
    x = np.arange(len(sample_names))
    bars = ax3.bar(x, magnitudes, color=colors, alpha=0.7)
    ax3.set_xlabel('Sample')
    ax3.set_ylabel('L2 Magnitude')
    ax3.set_title('Vector Magnitude Comparison')
    ax3.set_xticks(x)
    ax3.set_xticklabels(sample_names, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, mag in zip(bars, magnitudes):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{mag:.1f}', ha='center', va='bottom')
    
    # 4. Dimension comparison - subsample for visibility
    ax4 = plt.subplot(2, 3, 4)
    min_len = min(len(vectors[name]) for name in sample_names)
    
    # Subsample if vectors are very long
    if min_len > 1000:
        step = min_len // 500  # Show ~500 points
        indices = np.arange(0, min_len, step)
        for sample_name, color in zip(sample_names, colors):
            vector = vectors[sample_name]
            ax4.scatter(indices, vector[::step], label=sample_name, 
                    color=color, alpha=0.5, s=1)
        
        # Add boundaries for long mode
        if min_len == 20480:
            for i in range(1, 4):
                ax4.axvline(x=i*5120, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)
    else:
        for sample_name, color in zip(sample_names, colors):
            vector = vectors[sample_name]
            ax4.plot(range(min_len), vector[:min_len], label=sample_name, 
                    color=color, alpha=0.7, linewidth=1.5)
    
    ax4.set_xlabel('Dimension Index')
    ax4.set_ylabel('Value')
    ax4.set_title(f'Dimension Comparison ({min_len} dims)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # 5. Statistics comparison table
    ax5 = plt.subplot(2, 3, 5)
    ax5.axis('off')
    
    # Build comparison table
    header = f"{'Metric':<12}"
    for name in sample_names[:3]:  # Limit to 3 for space
        header += f"{name:>12}"
    
    stats_lines = [header, "=" * (12 + 12 * min(3, len(sample_names)))]
    
    metrics = [
        ('Mean', lambda v: np.mean(v)),
        ('Std Dev', lambda v: np.std(v)),
        ('Min', lambda v: np.min(v)),
        ('Max', lambda v: np.max(v)),
        ('Magnitude', lambda v: np.linalg.norm(v))
    ]
    
    for metric_name, metric_func in metrics:
        line = f"{metric_name:<12}"
        for name in sample_names[:3]:
            value = metric_func(vectors[name])
            line += f"{value:>12.4f}"
        stats_lines.append(line)
    
    stats_text = "\n".join(stats_lines)
    ax5.text(0.1, 0.5, stats_text, transform=ax5.transAxes, fontsize=10,
             verticalalignment='center', family='monospace')
    
    # 6. Scatter plot (if exactly 2 samples)
    ax6 = plt.subplot(2, 3, 6)
    if len(sample_names) == 2:
        vec1 = vectors[sample_names[0]]
        vec2 = vectors[sample_names[1]]
        n_dims = min(500, min(len(vec1), len(vec2)))
        ax6.scatter(vec1[:n_dims], vec2[:n_dims], alpha=0.3, s=1)
        ax6.plot([min(vec1[:n_dims]), max(vec1[:n_dims])], 
                [min(vec1[:n_dims]), max(vec1[:n_dims])], 
                'r--', alpha=0.5, label='y=x')
        ax6.set_xlabel(sample_names[0])
        ax6.set_ylabel(sample_names[1])
        ax6.set_title(f'Dimension-wise Comparison (first {n_dims})')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
    else:
        ax6.axis('off')
        ax6.text(0.5, 0.5, f'Comparing {len(sample_names)} samples', 
                transform=ax6.transAxes, ha='center', va='center')
    
    plt.tight_layout()
    plt.show()


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 visualize_vectors_simple.py <csv_file> [sample_names...]")
        print("Example: python3 visualize_vectors_simple.py vectors_long_mode_centered.csv long_1 short_1")
        return
    
    csv_file = sys.argv[1]
    sample_names = sys.argv[2:] if len(sys.argv) > 2 else None
    
    print(f"📂 Loading vectors from {csv_file}...")
    vectors = load_vectors_from_csv(csv_file)
    
    if sample_names:
        # Validate sample names
        valid_samples = []
        for name in sample_names:
            if name in vectors:
                valid_samples.append(name)
            else:
                print(f"⚠️  Sample '{name}' not found in CSV")
        
        if not valid_samples:
            print("❌ No valid samples specified. Available samples:")
            print(f"   {', '.join(list(vectors.keys())[:10])}...")
            return
        
        if len(valid_samples) == 1:
            # Single vector analysis
            print(f"\n📊 Analyzing vector: {valid_samples[0]}")
            plot_vector_analysis(vectors[valid_samples[0]], valid_samples[0])
        else:
            # Compare multiple vectors
            print(f"\n📊 Comparing vectors: {', '.join(valid_samples)}")
            compare_vectors(vectors, valid_samples)
    else:
        # Default: show first sample and comparison of first 3
        all_samples = list(vectors.keys())
        print(f"\n📊 Available samples: {', '.join(all_samples)}")
        
        # Analyze first sample
        first_sample = all_samples[0]
        print(f"\n📊 Analyzing first sample: {first_sample}")
        plot_vector_analysis(vectors[first_sample], first_sample)
        
        # Compare first few samples
        if len(all_samples) >= 2:
            compare_samples = all_samples[:min(4, len(all_samples))]
            print(f"\n📊 Comparing samples: {', '.join(compare_samples)}")
            compare_vectors(vectors, compare_samples)


if __name__ == "__main__":
    main()