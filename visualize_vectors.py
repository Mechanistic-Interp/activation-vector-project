#!/usr/bin/env python3
"""
Visualize activation vectors from CSV files.

This tool provides comprehensive visualizations and analysis of activation vectors,
including distributions, statistics, and comparisons between different vector types.

Usage:
    python visualize_vectors.py --file vectors_long_mode_centered.csv
    python visualize_vectors.py --compare vectors_long_mode_centered.csv vectors_long_mode_raw.csv
    python visualize_vectors.py --file vectors_short_mode_raw.csv --samples long_1,short_1
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple
import os
from pathlib import Path

# Set style for better-looking plots
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except (OSError, KeyError):
    try:
        plt.style.use('seaborn-whitegrid')
    except (OSError, KeyError):
        plt.style.use('ggplot')  # Fallback to a commonly available style
plt.rcParams['figure.figsize'] = (15, 10)


def load_vector_csv(filepath: str) -> pd.DataFrame:
    """
    Load vectors from CSV file.
    
    The CSV format has samples as columns (long_1, short_1, etc.)
    and dimensions as rows.
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        DataFrame with vectors (transposed so samples are rows)
    """
    print(f"📂 Loading {filepath}...")
    df = pd.read_csv(filepath)
    
    # The CSV has samples as columns, so we need to transpose
    # Each column is a sample (long_1, short_1, etc.)
    # Each row is a dimension value
    print(f"✅ Loaded {len(df.columns)} samples with {len(df)} dimensions")
    
    # Transpose so that samples are rows and dimensions are columns
    df_transposed = df.T
    df_transposed.index.name = 'sample'
    df_transposed.reset_index(inplace=True)
    
    return df_transposed


def calculate_statistics(vector: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive statistics for a vector.
    
    Args:
        vector: Numpy array of vector values
        
    Returns:
        Dictionary of statistics
    """
    return {
        'mean': np.mean(vector),
        'std': np.std(vector),
        'min': np.min(vector),
        'max': np.max(vector),
        'median': np.median(vector),
        'q25': np.percentile(vector, 25),
        'q75': np.percentile(vector, 75),
        'iqr': np.percentile(vector, 75) - np.percentile(vector, 25),
        'skewness': float(pd.Series(vector).skew()),
        'kurtosis': float(pd.Series(vector).kurtosis()),
        'zero_count': np.sum(vector == 0),
        'positive_count': np.sum(vector > 0),
        'negative_count': np.sum(vector < 0),
        'magnitude': np.linalg.norm(vector)
    }


def plot_distribution(vector: np.ndarray, title: str, ax=None, color='blue', bins=100):
    """
    Plot distribution histogram with statistics overlay.
    
    Args:
        vector: Vector values to plot
        title: Plot title
        ax: Matplotlib axis (creates new if None)
        color: Color for the histogram
        bins: Number of bins for histogram
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot histogram
    ax.hist(vector, bins=bins, alpha=0.7, color=color, edgecolor='black', linewidth=0.5)
    
    # Add vertical lines for statistics
    mean_val = np.mean(vector)
    median_val = np.median(vector)
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.4f}')
    ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.4f}')
    
    # Add labels and title
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_boxplot(vectors_dict: Dict[str, np.ndarray], title: str = "Vector Distribution Comparison"):
    """
    Create box plots comparing multiple vectors.
    
    Args:
        vectors_dict: Dictionary mapping names to vectors
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    data = [v for v in vectors_dict.values()]
    labels = list(vectors_dict.keys())
    
    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    
    # Color the box plots
    colors = plt.cm.Set3(np.linspace(0, 1, len(data)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.set_ylabel('Value')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()


def plot_density_comparison(vectors_dict: Dict[str, np.ndarray], title: str = "Density Comparison"):
    """
    Plot overlapping density curves for comparison.
    
    Args:
        vectors_dict: Dictionary mapping names to vectors
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(vectors_dict)))
    
    for (name, vector), color in zip(vectors_dict.items(), colors):
        # Use histogram with density=True for density plot
        ax.hist(vector, bins=50, alpha=0.5, density=True, label=name, color=color, edgecolor='none')
    
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_dimension_samples(vector: np.ndarray, title: str, n_samples: int = 100):
    """
    Plot bar chart of first N dimensions.
    
    Args:
        vector: Vector to plot
        title: Plot title
        n_samples: Number of dimensions to show
    """
    fig, ax = plt.subplots(figsize=(15, 6))
    
    n_dims = min(n_samples, len(vector))
    indices = np.arange(n_dims)
    values = vector[:n_dims]
    
    # Color based on sign
    colors = ['red' if v < 0 else 'blue' for v in values]
    
    ax.bar(indices, values, color=colors, alpha=0.7)
    
    ax.set_xlabel('Dimension Index')
    ax.set_ylabel('Value')
    ax.set_title(f'{title} - First {n_dims} Dimensions')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add zero line
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)


def create_comprehensive_plot(df: pd.DataFrame, sample_name: str, output_dir: Optional[str] = None):
    """
    Create comprehensive visualization for a single sample.
    
    Args:
        df: DataFrame with vectors
        sample_name: Name of sample to visualize
        output_dir: Directory to save plots (optional)
    """
    if sample_name not in df['sample'].values:
        print(f"❌ Sample '{sample_name}' not found in data")
        return
    
    # Extract vector and ensure it's a proper numpy array
    vector = df[df['sample'] == sample_name].iloc[0, 1:].values
    vector = np.array(vector, dtype=float).flatten()
    
    # Calculate statistics
    stats = calculate_statistics(vector)
    
    # Create subplot figure
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(f'Activation Vector Analysis: {sample_name}', fontsize=16, fontweight='bold')
    
    # 1. Distribution histogram
    ax1 = plt.subplot(2, 3, 1)
    plot_distribution(vector, 'Value Distribution', ax=ax1)
    
    # 2. Box plot
    ax2 = plt.subplot(2, 3, 2)
    ax2.boxplot(vector, vert=True, patch_artist=True)
    ax2.set_ylabel('Value')
    ax2.set_title('Box Plot')
    ax2.grid(True, alpha=0.3)
    
    # 3. Log-scale histogram for tail analysis
    ax3 = plt.subplot(2, 3, 3)
    abs_values = np.abs(vector[vector != 0])
    if len(abs_values) > 0:
        ax3.hist(abs_values, bins=50, alpha=0.7, color='orange', edgecolor='black', linewidth=0.5)
        ax3.set_yscale('log')
        ax3.set_xlabel('|Value| (absolute)')
        ax3.set_ylabel('Log Frequency')
        ax3.set_title('Absolute Value Distribution (Log Scale)')
    else:
        ax3.text(0.5, 0.5, 'No non-zero values', ha='center', va='center')
        ax3.set_title('Absolute Value Distribution')
    ax3.grid(True, alpha=0.3)
    
    # 4. Dimension samples
    ax4 = plt.subplot(2, 3, 4)
    n_dims = min(100, len(vector))
    indices = np.arange(n_dims)
    values = vector[:n_dims]
    colors = ['red' if v < 0 else 'blue' for v in values]
    ax4.bar(indices, values, color=colors, alpha=0.7)
    ax4.set_xlabel('Dimension Index')
    ax4.set_ylabel('Value')
    ax4.set_title(f'First {n_dims} Dimensions')
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Cumulative distribution
    ax5 = plt.subplot(2, 3, 5)
    sorted_values = np.sort(vector)
    cumulative = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
    ax5.plot(sorted_values, cumulative, linewidth=2)
    ax5.set_xlabel('Value')
    ax5.set_ylabel('Cumulative Probability')
    ax5.set_title('Cumulative Distribution')
    ax5.grid(True, alpha=0.3)
    
    # 6. Statistics table
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    stats_text = f"""
    Statistics Summary:
    
    Mean:      {stats['mean']:.6f}
    Std Dev:   {stats['std']:.6f}
    Min:       {stats['min']:.6f}
    Max:       {stats['max']:.6f}
    Median:    {stats['median']:.6f}
    Q25:       {stats['q25']:.6f}
    Q75:       {stats['q75']:.6f}
    IQR:       {stats['iqr']:.6f}
    
    Skewness:  {stats['skewness']:.6f}
    Kurtosis:  {stats['kurtosis']:.6f}
    
    Zero vals: {stats['zero_count']:,} ({100*stats['zero_count']/len(vector):.1f}%)
    Positive:  {stats['positive_count']:,} ({100*stats['positive_count']/len(vector):.1f}%)
    Negative:  {stats['negative_count']:,} ({100*stats['negative_count']/len(vector):.1f}%)
    
    Magnitude: {stats['magnitude']:.2f}
    Dimensions: {len(vector):,}
    """
    ax6.text(0.1, 0.5, stats_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='center', family='monospace')
    
    plt.tight_layout()
    
    # Save if output directory specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'analysis_{sample_name}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"💾 Saved plot to {output_path}")
    
    plt.show()


def compare_vectors(df1: pd.DataFrame, df2: pd.DataFrame, 
                   sample_names: Optional[List[str]] = None,
                   labels: Optional[Tuple[str, str]] = None):
    """
    Compare vectors from two different CSV files.
    
    Args:
        df1, df2: DataFrames to compare
        sample_names: Specific samples to compare (uses all if None)
        labels: Labels for the two datasets
    """
    if labels is None:
        labels = ('Dataset 1', 'Dataset 2')
    
    # Get common samples
    if sample_names is None:
        common_samples = set(df1['sample'].values) & set(df2['sample'].values)
        sample_names = list(common_samples)[:5]  # Limit to 5 for clarity
    
    print(f"📊 Comparing {len(sample_names)} samples: {sample_names}")
    
    # Create comparison plots
    fig = plt.figure(figsize=(20, 10))
    fig.suptitle('Vector Comparison Analysis', fontsize=16, fontweight='bold')
    
    # Collect all vectors for overall comparison
    all_vectors_1 = []
    all_vectors_2 = []
    
    for sample_name in sample_names:
        if sample_name in df1['sample'].values and sample_name in df2['sample'].values:
            vec1 = df1[df1['sample'] == sample_name].iloc[0, 1:].values
            vec2 = df2[df2['sample'] == sample_name].iloc[0, 1:].values
            # Ensure proper numpy arrays
            vec1 = np.array(vec1, dtype=float).flatten()
            vec2 = np.array(vec2, dtype=float).flatten()
            all_vectors_1.append(vec1)
            all_vectors_2.append(vec2)
    
    if not all_vectors_1:
        print("❌ No common samples found for comparison")
        return
    
    # Flatten for overall distribution
    flat_vec1 = np.concatenate(all_vectors_1)
    flat_vec2 = np.concatenate(all_vectors_2)
    
    # 1. Distribution comparison
    ax1 = plt.subplot(2, 3, 1)
    ax1.hist(flat_vec1, bins=100, alpha=0.5, label=labels[0], color='blue', density=True)
    ax1.hist(flat_vec2, bins=100, alpha=0.5, label=labels[1], color='red', density=True)
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Density')
    ax1.set_title('Distribution Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Box plot comparison
    ax2 = plt.subplot(2, 3, 2)
    bp = ax2.boxplot([flat_vec1, flat_vec2], labels=labels, patch_artist=True)
    bp['boxes'][0].set_facecolor('blue')
    bp['boxes'][1].set_facecolor('red')
    ax2.set_ylabel('Value')
    ax2.set_title('Box Plot Comparison')
    ax2.grid(True, alpha=0.3)
    
    # 3. Density curves (using histograms)
    ax3 = plt.subplot(2, 3, 3)
    ax3.hist(flat_vec1, bins=50, alpha=0.5, density=True, label=labels[0], color='blue', edgecolor='none')
    ax3.hist(flat_vec2, bins=50, alpha=0.5, density=True, label=labels[1], color='red', edgecolor='none')
    ax3.set_xlabel('Value')
    ax3.set_ylabel('Density')
    ax3.set_title('Density Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Magnitude comparison
    ax4 = plt.subplot(2, 3, 4)
    mags1 = [np.linalg.norm(v) for v in all_vectors_1]
    mags2 = [np.linalg.norm(v) for v in all_vectors_2]
    x = np.arange(len(sample_names))
    width = 0.35
    ax4.bar(x - width/2, mags1, width, label=labels[0], color='blue', alpha=0.7)
    ax4.bar(x + width/2, mags2, width, label=labels[1], color='red', alpha=0.7)
    ax4.set_xlabel('Sample')
    ax4.set_ylabel('Magnitude')
    ax4.set_title('Vector Magnitude Comparison')
    ax4.set_xticks(x)
    ax4.set_xticklabels(sample_names, rotation=45, ha='right')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Scatter plot (first two samples, first 100 dims)
    ax5 = plt.subplot(2, 3, 5)
    if len(all_vectors_1) >= 1:
        dims_to_plot = min(100, len(all_vectors_1[0]))
        ax5.scatter(all_vectors_1[0][:dims_to_plot], 
                   all_vectors_2[0][:dims_to_plot], alpha=0.5)
        ax5.plot([-10, 10], [-10, 10], 'r--', alpha=0.5)  # y=x line
        ax5.set_xlabel(f'{labels[0]} Values')
        ax5.set_ylabel(f'{labels[1]} Values')
        ax5.set_title(f'Dimension-wise Comparison ({sample_names[0]}, first {dims_to_plot} dims)')
        ax5.grid(True, alpha=0.3)
    
    # 6. Statistics comparison
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    stats1 = calculate_statistics(flat_vec1)
    stats2 = calculate_statistics(flat_vec2)
    
    stats_text = f"""
    Statistics Comparison:
    
    Metric        {labels[0]:>15}  {labels[1]:>15}
    {'='*50}
    Mean:         {stats1['mean']:>15.6f}  {stats2['mean']:>15.6f}
    Std Dev:      {stats1['std']:>15.6f}  {stats2['std']:>15.6f}
    Min:          {stats1['min']:>15.6f}  {stats2['min']:>15.6f}
    Max:          {stats1['max']:>15.6f}  {stats2['max']:>15.6f}
    Median:       {stats1['median']:>15.6f}  {stats2['median']:>15.6f}
    Skewness:     {stats1['skewness']:>15.6f}  {stats2['skewness']:>15.6f}
    
    Samples:      {len(sample_names):>15}  {len(sample_names):>15}
    Dimensions:   {len(all_vectors_1[0]):>15,}  {len(all_vectors_2[0]):>15,}
    """
    ax6.text(0.1, 0.5, stats_text, transform=ax6.transAxes, fontsize=9,
             verticalalignment='center', family='monospace')
    
    plt.tight_layout()
    plt.show()


def export_statistics(df: pd.DataFrame, output_file: str):
    """
    Export statistics for all samples to CSV.
    
    Args:
        df: DataFrame with vectors
        output_file: Output CSV filename
    """
    stats_list = []
    
    for sample_name in df['sample'].values:
        vector = df[df['sample'] == sample_name].iloc[0, 1:].values
        stats = calculate_statistics(vector)
        stats['sample'] = sample_name
        stats_list.append(stats)
    
    stats_df = pd.DataFrame(stats_list)
    stats_df.to_csv(output_file, index=False)
    print(f"💾 Exported statistics to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Visualize activation vectors from CSV files')
    parser.add_argument('--file', type=str, help='CSV file to visualize')
    parser.add_argument('--compare', nargs=2, help='Two CSV files to compare')
    parser.add_argument('--samples', type=str, help='Comma-separated list of samples to analyze')
    parser.add_argument('--save-plots', action='store_true', help='Save plots to files')
    parser.add_argument('--export-stats', action='store_true', help='Export statistics to CSV')
    parser.add_argument('--output-dir', type=str, default='plots', help='Directory for output files')
    
    args = parser.parse_args()
    
    # Parse sample names if provided
    sample_names = None
    if args.samples:
        sample_names = [s.strip() for s in args.samples.split(',')]
    
    if args.compare:
        # Compare mode
        print(f"📊 Comparing {args.compare[0]} vs {args.compare[1]}")
        df1 = load_vector_csv(args.compare[0])
        df2 = load_vector_csv(args.compare[1])
        
        # Extract labels from filenames
        label1 = Path(args.compare[0]).stem
        label2 = Path(args.compare[1]).stem
        
        compare_vectors(df1, df2, sample_names=sample_names, labels=(label1, label2))
        
    elif args.file:
        # Single file analysis
        df = load_vector_csv(args.file)
        
        if args.export_stats:
            stats_file = f"{Path(args.file).stem}_statistics.csv"
            export_statistics(df, stats_file)
        
        # Visualize samples
        if sample_names is None:
            # Default to first 3 samples
            sample_names = df['sample'].values[:3]
        
        for sample_name in sample_names:
            print(f"\n📈 Analyzing sample: {sample_name}")
            output_dir = args.output_dir if args.save_plots else None
            create_comprehensive_plot(df, sample_name, output_dir)
    
    else:
        # If no file specified, look for CSV files in current directory
        csv_files = list(Path('.').glob('vectors_*.csv'))
        if csv_files:
            print("📁 Found vector CSV files:")
            for i, f in enumerate(csv_files):
                print(f"  {i+1}. {f}")
            print("\nRun with --file <filename> to visualize, or --compare <file1> <file2> to compare")
        else:
            print("❌ No vector CSV files found in current directory")
            print("Run with --file <filename> or --compare <file1> <file2>")


if __name__ == "__main__":
    main()