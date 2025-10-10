import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

TS_BINS = [[4,4,2], [6,6,3], [8,8,4], [10,10,5], [12,12,6], [14,14,7], [16,16,8], [32,32,16]]
TS_GRIDS = [(1,1), (2,2), (3,3), (4,4), (5,5), (6,6), (7,7), (8,8), (9,9), (10,10), (11,11), (12,12)]
TS_LEVEL_GRIDS = [[(1,1), (2,2)], [(2,2), (4,4)], [(4,4), (8,8)], [(3,3), (9,9)], [(1,1), (2,2), (3,3)], [(2,2), (4,4), (8,8)], [(3,3), (6,6), (9,9)]]

def load_all_results(results_dir: str = "./df_results/preprocess_resize_256_no_norm") -> dict:
    """Load all pickle files and organize results"""
    results_path = Path(results_dir)
    all_results = {}
    
    # Load baseline (hsv_histogram_concat)
    baseline_files = list(results_path.glob("*hsv_histogram_concat*.pkl"))
    if baseline_files:
        with open(baseline_files[0], 'rb') as f:
            baseline_data = pickle.load(f)
        all_results['baseline'] = {
            'method': 'hsv_histogram_concat',
            'bins': None,
            'grid': None,
            'data': baseline_data
        }
    
    # Load block histogram results
    for bins in TS_BINS:
        for grid_idx, grid in enumerate(TS_GRIDS):
            hsv_block_hist_concat_file_path = results_path / f"hsv_block_hist_concat_bins_{bins[0]}-{bins[1]}-{bins[2]}_grid_{grid[0]}-{grid[1]}.pkl"
            
            with open(hsv_block_hist_concat_file_path, 'rb') as f:
                data = pickle.load(f)
                
            key = f"block_{bins}_{grid}"
            all_results[key] = {
                'method': 'hsv_block_hist_concat',
                'bins': bins,
                'grid': grid,
                'data': data
            }

            if grid_idx < len(TS_LEVEL_GRIDS):
                hsv_hier_block_hist_concat_file_path = results_path / f"hsv_hier_block_hist_concat_bins_{bins[0]}-{bins[1]}-{bins[2]}_levels_grid_{('-'.join(f'{lvlgrid[0]}-{lvlgrid[1]}' for lvlgrid in TS_LEVEL_GRIDS[grid_idx]))}.pkl"
                with open(hsv_hier_block_hist_concat_file_path, 'rb') as f:
                    data = pickle.load(f)
                
                key = f"hier_{bins}_{TS_LEVEL_GRIDS[grid_idx]}"
                all_results[key] = {
                    'method': 'hsv_hier_block_hist_concat',
                    'bins': bins,
                    'grid': TS_LEVEL_GRIDS[grid_idx],
                    'data': data
                }
    
    return all_results

def create_comparison_dataframe(all_results):
    """Convert results to a DataFrame for easy plotting"""
    rows = []
    
    for key, result in all_results.items():
        method = result['method']
        bins = result['bins']
        grid = result['grid']
        data = result['data']
        
        for map_metric, distances in data.items():
            for distance_func, score in distances.items():
                rows.append({
                    'method': method,
                    'bins': str(bins) if bins else 'N/A',
                    'grid': str(grid) if grid else 'N/A',
                    'map_metric': map_metric,
                    'distance_function': distance_func,
                    'score': score,
                    'key': key
                })
    
    return pd.DataFrame(rows)

def plot_method_comparison(df, save_plots=True, save_dir: Path = Path("./plots")):
    """Create comprehensive comparison plots"""
    
    save_dir.mkdir(parents=True, exist_ok=True)
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Overall method comparison (best performing distance function for each)
    _, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, map_metric in enumerate(['mAP@K=1', 'mAP@K=5', 'mAP@K=10']):
        metric_data = df[df['map_metric'] == map_metric]
        
        # Get best score for each method-parameter combination
        best_scores = metric_data.groupby(['method', 'key'])['score'].max().reset_index()
        
        # Create box plot
        method_scores = []
        method_labels = []
        
        for method in best_scores['method'].unique():
            scores = best_scores[best_scores['method'] == method]['score'].values
            method_scores.append(scores)
            method_labels.append(method) # .replace('hsv_', '').replace('_concat', '')
        
        axes[i].boxplot(method_scores, labels=method_labels)
        axes[i].set_title(f'{map_metric} - Best Scores Distribution')
        axes[i].set_ylabel('mAP Score')
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_plots:
        plt.savefig(save_dir / 'method_comparison_overview.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Heatmap for block histogram method (grid vs bins)
    block_data = df[df['method'] == 'hsv_block_hist_concat']
    if not block_data.empty:
        _, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        for i, map_metric in enumerate(['mAP@K=1', 'mAP@K=5', 'mAP@K=10']):
            metric_data = block_data[block_data['map_metric'] == map_metric]
            
            # Best distance function performance
            best_perf = metric_data.groupby(['bins', 'grid'])['score'].max().reset_index()
            pivot_best = best_perf.pivot(index='bins', columns='grid', values='score')
            
            sns.heatmap(pivot_best, annot=True, fmt='.3f', cmap='viridis', 
                       ax=axes[0, i], cbar_kws={'label': 'mAP Score'})
            axes[0, i].set_title(f'{map_metric} - Best Performance\n(Block Histogram)')
            
            # Manhattan distance performance (usually performs well)
            manhattan_data = metric_data[metric_data['distance_function'] == 'compute_manhattan_distance']
            if not manhattan_data.empty:
                pivot_manhattan = manhattan_data.pivot(index='bins', columns='grid', values='score')
                sns.heatmap(pivot_manhattan, annot=True, fmt='.3f', cmap='plasma', 
                           ax=axes[1, i], cbar_kws={'label': 'mAP Score'})
                axes[1, i].set_title(f'{map_metric} - Manhattan Distance\n(Block Histogram)')
        
        plt.tight_layout()
        if save_plots:
            plt.savefig(save_dir / 'block_histogram_heatmaps.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # 3. Distance function comparison
    _, axes = plt.subplots(1, 3, figsize=(18, 6))

    for i, map_metric in enumerate(['mAP@K=1', 'mAP@K=5', 'mAP@K=10']):
        metric_data = df[df['map_metric'] == map_metric]
        
        # Group by distance function and get statistics
        distance_stats = metric_data.groupby('distance_function')['score'].agg(['mean', 'std', 'max']).reset_index()
        distance_stats['distance_function'] = distance_stats['distance_function'].str.replace('compute_', '').str.replace('_distance', '').str.replace('_', ' ')
        
        x_pos = np.arange(len(distance_stats))
        axes[i].bar(x_pos, distance_stats['mean'], yerr=distance_stats['std'], 
                   alpha=0.7, capsize=5)
        axes[i].scatter(x_pos, distance_stats['max'], color='red', s=50, 
                       label='Best Performance', zorder=5)
        
        axes[i].set_title(f'{map_metric} - Distance Function Performance')
        axes[i].set_ylabel('mAP Score')
        axes[i].set_xticks(x_pos)
        axes[i].set_xticklabels(distance_stats['distance_function'], rotation=45)
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_plots:
        plt.savefig(save_dir / 'distance_function_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. Top performing configurations
    print("\n=== TOP 10 CONFIGURATIONS ===")
    top_configs = df.groupby(['method', 'bins', 'grid', 'distance_function', 'map_metric'])['score'].first().reset_index()
    top_configs = top_configs.sort_values('score', ascending=False).head(10)
    
    for _, row in top_configs.iterrows():
        print(f"{row['method']} | bins={row['bins']} | grid={row['grid']} | "
              f"{row['distance_function']} | {row['map_metric']}: {row['score']:.4f}")

def plot_parameter_sensitivity(df, save_plots=True, save_dir: Path = Path("./plots")):
    """Plot how performance changes with different parameters"""
    
    block_data = df[df['method'] == 'hsv_block_hist_concat']
    if block_data.empty:
        return
    
    _, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Extract grid sizes for analysis
    block_data['grid_size'] = block_data['grid'].apply(lambda x: eval(x)[0] if x != 'N/A' else 0)
    block_data['bin_total'] = block_data['bins'].apply(lambda x: np.prod(eval(x)) if x != 'N/A' else 0)
    
    # 1. Grid size vs performance
    for i, map_metric in enumerate(['mAP@K=1', 'mAP@K=10']):
        metric_data = block_data[block_data['map_metric'] == map_metric]
        best_per_grid = metric_data.groupby('grid_size')['score'].max().reset_index()
        
        axes[0, i].plot(best_per_grid['grid_size'], best_per_grid['score'], 'o-', linewidth=2, markersize=8)
        axes[0, i].set_title(f'{map_metric} vs Grid Size (Best Performance)')
        axes[0, i].set_xlabel('Grid Size (NxN)')
        axes[0, i].set_ylabel('mAP Score')
        axes[0, i].grid(True, alpha=0.3)
    
    # 2. Bin count vs performance
    for i, map_metric in enumerate(['mAP@K=1', 'mAP@K=10']):
        metric_data = block_data[block_data['map_metric'] == map_metric]
        best_per_bins = metric_data.groupby('bin_total')['score'].max().reset_index()
        
        axes[1, i].plot(best_per_bins['bin_total'], best_per_bins['score'], 's-', linewidth=2, markersize=8)
        axes[1, i].set_title(f'{map_metric} vs Total Bins (Best Performance)')
        axes[1, i].set_xlabel('Total Histogram Bins')
        axes[1, i].set_ylabel('mAP Score')
        axes[1, i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_plots:
        plt.savefig(save_dir / 'parameter_sensitivity.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    # Load all results
    print("Loading results...")
    all_results = load_all_results()
    print(f"Loaded {len(all_results)} result files")
    
    # Convert to DataFrame
    df = create_comparison_dataframe(all_results)
    print(f"Created DataFrame with {len(df)} rows")
    
    # Create visualizations
    print("\nCreating comparison plots...")
    plot_method_comparison(df)
    
    print("\nCreating parameter sensitivity plots...")
    plot_parameter_sensitivity(df)
    
    # Print summary statistics
    print("\n=== SUMMARY STATISTICS ===")
    summary = df.groupby(['method', 'map_metric'])['score'].agg(['count', 'mean', 'std', 'min', 'max'])
    print(summary)
    
    return df

if __name__ == "__main__":
    df = main()