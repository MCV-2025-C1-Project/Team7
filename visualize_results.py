import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Import parameters from main.py to match file naming conventions
# These should match the parameters in test_weekn_weekm function
TESTING_BINS = [[4, 4, 2]]
TESTING_GRIDS = [(9, 9)]
WEEK3_BINS = [4, 16, 32, 64, 128, 257]
WEEK3_GRIDS = [(3, 3), (5, 5), (7, 7), (11, 11)]
RELATIVE_COEFS = False
if RELATIVE_COEFS:
    NCOEFS_LIST = [25, 50, 75, 100]
else:
    NCOEFS_LIST = [10, 20, 30, 40, 50]
LBP_POINTS = [8, 16, 24]
LBP_RADIUS = [1, 2, 3]
GLCM_GRIDS = [
    (1, 1),
    (2, 2),
    (4, 4),
    (6, 6),
    (8, 8),
    (10, 10),
    (12, 12),
    (14, 14),
    (16, 16),
]
GLCM_DISTANCES = [[1, 2], [1, 2, 3], [1, 3, 5]]
GLCM_LEVELS = [256, 128, 64, 32, 16, 8]


def load_all_results(results_dir: str = "./df_results/preprocess_resize_256") -> dict:
    """Load all pickle files and organize results automatically"""
    results_path = Path(results_dir)
    all_results = {}

    if not results_path.exists():
        print(f"Warning: Results directory {results_dir} does not exist!")
        return all_results

    # Load all pickle files
    pickle_files = list(results_path.glob("*.pkl"))
    print(f"Found {len(pickle_files)} pickle files in {results_dir}")

    for pkl_file in pickle_files:
        try:
            with open(pkl_file, "rb") as f:
                data = pickle.load(f)

            filename = pkl_file.stem

            # Parse filename to extract method and parameters
            params = parse_filename(filename)

            all_results[filename] = {
                "method": params["method"],
                "params": params,
                "data": data,
                "filename": filename,
            }

        except Exception as e:
            print(f"Error loading {pkl_file.name}: {e}")

    return all_results


def parse_filename(filename: str) -> dict:
    """Parse filename to extract method name and parameters"""
    params = {"method": "unknown", "raw": filename}

    # Detect method type
    if (
        "hsv_histogram_concat" in filename
        and "block" not in filename
        and "hier" not in filename
    ):
        params["method"] = "hsv_histogram_concat"
    elif "hsv_block_hist_concat" in filename and "hier" not in filename:
        params["method"] = "hsv_block_hist_concat"
    elif "hsv_hier_block_hist_concat" in filename:
        params["method"] = "hsv_hier_block_hist_concat"
    elif "dct_descriptor" in filename:
        params["method"] = "dct_descriptor"
    elif "lbp_descriptor" in filename:
        params["method"] = "lbp_descriptor"
    elif "glcm_descriptor" in filename:
        params["method"] = "glcm_descriptor"
    else:
        # Try to extract method from filename
        parts = filename.split("_")
        if len(parts) > 0:
            params["method"] = parts[0]

    # Extract bins if present
    if "bins_" in filename:
        try:
            bins_part = filename.split("bins_")[1].split("_")[0]
            bins_values = bins_part.split("-")
            params["bins_str"] = f"[{','.join(bins_values)}]"
        except Exception:
            pass

    # Extract grid if present
    if "grid_" in filename and "levels_grid" not in filename:
        try:
            grid_part = filename.split("grid_")[1].split("_")[0].split(".")[0]
            grid_values = grid_part.split("-")
            params["grid_str"] = f"({','.join(grid_values)})"
        except Exception:
            pass

    # Extract levels_grid if present
    if "levels_grid_" in filename:
        try:
            levels_part = filename.split("levels_grid_")[1].split(".")[0]
            params["levels_grid_str"] = levels_part
            params["grid_str"] = levels_part
        except Exception:
            pass

    # Extract ncoefs for DCT
    if "ncoefs_" in filename:
        try:
            ncoefs_part = filename.split("ncoefs_")[1].split("_")[0]
            params["ncoefs_str"] = ncoefs_part
        except Exception:
            pass

    # Extract LBP parameters
    if "points_" in filename:
        try:
            points_part = filename.split("points_")[1].split("_")[0]
            params["points_str"] = points_part
        except Exception:
            pass

    if "radius_" in filename:
        try:
            radius_part = filename.split("radius_")[1].split("_")[0].split(".")[0]
            params["radius_str"] = radius_part
        except Exception:
            pass

    # Extract GLCM parameters
    if "distances_" in filename:
        try:
            dist_part = filename.split("distances_")[1].split("_")[0]
            params["glcm_distances_str"] = dist_part
        except Exception:
            pass

    if "levels_" in filename and "levels_grid" not in filename:
        try:
            levels_part = filename.split("levels_")[1].split("_")[0].split(".")[0]
            params["levels_str"] = levels_part
        except Exception:
            pass

    return params


def create_comparison_dataframe(all_results):
    """Convert results to a DataFrame for easy plotting"""
    rows = []

    for key, result in all_results.items():
        method = result["method"]
        params = result["params"]
        data = result["data"]

        # Build parameter string for display
        param_parts = []
        if "bins_str" in params:
            param_parts.append(f"bins={params['bins_str']}")
        if "grid_str" in params:
            param_parts.append(f"grid={params['grid_str']}")
        if "ncoefs_str" in params:
            param_parts.append(f"ncoefs={params['ncoefs_str']}")
        if "points_str" in params:
            param_parts.append(f"pts={params['points_str']}")
        if "radius_str" in params:
            param_parts.append(f"r={params['radius_str']}")
        if "glcm_distances_str" in params:
            param_parts.append(f"dist={params['glcm_distances_str']}")
        if "levels_str" in params:
            param_parts.append(f"lvl={params['levels_str']}")

        param_str = ", ".join(param_parts) if param_parts else "default"

        for map_metric, distances in data.items():
            for distance_func, score in distances.items():
                if distance_func != "compute_histogram_intersection":
                    rows.append(
                        {
                            "method": method,
                            "params": param_str,
                            "map_metric": map_metric,
                            "distance_function": distance_func,
                            "score": score,
                            "filename": result["filename"],
                            **params,  # Include all parsed parameters
                        }
                    )

    return pd.DataFrame(rows)


def plot_method_comparison(df, save_plots=True, save_dir: Path = Path("./plots")):
    """Create comprehensive comparison plots with hsv_histogram_concat as baseline"""

    save_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("default")
    sns.set_palette("husl")

    # ===== 1. BASELINE VS TEXTURE DESCRIPTORS =====
    print("\n" + "=" * 80)
    print("BASELINE (hsv_histogram_concat) VS TEXTURE DESCRIPTORS")
    print("=" * 80)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    map_metric = "mAP@K=10"
    df_metric = df[df["map_metric"] == map_metric].copy()

    if not df_metric.empty:
        # Get best score per method
        best_per_method = df_metric.groupby("method")["score"].max().reset_index()

        # Separate baseline from texture descriptors
        baseline = best_per_method[best_per_method["method"] == "hsv_block_hist_concat"]
        texture_descriptors = best_per_method[
            best_per_method["method"] != "hsv_block_hist_concat"
        ]
        texture_descriptors = texture_descriptors.sort_values("score", ascending=False)

        # Create combined dataframe
        if not baseline.empty:
            combined = pd.concat([texture_descriptors, baseline])
        else:
            combined = texture_descriptors

        # Create colors (baseline in red, others in different colors)
        colors = ["red"] + list(sns.color_palette("husl", len(texture_descriptors)))

        bars = ax.barh(range(len(combined)), combined["score"], color=colors)
        ax.set_yticks(range(len(combined)))
        method_labels = (
            combined["method"]
            .str.replace("hsv_", "")
            .str.replace("_concat", "")
            .str.replace("_descriptor", "")
            .str.replace("_hist", "")
            .str.replace("block", "baseline")
        )
        ax.set_yticklabels(method_labels, fontsize=11)
        ax.set_xlabel("Score", fontsize=12)
        ax.set_title(
            f"Baseline vs Texture Descriptors - Best Performance ({map_metric})",
            fontsize=14,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3, axis="x")
        ax.set_xlim(0, 1.0)

        # Add value labels and baseline marker
        for i, (bar, score, method) in enumerate(
            zip(bars, combined["score"], combined["method"])
        ):
            label = f"{score:.3f}"
            if method == "hsv_histogram_concat":
                label += " (BASELINE)"
            ax.text(
                score + 0.01,
                i,
                label,
                va="center",
                fontsize=10,
                fontweight="bold" if method == "hsv_histogram_concat" else "normal",
            )
    else:
        print(f"No {map_metric} data found!")

    plt.tight_layout()
    if save_plots:
        plt.savefig(
            save_dir / "01_baseline_vs_texture_descriptors.png",
            dpi=300,
            bbox_inches="tight",
        )
    plt.show()

    # Print summary
    print("\nPerformance Summary (sorted by mAP@K=10):")
    df_map10 = df[df["map_metric"] == "mAP@K=10"].copy()
    best_per_method = df_map10.groupby("method")["score"].max().reset_index()
    best_per_method = best_per_method.sort_values("score", ascending=False)

    for _, row in best_per_method.iterrows():
        marker = " *** BASELINE ***" if row["method"] == "hsv_histogram_concat" else ""
        print(f"  {row['method']:30s}: {row['score']:.4f}{marker}")

    # ===== 2. TEXTURE DESCRIPTOR DETAILED ANALYSIS =====
    print("\n" + "=" * 80)
    print("TEXTURE DESCRIPTOR PARAMETER ANALYSIS")
    print("=" * 80)

    # Analyze each texture descriptor separately
    texture_methods = ["lbp", "dct", "glcm"]

    for method in texture_methods:
        method_data = df[df["method"] == method].copy()

        if method_data.empty:
            continue

        print(f"\n{'=' * 80}")
        print(f"DETAILED ANALYSIS: {method.upper()}")
        print(f"{'=' * 80}")

        # Print best configuration for each metric
        for map_metric in ["mAP@K=1", "mAP@K=5", "mAP@K=10"]:
            metric_data = method_data[method_data["map_metric"] == map_metric]
            if not metric_data.empty:
                best_config = metric_data.loc[metric_data["score"].idxmax()]
                print(
                    f"{map_metric}: {best_config['score']:.4f} | Config: {best_config['params']} | Dist: {best_config['distance_function']}"
                )

        # Create detailed parameter plots
        if method == "lbp":
            plot_lbp_parameters(method_data, save_plots, save_dir)
        elif method == "dct":
            plot_dct_parameters(method_data, save_plots, save_dir)
        elif method == "glcm":
            plot_glcm_parameters(method_data, save_plots, save_dir)

    # ===== 3. DISTANCE FUNCTION COMPARISON =====
    print("\n" + "=" * 80)
    print("DISTANCE FUNCTION COMPARISON")
    print("=" * 80)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    map_metric = "mAP@K=10"
    df_metric = df[df["map_metric"] == map_metric].copy()

    if not df_metric.empty:
        distance_stats = (
            df_metric.groupby("distance_function")["score"]
            .agg(["mean", "std", "max", "count"])
            .reset_index()
        )
        distance_stats = distance_stats.sort_values("max", ascending=False)
        distance_stats["distance_function"] = (
            distance_stats["distance_function"]
            .str.replace("compute_", "")
            .str.replace("_distance", "")
            .str.replace("_", " ")
        )

        x_pos = np.arange(len(distance_stats))
        bars = ax.bar(
            x_pos,
            distance_stats["mean"],
            yerr=distance_stats["std"],
            alpha=0.7,
            capsize=5,
            label="Mean ± Std",
        )
        ax.scatter(
            x_pos,
            distance_stats["max"],
            color="red",
            s=100,
            label="Best",
            zorder=5,
            marker="*",
        )

        ax.set_title(
            f"Distance Function Performance Comparison ({map_metric})",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_ylabel("Score", fontsize=12)
        ax.set_xlabel("Distance Function", fontsize=12)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(distance_stats["distance_function"], rotation=45, ha="right")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim(0, 1.0)

    plt.tight_layout()
    if save_plots:
        plt.savefig(
            save_dir / "02_distance_function_comparison.png",
            dpi=300,
            bbox_inches="tight",
        )
    plt.show()


def plot_lbp_parameters(lbp_data: pd.DataFrame, save_plots: bool, save_dir: Path):
    """Create detailed parameter analysis for LBP descriptor
    Parameters: lbp_points, lbp_radius, week3_bins"""

    print("\nLBP Parameters: points, radius, bins")

    # Create 1x3 grid: 1 metric (mAP@K=10) x 3 parameters (cols)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    param_configs = [
        ("points_str", "LBP Points", LBP_POINTS),
        ("radius_str", "LBP Radius", LBP_RADIUS),
        ("bins_str", "Histogram Bins", WEEK3_BINS),
    ]

    map_metric = "mAP@K=10"
    metric_data = lbp_data[lbp_data["map_metric"] == map_metric]

    if not metric_data.empty:
        for param_idx, (param_col, param_label, param_order) in enumerate(
            param_configs
        ):
            if (
                param_col not in metric_data.columns
                or not metric_data[param_col].notna().any()
            ):
                continue

            # Get best score per parameter value
            param_best = metric_data.groupby(param_col)["score"].max().reset_index()

            # Sort by parameter order defined in constants
            if param_col == "bins_str":
                # Convert bins string to first value for sorting
                param_best["sort_key"] = param_best[param_col].apply(
                    lambda x: int(x.strip("[]").split(",")[0])
                )
                param_best["order"] = param_best["sort_key"].map(
                    {val: idx for idx, val in enumerate(param_order)}
                )
            else:
                # Convert to int for sorting
                param_best["sort_key"] = param_best[param_col].astype(int)
                param_best["order"] = param_best["sort_key"].map(
                    {val: idx for idx, val in enumerate(param_order)}
                )

            param_best = param_best.sort_values("order")

            # Create bar plot
            colors = sns.color_palette("viridis", len(param_best))
            axes[param_idx].bar(
                range(len(param_best)), param_best["score"], color=colors
            )
            axes[param_idx].set_xticks(range(len(param_best)))
            axes[param_idx].set_xticklabels(
                param_best[param_col], rotation=45, ha="right"
            )
            axes[param_idx].set_title(f"{param_label}", fontsize=12, fontweight="bold")
            axes[param_idx].set_ylabel("Score (mAP@K=10)", fontsize=11)
            axes[param_idx].grid(True, alpha=0.3, axis="y")
            axes[param_idx].set_ylim(0, 1.0)

            # Add value labels
            for i, score in enumerate(param_best["score"]):
                axes[param_idx].text(
                    i, score + 0.02, f"{score:.3f}", ha="center", fontsize=9
                )

    fig.suptitle(
        "LBP Descriptor - Parameter Effects (mAP@K=10)", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()

    if save_plots:
        plt.savefig(
            save_dir / "03_lbp_parameter_analysis.png", dpi=300, bbox_inches="tight"
        )
    plt.show()

    # Print parameter interaction analysis
    print("\n  Parameter Interaction Analysis (mAP@K=10):")
    metric_data = lbp_data[lbp_data["map_metric"] == "mAP@K=10"]
    if not metric_data.empty and all(
        col in metric_data.columns for col in ["points_str", "radius_str"]
    ):
        interaction = (
            metric_data.groupby(["points_str", "radius_str"])["score"]
            .max()
            .reset_index()
        )
        interaction = interaction.sort_values("score", ascending=False).head(5)
        for _, row in interaction.iterrows():
            print(
                f"    Points={row['points_str']}, Radius={row['radius_str']}: {row['score']:.4f}"
            )


def plot_dct_parameters(dct_data: pd.DataFrame, save_plots: bool, save_dir: Path):
    """Create detailed parameter analysis for DCT descriptor
    Parameters: ncoefs, grid"""

    print("\nDCT Parameters: ncoefs, grid")

    # Create 1x2 grid: 1 metric (mAP@K=10) x 2 parameters (cols)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    param_configs = [
        ("ncoefs_str", "N Coefficients", NCOEFS_LIST),
        ("grid_str", "Grid Size", WEEK3_GRIDS),
    ]

    map_metric = "mAP@K=10"
    metric_data = dct_data[dct_data["map_metric"] == map_metric]

    if not metric_data.empty:
        for param_idx, (param_col, param_label, param_order) in enumerate(
            param_configs
        ):
            if (
                param_col not in metric_data.columns
                or not metric_data[param_col].notna().any()
            ):
                continue

            # Get best score per parameter value
            param_best = metric_data.groupby(param_col)["score"].max().reset_index()

            # Sort by parameter order defined in constants
            if param_col == "ncoefs_str":
                param_best["sort_key"] = param_best[param_col].astype(int)
                param_best["order"] = param_best["sort_key"].map(
                    {val: idx for idx, val in enumerate(param_order)}
                )
            else:  # grid_str
                # Convert "(3,3)" to (3,3) tuple for sorting
                param_best["sort_key"] = param_best[param_col].apply(lambda x: eval(x))
                param_best["order"] = param_best["sort_key"].map(
                    {val: idx for idx, val in enumerate(param_order)}
                )

            param_best = param_best.sort_values("order")

            # Create bar plot
            colors = sns.color_palette("viridis", len(param_best))
            axes[param_idx].bar(
                range(len(param_best)), param_best["score"], color=colors
            )
            axes[param_idx].set_xticks(range(len(param_best)))
            axes[param_idx].set_xticklabels(
                param_best[param_col], rotation=45, ha="right"
            )
            axes[param_idx].set_title(f"{param_label}", fontsize=12, fontweight="bold")
            axes[param_idx].set_ylabel("Score (mAP@K=10)", fontsize=11)
            axes[param_idx].grid(True, alpha=0.3, axis="y")
            axes[param_idx].set_ylim(0, 1.0)

            # Add value labels
            for i, score in enumerate(param_best["score"]):
                axes[param_idx].text(
                    i, score + 0.02, f"{score:.3f}", ha="center", fontsize=9
                )

    fig.suptitle(
        "DCT Descriptor - Parameter Effects (mAP@K=10)", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()

    if save_plots:
        plt.savefig(
            save_dir / "04_dct_parameter_analysis.png", dpi=300, bbox_inches="tight"
        )
    plt.show()

    # Print parameter interaction analysis
    print("\n  Parameter Interaction Analysis (mAP@K=10):")
    metric_data = dct_data[dct_data["map_metric"] == "mAP@K=10"]
    if not metric_data.empty and all(
        col in metric_data.columns for col in ["ncoefs_str", "grid_str"]
    ):
        interaction = (
            metric_data.groupby(["ncoefs_str", "grid_str"])["score"].max().reset_index()
        )
        interaction = interaction.sort_values("score", ascending=False).head(5)
        for _, row in interaction.iterrows():
            print(
                f"    Ncoefs={row['ncoefs_str']}, Grid={row['grid_str']}: {row['score']:.4f}"
            )


def plot_glcm_parameters(glcm_data: pd.DataFrame, save_plots: bool, save_dir: Path):
    """Create detailed parameter analysis for GLCM descriptor
    Parameters: grid, distances, levels"""

    print("\nGLCM Parameters: grid, distances, levels")

    # Create 1x3 grid: 1 metric (mAP@K=10) x 3 parameters (cols)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    param_configs = [
        ("grid_str", "Grid Size", GLCM_GRIDS),
        ("glcm_distances_str", "Distances", GLCM_DISTANCES),
        ("levels_str", "Gray Levels", GLCM_LEVELS),
    ]

    map_metric = "mAP@K=10"
    metric_data = glcm_data[glcm_data["map_metric"] == map_metric]

    if not metric_data.empty:
        for param_idx, (param_col, param_label, param_order) in enumerate(
            param_configs
        ):
            if (
                param_col not in metric_data.columns
                or not metric_data[param_col].notna().any()
            ):
                continue

            # Get best score per parameter value
            param_best = metric_data.groupby(param_col)["score"].max().reset_index()

            # Sort by parameter order defined in constants
            if param_col == "grid_str":
                # Convert "(3,3)" to (3,3) tuple for sorting
                param_best["sort_key"] = param_best[param_col].apply(lambda x: eval(x))
                param_best["order"] = param_best["sort_key"].map(
                    {val: idx for idx, val in enumerate(param_order)}
                )
            elif param_col == "glcm_distances_str":
                # Convert "1-2-3" to [1,2,3] list for sorting
                # Create mapping from string representation to order index
                dist_order_map = {
                    "-".join(map(str, val)): idx for idx, val in enumerate(param_order)
                }
                param_best["order"] = param_best[param_col].map(dist_order_map)
            else:  # levels_str
                param_best["sort_key"] = param_best[param_col].astype(int)
                param_best["order"] = param_best["sort_key"].map(
                    {val: idx for idx, val in enumerate(param_order)}
                )

            param_best = param_best.sort_values("order")

            # Create bar plot
            colors = sns.color_palette("viridis", len(param_best))
            axes[param_idx].bar(
                range(len(param_best)), param_best["score"], color=colors
            )
            axes[param_idx].set_xticks(range(len(param_best)))
            axes[param_idx].set_xticklabels(
                param_best[param_col], rotation=45, ha="right"
            )
            axes[param_idx].set_title(f"{param_label}", fontsize=12, fontweight="bold")
            axes[param_idx].set_ylabel("Score (mAP@K=10)", fontsize=11)
            axes[param_idx].grid(True, alpha=0.3, axis="y")
            axes[param_idx].set_ylim(0, 1.0)

            # Add value labels
            for i, score in enumerate(param_best["score"]):
                axes[param_idx].text(
                    i, score + 0.02, f"{score:.3f}", ha="center", fontsize=9
                )

    fig.suptitle(
        "GLCM Descriptor - Parameter Effects (mAP@K=10)", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()

    if save_plots:
        plt.savefig(
            save_dir / "05_glcm_parameter_analysis.png", dpi=300, bbox_inches="tight"
        )
    plt.show()

    # Print parameter interaction analysis
    print("\n  Parameter Interaction Analysis (mAP@K=10):")
    metric_data = glcm_data[glcm_data["map_metric"] == "mAP@K=10"]
    if not metric_data.empty and all(
        col in metric_data.columns
        for col in ["grid_str", "glcm_distances_str", "levels_str"]
    ):
        interaction = (
            metric_data.groupby(["grid_str", "glcm_distances_str", "levels_str"])[
                "score"
            ]
            .max()
            .reset_index()
        )
        interaction = interaction.sort_values("score", ascending=False).head(5)
        for _, row in interaction.iterrows():
            print(
                f"    Grid={row['grid_str']}, Dist={row['glcm_distances_str']}, Levels={row['levels_str']}: {row['score']:.4f}"
            )


def print_best_configurations(df):
    """Print the best overall configuration and best per method for all mAP metrics"""

    print("\n" + "=" * 80)
    print("BEST OVERALL CONFIGURATION (Sorted by mAP@K=10)")
    print("=" * 80)

    # Find absolute best for mAP@10
    df_map10 = df[df["map_metric"] == "mAP@K=10"].copy()
    best_idx = df_map10["score"].idxmax()
    best = df_map10.loc[best_idx]

    print("\nBest Overall (mAP@K=10):")
    print(f"  Method: {best['method']}")
    print(f"  Parameters: {best['params']}")
    print(f"  Distance Function: {best['distance_function']}")
    print(f"  mAP@K=10: {best['score']:.4f}")

    # Also show the same config for other metrics
    same_config = df[
        (df["method"] == best["method"])
        & (df["params"] == best["params"])
        & (df["distance_function"] == best["distance_function"])
    ]
    for _, row in same_config.iterrows():
        if row["map_metric"] != "mAP@K=10":
            print(f"  {row['map_metric']}: {row['score']:.4f}")

    print("\n" + "=" * 80)
    print("BEST CONFIGURATION PER METHOD (Sorted by mAP@K=10)")
    print("=" * 80)

    # Separate baseline from texture descriptors
    baseline_data = df_map10[df_map10["method"] == "hsv_histogram_concat"]
    texture_data = df_map10[df_map10["method"] != "hsv_histogram_concat"]

    # Sort texture descriptors by best mAP@10 performance
    best_per_method_map10 = texture_data.groupby("method")["score"].max().reset_index()
    best_per_method_map10 = best_per_method_map10.sort_values("score", ascending=False)

    # Print baseline first
    if not baseline_data.empty:
        print("\n*** BASELINE METHOD ***")
        print("hsv_histogram_concat:")
        method_data = df[df["method"] == "hsv_histogram_concat"]
        for map_metric in ["mAP@K=1", "mAP@K=5", "mAP@K=10"]:
            metric_data = method_data[method_data["map_metric"] == map_metric]
            if not metric_data.empty:
                best_idx = metric_data["score"].idxmax()
                best = metric_data.loc[best_idx]
                print(
                    f"  {map_metric}: {best['score']:.4f} | Config: {best['params']} | Dist: {best['distance_function']}"
                )

    # Print texture descriptors
    print("\n*** TEXTURE DESCRIPTORS ***")
    for _, method_row in best_per_method_map10.iterrows():
        method = method_row["method"]
        method_data = df[df["method"] == method]

        print(f"\n{method}:")

        # Find best config for each metric
        for map_metric in ["mAP@K=1", "mAP@K=5", "mAP@K=10"]:
            metric_data = method_data[method_data["map_metric"] == map_metric]
            if not metric_data.empty:
                best_idx = metric_data["score"].idxmax()
                best = metric_data.loc[best_idx]
                print(
                    f"  {map_metric}: {best['score']:.4f} | Config: {best['params']} | Dist: {best['distance_function']}"
                )

    # Top 10 overall (based on mAP@10)
    print("\n" + "=" * 80)
    print("TOP 10 CONFIGURATIONS (Based on mAP@K=10)")
    print("=" * 80)

    top10 = df_map10.nlargest(10, "score")
    for i, (_, row) in enumerate(top10.iterrows(), 1):
        baseline_marker = (
            " *** BASELINE ***" if row["method"] == "hsv_histogram_concat" else ""
        )
        print(f"\n{i}. {row['method']}{baseline_marker}")
        print(f"   Config: {row['params']}")
        print(f"   Distance: {row['distance_function']}")
        print(f"   mAP@K=10: {row['score']:.4f}")

        # Show other metrics for the same configuration
        same_config = df[
            (df["method"] == row["method"])
            & (df["params"] == row["params"])
            & (df["distance_function"] == row["distance_function"])
            & (df["map_metric"] != "mAP@K=10")
        ]
        for _, other_row in same_config.iterrows():
            print(f"   {other_row['map_metric']}: {other_row['score']:.4f}")

    # Comparison summary
    print("\n" + "=" * 80)
    print("TEXTURE DESCRIPTORS vs BASELINE COMPARISON")
    print("=" * 80)

    if not baseline_data.empty:
        baseline_score = baseline_data["score"].max()
        print(f"\nBaseline (hsv_histogram_concat) mAP@K=10: {baseline_score:.4f}")
        print("\nTexture Descriptor Improvements:")

        for _, method_row in best_per_method_map10.iterrows():
            method = method_row["method"]
            method_score = method_row["score"]
            improvement = method_score - baseline_score
            improvement_pct = (improvement / baseline_score) * 100

            symbol = "↑" if improvement > 0 else "↓" if improvement < 0 else "="
            print(
                f"  {method:20s}: {method_score:.4f} ({symbol} {improvement:+.4f}, {improvement_pct:+.2f}%)"
            )


def plot_parameter_sensitivity(df, save_plots=True, save_dir: Path = Path("./plots")):
    """This function is deprecated - parameter effects are now shown in plot_method_comparison"""
    pass


def main(results_dir: str = "./df_results/qsd1_w3"):
    """
    Main function to visualize results from test_weekn_weekm function.

    Args:
        results_dir: Path to the results directory containing .pkl files from test_weekn_weekm
                     Default: "./df_results/qsd1_w3" (matching test_weekn_weekm output)
    """
    # Load all results
    print("=" * 80)
    print(f"Loading results from: {results_dir}")
    print("=" * 80)

    all_results = load_all_results(results_dir)

    if not all_results:
        print("No results found!")
        print(
            "\nMake sure you've run the test_weekn_weekm() function from main.py first!"
        )
        print("This will generate .pkl files in ./df_results/preprocess_resize_256/")
        return None

    print(f"Loaded {len(all_results)} result files")

    # Convert to DataFrame
    df = create_comparison_dataframe(all_results)
    print(f"Created DataFrame with {len(df)} rows")

    # Extract results_dir name for plot directory
    results_dir_name = Path(results_dir).name
    save_dir = Path("./plots") / results_dir_name

    # Create visualizations
    print("\nCreating comparison plots...")
    plot_method_comparison(df, save_plots=True, save_dir=save_dir)

    # Print best configurations
    print_best_configurations(df)

    # Print summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    summary = df.groupby(["method", "map_metric"])["score"].agg(
        ["count", "mean", "std", "min", "max"]
    )
    print(summary)

    print("\n" + "=" * 80)
    print(f"Plots saved to: {save_dir.absolute()}")
    print("=" * 80)

    return df


if __name__ == "__main__":
    # Default results directory from test_weekn_weekm function
    results_dir = "./df_results/preprocess_resize"

    # You can also analyze other result directories:
    # results_dir = "./df_results/preprocess_resize_256"
    # results_dir = "./df_results/qsd2_w2"

    df = main(results_dir)
