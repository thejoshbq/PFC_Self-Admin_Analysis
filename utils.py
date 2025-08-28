# utils.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
from tqdm import tqdm

def blob_to_array(blob) -> np.ndarray:
    """
    Deserialize a binary blob into a numpy array using np.load.

    Args:
        blob (bytes): The binary data from the database.

    Returns:
        np.ndarray: The loaded numpy array.
    """
    bio = io.BytesIO(blob)
    return np.load(bio)

def safe_int(val) -> int:
    """
    Safely convert a value to an integer, handling NaN, bytes, and conversion errors.

    Args:
        val: The value to convert (could be NaN, bytes, int, str, etc.).

    Returns:
        int: The integer value or -1 on failure.
    """
    if pd.isna(val):
        return -1
    if isinstance(val, bytes):
        try:
            return int.from_bytes(val, 'little', signed=False)
        except (ValueError, OverflowError):
            return -1
    try:
        return int(val)
    except (ValueError, TypeError):
        return -1

def query_windows_data(conn, day_id) -> tuple[pd.DataFrame, int]:
    """
    Query the EventWindows data for a specific day, joining with FOVs.

    Args:
        conn: SQLite connection object.
        day_id (int): The ID of the day to query.

    Returns:
        pd.DataFrame: DataFrame with neuron_id, blob, and cluster_id columns.
        int: Total number of windows.
    """
    cur = conn.cursor()
    cur.execute("""
                SELECT COUNT(*)
                FROM EventWindows w
                         JOIN FOVs f ON w.fov_id = f.fov_id
                WHERE f.day_id = ?
                """, (day_id,))
    total_windows = cur.fetchone()[0]

    if total_windows == 0:
        return pd.DataFrame(), 0

    windows_df = pd.read_sql_query("""
                                   SELECT w.neuron_id, w.blob, w.cluster_id
                                   FROM EventWindows w
                                            JOIN FOVs f ON w.fov_id = f.fov_id
                                   WHERE f.day_id = ?
                                   """, conn, params=(day_id,))

    if windows_df.empty:
        return windows_df, 0

    return windows_df, total_windows

def compute_per_neuron_means(windows_df) -> tuple[np.ndarray, np.ndarray, pd.Series]:
    """
    Deserialize blobs and compute mean traces per neuron.

    Args:
        windows_df (pd.DataFrame): DataFrame with 'neuron_id' and 'blob' columns.

    Returns:
        np.ndarray: Stacked array of mean traces per neuron.
        np.ndarray: Neuron IDs.
        pd.Series: First cluster_id per neuron.
    """
    windows_df['array'] = windows_df['blob'].apply(blob_to_array)
    grouped = windows_df.groupby('neuron_id')['array'].apply(
        lambda x: np.nanmean(np.stack(x), axis=0) if len(x) > 0 else None
    ).dropna()

    if grouped.empty:
        raise ValueError("No valid per-neuron means computed.")

    per_neuron_means = np.stack(grouped.values)
    neuron_ids = grouped.index.values
    neuron_clusters_df = windows_df.groupby('neuron_id')['cluster_id'].first()

    return per_neuron_means, neuron_ids, neuron_clusters_df

def filter_by_valid_clusters(per_neuron_means, neuron_ids, neuron_clusters_df) -> tuple[np.ndarray, np.ndarray]:
    """
    Filter neurons to those with valid (non-negative) cluster IDs.

    Args:
        per_neuron_means (np.ndarray): Array of per-neuron traces.
        neuron_ids (np.ndarray): Array of neuron IDs.
        neuron_clusters_df (pd.Series): Cluster IDs per neuron.

    Returns:
        tuple: Filtered per_neuron_means, clusters.
    """
    clusters = np.array([safe_int(neuron_clusters_df.get(nid, -1)) for nid in neuron_ids])
    valid_cluster_mask = clusters >= 0

    if not np.any(valid_cluster_mask):
        raise ValueError("No neurons with valid clusters.")

    return per_neuron_means[valid_cluster_mask], clusters[valid_cluster_mask]

def normalize_traces(per_neuron_means, baseline_range) -> np.ndarray:
    """
    Normalize per-neuron traces by subtracting baseline mean and dividing by baseline std.

    Args:
        per_neuron_means (np.ndarray): Array of per-neuron traces.
        baseline_range (np.ndarray): Indices for baseline period.

    Returns:
        np.ndarray: Normalized traces.
    """
    baseline = np.nanmean(per_neuron_means[:, baseline_range], axis=1)[:, np.newaxis]
    per_neuron_means -= baseline

    baseline_stds = np.nanstd(per_neuron_means[:, baseline_range], axis=1)[:, np.newaxis]
    per_neuron_means = np.divide(
        per_neuron_means,
        baseline_stds,
        out=np.zeros_like(per_neuron_means),
        where=baseline_stds > 0
    )

    return per_neuron_means

def filter_valid_responses(per_neuron_means, valid_clusters) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Filter out neurons with NaN mean responses and update clusters.

    Args:
        per_neuron_means (np.ndarray): Normalized traces.
        valid_clusters (np.ndarray): Cluster IDs for neurons.

    Returns:
        tuple: Filtered traces, mean responses, clusters, number of valid neurons.
    """
    mean_responses = np.nanmean(per_neuron_means, axis=1)
    valid_mask = ~np.isnan(mean_responses)

    if not np.any(valid_mask):
        raise ValueError("No valid responses after filtering.")

    filtered_traces = per_neuron_means[valid_mask]
    filtered_means = mean_responses[valid_mask]
    filtered_clusters = valid_clusters[valid_mask]
    num_valid = filtered_traces.shape[0]

    return filtered_traces, filtered_means, filtered_clusters, num_valid

def compute_population_stats(per_neuron_means) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute population-level mean, SEM-based CI, and median traces.

    Args:
        per_neuron_means (np.ndarray): Filtered and normalized traces.

    Returns:
        tuple: (pop_mean, pop_upper, pop_lower, pop_median)
    """
    num_n = per_neuron_means.shape[0]
    pop_mean = np.nanmean(per_neuron_means, axis=0)
    pop_std = np.nanstd(per_neuron_means, axis=0)
    pop_sem = pop_std / np.sqrt(num_n)
    pop_upper = pop_mean + 1.96 * pop_sem
    pop_lower = pop_mean - 1.96 * pop_sem
    pop_median = np.nanmedian(per_neuron_means, axis=0)

    return pop_mean, pop_upper, pop_lower, pop_median

def sort_traces_by_response(per_neuron_means, mean_responses) -> np.ndarray:
    """
    Sort neuron traces by descending mean response.

    Args:
        per_neuron_means (np.ndarray): Traces to sort.
        mean_responses (np.ndarray): Mean responses for sorting.

    Returns:
        np.ndarray: Sorted traces.
    """
    sort_idx = np.argsort(mean_responses)[::-1]
    return per_neuron_means[sort_idx]

def compute_cluster_stats(per_neuron_means, valid_clusters) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Compute mean traces and CI for each cluster.

    Args:
        per_neuron_means (np.ndarray): Filtered traces.
        valid_clusters (np.ndarray): Cluster IDs.

    Returns:
        list: List of tuples (means, uppers, lowers) for each cluster (0-8), None if empty.
    """
    cluster_data = [None] * 9
    available_clusters = np.unique(valid_clusters)

    for c in available_clusters:
        mask = valid_clusters == c
        if np.any(mask):
            num_n = np.sum(mask)
            means = np.nanmean(per_neuron_means[mask], axis=0)
            stds = np.nanstd(per_neuron_means[mask], axis=0)
            sems = stds / np.sqrt(num_n)
            uppers = means + 1.96 * sems
            lowers = means - 1.96 * sems
            cluster_data[int(c)] = (means, uppers, lowers)

    return cluster_data

def process_day_data(conn, day_id) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray, np.ndarray]], tuple, int]:
    """
    Main function to process data for a single day: query, compute means, normalize, filter, and stats.

    Args:
        conn: SQLite connection.
        day_id (int): Day ID to process.

    Returns:
        tuple: (sorted_temp, cluster_data, pop_data, num_valid_neurons) or (None, None, None, 0) on failure.
    """
    frame_rate = 30
    frame_averaging = 4
    sampling_rate = frame_rate / frame_averaging
    baseline_range = np.arange(int(3 * sampling_rate))

    try:
        windows_df, total_windows = query_windows_data(conn, day_id)
        if total_windows == 0 or windows_df.empty:
            return None, None, None, 0

        per_neuron_means, neuron_ids, neuron_clusters_df = compute_per_neuron_means(windows_df)
        per_neuron_means, clusters = filter_by_valid_clusters(per_neuron_means, neuron_ids, neuron_clusters_df)
        per_neuron_means = normalize_traces(per_neuron_means, baseline_range)
        per_neuron_means, mean_responses, valid_clusters, num_valid_neurons = filter_valid_responses(per_neuron_means,
                                                                                                     clusters)

        pop_mean, pop_upper, pop_lower, pop_median = compute_population_stats(per_neuron_means)
        sorted_temp = sort_traces_by_response(per_neuron_means, mean_responses)
        cluster_data = compute_cluster_stats(per_neuron_means, valid_clusters)

        return sorted_temp, cluster_data, (pop_mean, pop_upper, pop_lower, pop_median), num_valid_neurons

    except (ValueError, IndexError):
        return None, None, None, 0

def compute_global_limits(conn, day_ids) -> tuple[tuple, tuple]:
    """
    Compute global y-limits for population and cluster plots across all days.

    Args:
        conn: SQLite connection.
        day_ids (list): List of day IDs.

    Returns:
        tuple: (pop_ylim, cluster_ylim)
    """
    global_pop_min = float('inf')
    global_pop_max = float('-inf')
    global_cluster_min = float('inf')
    global_cluster_max = float('-inf')
    has_pop_data = False
    has_cluster_data = False

    for did in tqdm(day_ids, desc="Processing days for limits"):
        sorted_temp, cluster_data, pop_data, num_valid_neurons = process_day_data(conn, did)
        if num_valid_neurons > 0:
            has_pop_data = True
            pop_mean, pop_upper, pop_lower, pop_median = pop_data
            global_pop_min = min(global_pop_min, np.nanmin(pop_lower))
            global_pop_max = max(global_pop_max, np.nanmax(pop_upper))
            for c in range(9):
                if cluster_data[c] is not None:
                    means, uppers, lowers = cluster_data[c]
                    global_cluster_min = min(global_cluster_min, np.nanmin(lowers))
                    global_cluster_max = max(global_cluster_max, np.nanmax(uppers))
                    has_cluster_data = True

    # Compute y-limits with padding
    if has_pop_data:
        pop_range = global_pop_max - global_pop_min
        pop_pad = 0.1 * pop_range if pop_range > 0 else 0.5
        pop_ylim = (global_pop_min - pop_pad, global_pop_max + pop_pad)
    else:
        pop_ylim = (-0.5, 2.5)

    if has_cluster_data:
        cluster_range = global_cluster_max - global_cluster_min
        cluster_pad = 0.1 * cluster_range if cluster_range > 0 else 2.0
        cluster_ylim = (global_cluster_min - cluster_pad, global_cluster_max + cluster_pad)
    else:
        cluster_ylim = (-12, 12)

    return pop_ylim, cluster_ylim

def assign_cluster_colors(valid_cs, cluster_data) -> dict[int, str]:
    """
    Assign colors to clusters based on their overall mean responses (purple for positive, green for negative, black for neutral).

    Args:
        valid_cs (list): List of valid cluster IDs.
        cluster_data (list): Cluster stats data.

    Returns:
        dict: Cluster ID to color mapping.
    """
    cluster_colors = {}
    if valid_cs:
        overall_means = [np.nanmean(cluster_data[c][0]) for c in valid_cs]
        neutral_idx = np.argmin(np.abs(overall_means))  # Find neutral (smallest abs mean)
        neutral_c = valid_cs[neutral_idx]
        cluster_colors[neutral_c] = 'black'

        # Positives: sort by mean descending
        pos_indices = [i for i in range(len(valid_cs)) if i != neutral_idx and overall_means[i] > 0]
        pos_indices.sort(key=lambda i: overall_means[i], reverse=True)
        n_pos = len(pos_indices)
        if n_pos > 0:
            pos_shades = np.linspace(1.0, 0.6, n_pos)
            for j, i in enumerate(pos_indices):
                c = valid_cs[i]
                color = plt.cm.Purples(pos_shades[j])
                cluster_colors[c] = color

        # Negatives: sort by abs(mean) descending
        neg_indices = [i for i in range(len(valid_cs)) if i != neutral_idx and overall_means[i] < 0]
        neg_indices.sort(key=lambda i: abs(overall_means[i]), reverse=True)
        n_neg = len(neg_indices)
        if n_neg > 0:
            neg_shades = np.linspace(1.0, 0.6, n_neg)
            for j, i in enumerate(neg_indices):
                c = valid_cs[i]
                color = plt.cm.Greens(neg_shades[j])
                cluster_colors[c] = color
    return cluster_colors

def plot_day(ax1, ax2, ax3, sorted_temp, cluster_data, pop_data, num_valid_neurons, day_label, pop_ylim, cluster_ylim,
             idx) -> None:
    """
    Plot data for a single day across the three subplots.

    Args:
        ax1, ax2, ax3: Matplotlib axes for heatmap, population, and cluster plots.
        sorted_temp (np.ndarray): Sorted neuron traces for heatmap.
        cluster_data (list): Cluster stats.
        pop_data (tuple): Population stats.
        num_valid_neurons (int): Number of valid neurons.
        day_label (str): Day label for title.
        pop_ylim (tuple): Y-limits for population plot.
        cluster_ylim (tuple): Y-limits for cluster plot.
        idx (int): Column index for labels.
    """
    pop_mean, pop_upper, pop_lower, pop_median = pop_data

    # Heatmap subplot
    im = ax1.imshow(
        sorted_temp,
        cmap=plt.get_cmap('PRGn_r'),
        vmin=-4, vmax=4,
        aspect='auto'
    )
    ax1.set_title(f'{day_label[2:]} (n={num_valid_neurons})')
    ax1.axvline(x=75, color='k', linestyle='--', alpha=0.7, label='Event')
    ax1.tick_params(colors='k', labelbottom=False)

    # Population subplot
    ax2.set_ylim(pop_ylim)
    ax2.plot(pop_mean, label='Mean', color='k', linewidth=2)
    ax2.fill_between(range(len(pop_mean)), pop_lower, pop_upper, color='k', alpha=0.3)
    ax2.plot(pop_median, label='Median', color='r', linewidth=2)
    ax2.axvline(x=75, color='k', linestyle='--', alpha=0.7, label='Event')
    ax2.set_xlabel('Frames Relative to Event', color='k')
    ax2.grid(True, color='gray', alpha=0.3)
    ax2.tick_params(colors='k')
    ax2.set_xticks([0, 75, len(sorted_temp[0]) - 1])

    # Cluster subplot
    ax3.set_ylim(cluster_ylim)
    valid_cs = [c for c in range(9) if cluster_data[c] is not None]
    cluster_colors = assign_cluster_colors(valid_cs, cluster_data)

    for c in range(9):
        if cluster_data[c] is not None:
            mean_t, upper, lower = cluster_data[c]
            color = cluster_colors.get(c, 'gray')
            ax3.plot(mean_t, color=color, linewidth=2, label=f'Cluster {c}')
            ax3.fill_between(range(len(mean_t)), lower, upper, color=color, alpha=0.3)

    ax3.axvline(x=75, color='k', linestyle='--', alpha=0.7, label='Event')
    ax3.set_xlabel('Frames Relative to Event', color='k')
    ax3.grid(True, color='gray', alpha=0.3)
    ax3.tick_params(colors='k')
    ax3.set_xticks([0, 75, len(sorted_temp[0]) - 1])

    # Labels for first column
    if idx == 0:
        ax1.set_ylabel('Neurons', color='k')
        ax2.set_ylabel('Population Response', color='k')
        ax3.set_ylabel('Cluster Response', color='k')

def plot_no_data(axes, day_label, pop_ylim, cluster_ylim, idx) -> None:
    """
    Handle plotting for days with no data.

    Args:
        axes: Array of axes.
        day_label (str): Day label.
        pop_ylim (tuple): Population y-limits.
        cluster_ylim (tuple): Cluster y-limits.
        idx (int): Column index.
    """
    axes[0, idx].set_title(f'{day_label} (n=0)')
    axes[1, idx].set_title('No data')
    axes[2, idx].set_title('No data')
    axes[1, idx].set_ylim(pop_ylim)
    axes[2, idx].set_ylim(cluster_ylim)