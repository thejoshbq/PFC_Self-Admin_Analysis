# main.py

import os
import sqlite3
import io
import numpy as np
from tqdm import tqdm
from nsync2p.sample import NSyncSample
from nsync2p.population import NSyncPopulation

def array_to_blob(arr) -> bytes:
    """
    Serialize a numpy array into a binary blob using np.save.

    Args:
        arr (np.ndarray): The array to serialize.

    Returns:
        bytes: The binary blob.
    """
    bio = io.BytesIO()
    np.save(bio, arr)
    return bio.getvalue()

def get_or_insert_day(cur, label):
    """
    Get the day_id for a given label, inserting if it doesn't exist.

    Args:
        cur: SQLite cursor.
        label (str): The day label.

    Returns:
        int: The day_id.
    """
    cur.execute("SELECT day_id FROM Days WHERE label=?", (label,))
    row = cur.fetchone()
    if row:
        return row[0]
    cur.execute("INSERT INTO Days (label) VALUES (?)", (label,))
    return cur.lastrowid

def get_or_insert_animal(cur, label):
    """
    Get the animal_id for a given label, inserting if it doesn't exist.

    Args:
        cur: SQLite cursor.
        label (str): The animal label.

    Returns:
        int: The animal_id.
    """
    cur.execute("SELECT animal_id FROM Animals WHERE label=?", (label,))
    row = cur.fetchone()
    if row:
        return row[0]
    cur.execute("INSERT INTO Animals (label) VALUES (?)", (label,))
    return cur.lastrowid

def insert_fov(cur, label, animal_id, day_id):
    """
    Insert a new FOV and return its ID.

    Args:
        cur: SQLite cursor.
        label (str): FOV label.
        animal_id (int): Associated animal ID.
        day_id (int): Associated day ID.

    Returns:
        int: The fov_id.
    """
    cur.execute("INSERT INTO FOVs (label, animal_id, day_id) VALUES (?,?,?)", (label, animal_id, day_id))
    return cur.lastrowid

def setup_database(conn, cur) -> None:
    """
    Drop existing tables and create the database schema.

    Args:
        conn: SQLite connection.
        cur: SQLite cursor.
    """
    print("Clearing existing tables...")
    cur.execute("DROP TABLE IF EXISTS EventWindows")
    cur.execute("DROP TABLE IF EXISTS EventLog")
    cur.execute("DROP TABLE IF EXISTS ExtractedSignals")
    cur.execute("DROP TABLE IF EXISTS FOVs")
    cur.execute("DROP TABLE IF EXISTS Animals")
    cur.execute("DROP TABLE IF EXISTS Days")
    conn.commit()

    print("Creating tables...")
    cur.execute("""
                CREATE TABLE IF NOT EXISTS Days
                (
                    day_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    label  TEXT UNIQUE
                )
                """)
    cur.execute("""
                CREATE TABLE IF NOT EXISTS Animals
                (
                    animal_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    label     TEXT UNIQUE
                )
                """)
    cur.execute("""
                CREATE TABLE IF NOT EXISTS FOVs
                (
                    fov_id    INTEGER PRIMARY KEY AUTOINCREMENT,
                    label     TEXT,
                    animal_id INTEGER,
                    day_id    INTEGER,
                    FOREIGN KEY (animal_id) REFERENCES Animals (animal_id),
                    FOREIGN KEY (day_id) REFERENCES Days (day_id)
                )
                """)
    cur.execute("""
                CREATE TABLE IF NOT EXISTS ExtractedSignals
                (
                    neuron_id  INTEGER PRIMARY KEY AUTOINCREMENT,
                    cluster_id INTEGER,
                    fov_id     INTEGER,
                    blob       BLOB,
                    FOREIGN KEY (fov_id) REFERENCES FOVs (fov_id)
                )
                """)
    cur.execute("""
                CREATE TABLE IF NOT EXISTS EventLog
                (
                    event_id  INTEGER PRIMARY KEY AUTOINCREMENT,
                    type      INTEGER,
                    timestamp REAL,
                    fov_id    INTEGER,
                    FOREIGN KEY (fov_id) REFERENCES FOVs (fov_id)
                )
                """)
    cur.execute("""
                CREATE TABLE IF NOT EXISTS EventWindows
                (
                    window_id  INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id   INTEGER,
                    neuron_id  INTEGER,
                    cluster_id INTEGER,
                    fov_id     INTEGER,
                    blob       BLOB,
                    FOREIGN KEY (event_id) REFERENCES EventLog (event_id),
                    FOREIGN KEY (neuron_id) REFERENCES ExtractedSignals (neuron_id),
                    FOREIGN KEY (cluster_id) REFERENCES ExtractedSignals (cluster_id),
                    FOREIGN KEY (fov_id) REFERENCES FOVs (fov_id)
                )
                """)
    conn.commit()

def load_cluster_ids(root) -> np.ndarray:
    """
    Load the cluster IDs from the npy file.

    Args:
        root (str): Root data directory.

    Returns:
        np.ndarray: Array of cluster IDs.
    """
    return np.load(os.path.join(root, "cluster_ids.npy"))

def get_days(root) -> list:
    """
    Get the list of day directories from the root.

    Args:
        root (str): Root data directory.

    Returns:
        list: Sorted list of day names.
    """
    return [d for d in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, d))]

def collect_samples_for_day(root, day, day_id, cur) -> tuple[list, list]:
    """
    Collect NSyncSample instances for a day across animals and FOVs, inserting FOVs as needed.

    Args:
        root (str): Root data directory.
        day (str): Day name.
        day_id (int): Database day ID.
        cur: SQLite cursor.

    Returns:
        tuple: (list of samples, list of fov_ids)
    """
    day_path = os.path.join(root, day)
    animals = [a for a in sorted(os.listdir(day_path)) if os.path.isdir(os.path.join(day_path, a))]

    samples = []
    fov_ids = []

    for animal in animals:
        animal_id = get_or_insert_animal(cur, animal)
        animal_path = os.path.join(day_path, animal)
        fovs = [f for f in sorted(os.listdir(animal_path)) if os.path.isdir(os.path.join(animal_path, f))]

        for fov in fovs:
            fov_path = os.path.join(animal_path, fov)

            extracted_signals_files = sorted([
                os.path.join(fov_path, ff) for ff in os.listdir(fov_path)
                if ff.endswith(".npy") and "extractedsignals_raw" in ff
            ])
            behavior_event_log_files = sorted([
                os.path.join(fov_path, ff) for ff in os.listdir(fov_path)
                if ff.endswith(".mat")
            ])

            try:
                dataset = NSyncSample(
                    eventlog=behavior_event_log_files,
                    extracted_signals=extracted_signals_files,
                    frame_correction=os.path.join(root, 'empty.mat'),
                    animal_name=animal,
                    target_id=[22, 222],
                    isolate_events=True,
                    min_events=2,
                    normalize=False,
                )
                dataset_windows = dataset.get_event_windows()
                if dataset_windows.ndim == 3 and dataset_windows.size > 0:
                    fov_id = insert_fov(cur, fov, animal_id, day_id)
                    samples.append(dataset)
                    fov_ids.append(fov_id)
                else:
                    print(f"Skipping {fov_path} due to invalid data")
            except Exception as e:
                print(f"Error processing {fov_path}: {e}")
                continue

    return samples, fov_ids

def create_population(samples, day) -> NSyncPopulation:
    """
    Create an NSyncPopulation from the list of samples.

    Args:
        samples (list): List of NSyncSample instances.
        day (str): Day name.

    Returns:
        NSyncPopulation: The population object.
    """
    return NSyncPopulation(
        samples,
        name=day[2:],
        subtract_baseline=False,
        z_score_baseline=False,
        compute_significance=False,
        bh_correction=False,
    )

def compute_valid_mask(population) -> np.ndarray:
    """
    Compute the mask for valid (non-NaN mean response) neurons.

    Args:
        population (NSyncPopulation): The population object.

    Returns:
        np.ndarray: Boolean mask.
    """
    mean_responses_pre = np.nanmean(population.per_neuron_means, axis=1)
    return ~np.isnan(mean_responses_pre)

def assign_clusters(population, cluster_ids, start_used, valid_mask):
    """
    Assign cluster IDs to valid neurons in the population samples.

    Args:
        population (NSyncPopulation): The population object.
        cluster_ids (np.ndarray): Array of available cluster IDs.
        start_used (int): Starting index in cluster_ids.
        valid_mask (np.ndarray): Validity mask for neurons.

    Returns:
        int: Updated used index in cluster_ids.
    """
    used_cluster_ids = start_used
    cumul_neuron_start = 0

    for sample in tqdm(population.get_used_samples(), desc="Assigning cluster IDs"):
        num_sample_neurons = sample.get_num_neurons()
        if num_sample_neurons == 0:
            cumul_neuron_start += num_sample_neurons
            continue

        sample_neuron_start = cumul_neuron_start
        sample_neuron_end = sample_neuron_start + num_sample_neurons

        sample_valid_mask = valid_mask[sample_neuron_start:sample_neuron_end]
        num_valid_in_sample = np.sum(sample_valid_mask)

        sample_cluster_ids = np.full(num_sample_neurons, -1, dtype=cluster_ids.dtype)

        valid_offset = 0
        for local_idx in range(num_sample_neurons):
            if sample_valid_mask[local_idx]:
                sample_cluster_ids[local_idx] = cluster_ids[used_cluster_ids + valid_offset]
                valid_offset += 1

        sample.set_cluster_ids(sample_cluster_ids)

        used_cluster_ids += num_valid_in_sample
        cumul_neuron_start = sample_neuron_end

    return used_cluster_ids

def filter_target_events(eventlog, event_id_map, event_ids_arr, event_timestamps_arr, min_events) -> list[int]:
    """
    Filter and process target events (types 22 and 222), removing close events.

    Args:
        eventlog: Sample eventlog.
        event_id_map (dict): Mapping of timestamps to event IDs.
        event_ids_arr (np.ndarray): Event IDs array.
        event_timestamps_arr (np.ndarray): Event timestamps array.
        min_events (int): Minimum number of events required.

    Returns:
        list: List of target event IDs, or empty if insufficient.
    """
    target_ids = [22, 222]
    target_mask = np.isin(event_ids_arr, target_ids)
    events_ts = event_timestamps_arr[target_mask]
    events_ts = np.sort(events_ts)

    if len(events_ts) < min_events:
        return []

    sep = 1000.0
    if True:  # Placeholder condition
        to_delete = np.argwhere(np.diff(events_ts) < sep) + 1
        events_ts = np.delete(events_ts, to_delete)

    if len(events_ts) < min_events:
        return []

    target_event_ids = []
    for valid_ts in events_ts:
        eid = event_id_map.get(float(valid_ts), None)
        if eid is not None:
            target_event_ids.append(eid)

    if len(target_event_ids) < min_events:
        return []

    return target_event_ids

def insert_data_for_day(population, fov_id_list, valid_mask, cur, conn) -> None:
    """
    Insert extracted signals, event logs, and event windows for the day's samples.

    Args:
        population (NSyncPopulation): The population object.
        fov_id_list (list): List of FOV IDs corresponding to samples.
        valid_mask (np.ndarray): Validity mask for neurons.
        cur: SQLite cursor.
        conn: SQLite connection.
    """
    cumul_neuron_start = 0

    for sample_idx, sample in enumerate(tqdm(population.get_used_samples(), desc="Inserting data")):
        fov_id = fov_id_list[sample_idx]
        num_sample_neurons = sample.get_num_neurons()
        if num_sample_neurons == 0:
            cumul_neuron_start += num_sample_neurons
            continue

        sample_neuron_start = cumul_neuron_start
        sample_neuron_end = sample_neuron_start + num_sample_neurons

        sample_valid_mask = valid_mask[sample_neuron_start:sample_neuron_end]
        valid_local_indices = np.where(sample_valid_mask)[0]
        num_valid_in_sample = len(valid_local_indices)

        signals = sample.get_extracted_signals()
        sample_cluster_ids = sample.get_cluster_ids()

        local_neuron_ids = []
        for v, local_idx in enumerate(valid_local_indices):
            signal_blob = array_to_blob(signals[local_idx])
            cluster_id_val = int(sample_cluster_ids[local_idx])
            cur.execute("INSERT INTO ExtractedSignals (cluster_id, fov_id, blob) VALUES (?, ?, ?)",
                        (cluster_id_val, fov_id, signal_blob))
            local_neuron_ids.append(cur.lastrowid)

        eventlog = sample.get_eventlog()
        event_id_map = {}
        for typ, ts in eventlog:
            cur.execute("INSERT INTO EventLog (type, timestamp, fov_id) VALUES (?, ?, ?)",
                        (int(typ), float(ts), fov_id))
            eid = cur.lastrowid
            event_id_map[float(ts)] = eid

        conn.commit()

        event_ids_arr = sample.get_event_ids()
        event_timestamps_arr = sample.get_event_timestamps()
        min_events = sample.get_min_events()

        target_event_ids = filter_target_events(
            eventlog, event_id_map, event_ids_arr, event_timestamps_arr, min_events
        )
        if not target_event_ids:
            cumul_neuron_start = sample_neuron_end
            continue

        windows = sample.get_event_windows()
        if windows.ndim != 3 or windows.shape[2] == 0:
            cumul_neuron_start = sample_neuron_end
            continue

        num_trials = windows.shape[2]
        num_trials = min(num_trials, len(target_event_ids))

        for i in range(num_trials):
            event_id = target_event_ids[i]
            for v, local_idx in enumerate(valid_local_indices):
                window_arr = windows[local_idx, :, i]
                window_blob = array_to_blob(window_arr)
                neuron_id = local_neuron_ids[v]
                cluster_id_val = int(sample_cluster_ids[local_idx])
                cur.execute(
                    "INSERT INTO EventWindows (event_id, neuron_id, cluster_id, fov_id, blob) VALUES (?, ?, ?, ?, ?)",
                    (event_id, neuron_id, cluster_id_val, fov_id, window_blob))

        cumul_neuron_start = sample_neuron_end
        conn.commit()

def process_day(root, day, cur, conn, cluster_ids, used_cluster_ids):
    """
    Process a single day: collect samples, create population, assign clusters, insert data.

    Args:
        root (str): Root data directory.
        day (str): Day name.
        cur: SQLite cursor.
        conn: SQLite connection.
        cluster_ids (np.ndarray): Cluster IDs array.
        used_cluster_ids (int): Current used index.

    Returns:
        int: Updated used_cluster_ids.
    """
    day_id = get_or_insert_day(cur, day)
    samples, fov_id_list = collect_samples_for_day(root, day, day_id, cur)

    if not samples:
        print(f"No valid samples for day {day}")
        return used_cluster_ids

    population = create_population(samples, day)

    if population.per_neuron_means.size == 0:
        print(f"No per_neuron_means for day {day}")
        return used_cluster_ids

    valid_mask_full = compute_valid_mask(population)

    used_cluster_ids = assign_clusters(population, cluster_ids, used_cluster_ids, valid_mask_full)

    insert_data_for_day(population, fov_id_list, valid_mask_full, cur, conn)

    return used_cluster_ids

def main() -> None:
    """
    Main function to orchestrate the data processing pipeline.
    """
    root = "./data"
    db_path = "./output/PFC_Self-Admin.db"
    os.makedirs("./output", exist_ok=True)

    cluster_ids = load_cluster_ids(root)
    used_cluster_ids = 0

    print("Initializing database...")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    try:
        setup_database(conn, cur)

        print("Processing data...")
        days = get_days(root)

        for day in days:
            used_cluster_ids = process_day(root, day, cur, conn, cluster_ids, used_cluster_ids)

    finally:
        conn.close()

if __name__ == "__main__":
    main()