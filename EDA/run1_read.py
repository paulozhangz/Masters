import bioread
import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk
import pandas as pd
from pathlib import Path
from scipy.stats import shapiro
import seaborn as sns


def list_pairs(root_dir):
    root = Path(root_dir)
    pairs = []

    for subject_folder in root.iterdir():
        if not subject_folder.is_dir():
            continue

        name = subject_folder.name.upper()
        if not (name.startswith("C") or name.startswith("S")):
            continue

        acq = next(subject_folder.glob(f"*run1*.acq"), None)
        txt = next(subject_folder.glob(f"*run1*.txt"), None)

        if acq and txt:
            pairs.append((subject_folder.name, acq, txt))

    return pairs

def safe_read_acq(filepath):
    """Safely read .acq file."""
    try:
        return bioread.read_file(filepath)
    except Exception as e:
        print(f"[ERROR] Failed to read {filepath}: {e}")
        return None

def parse_log_file(filepath):
    """Parse .txt log into labels and event times."""
    labels, events = [], []

    with open(filepath) as f:
        for _ in range(13):
            next(f)

        for line in f:
            parts = line.split()
            labels.append(parts[0])
            events.append(float(parts[-1]))

    return labels, events

def extract_eda_features(signal, fs, events, labels,
                         dicom, subj, run):

    dicom_count = dicom.get((subj, f"Run{run}"))

    if dicom_count is None:
        print(f"No DicomCount found for {subj} Run{run}")

    else:
        signal_duration = len(signal) / fs

        if signal_duration > dicom_count:

            excess_seconds = signal_duration - dicom_count
            excess_samples = int(round(excess_seconds * fs))

            print(
                f"{subj} Run{run}: trimming "
                f"{excess_samples} samples "
                f"({excess_seconds:.2f} s)"
            )

            signal = signal[excess_samples:]

        elif signal_duration < dicom_count:

            print(
                f"{subj} Run{run}: signal shorter "
                f"({signal_duration:.2f} s) than DicomCount "
                f"({dicom_count})"
            )

    eda, _ = nk.eda_process(signal, sampling_rate=fs)

    eda_phasic = eda["EDA_Phasic"]
    peak_indices = np.where(eda["SCR_Peaks"] == 1)[0]
    recording_end = len(eda_phasic) / fs

    amplitudes, latencys, types = [], [], []
    event_indices_used = []  # NEW

    for i in range(len(events)):

        if i < len(events) - 1:
            start_time = events[i]
            end_time = events[i + 1]
        else:
            # last event goes until recording end
            start_time = events[i]
            end_time = recording_end

        index = []

        for idx in peak_indices:
            t = idx / fs
            if start_time <= t <= end_time:
                index.append(idx)

        if not index:
            continue

        values = eda_phasic.iloc[index].values
        max_idx = np.argmax(values)

        peak_index = index[max_idx]
        peak_value = values[max_idx]

        amplitudes.append(round(peak_value, 4))
        latencys.append(round(peak_index / fs - events[i], 4))
        types.append(labels[i])

        event_indices_used.append(i)  # 🔑 track alignment

    return amplitudes, latencys, types, eda, event_indices_used

def plot_eda(eda, events, labels, peaks, amplitudes, fs, filename):
    """Plot EDA with peaks and events."""
    plt.figure(figsize=(12, 5))
    plt.plot(eda.index / fs, eda["EDA_Phasic"], label="EDA")
    used_labels = set()

    for e, label in zip(events, labels):
        if label == "rest":
            color = "red"
        elif label == "grasp":
            color= "green"
        else:
            color = "blue"

        legend_label = label if label not in used_labels else "_nolegend_"
        used_labels.add(label)
        plt.axvline(e, color=color, linestyle="--", label=legend_label)

    peak_times = [p / fs for p in peaks]
    plt.scatter(peak_times, amplitudes, color="red", label="Peaks")

    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("EDA (µS)")
    plt.title("EDA Analysis")
    plt.savefig(filename)
    plt.close()

# -----------------------------
# Main pipeline
# -----------------------------


def main(folder, plot_first=False,plot_all=False):

    dataset = list_pairs(folder)
    dicom = {
        (row.Subject, row.Run): row.DicomCount
        for _, row in pd.read_csv(r"C:\Users\paulo\PycharmProjects\code\Ze_kitchen\max_dicom_per_run.csv").iterrows()
    }


    results = {
        "names": [],
        "amplitudes": [],
        "latencys": [],
        "types": []
    }

    plotted = False

    for i in range(len(dataset)):

        print(f"Processing: {dataset[i][0]}")

        data = safe_read_acq(dataset[i][1])
        if data is None:
            continue

        labels, events = parse_log_file(dataset[i][2])

        channel = data.channels[0]
        signal = channel.data
        fs = channel.samples_per_second

        amps, ints, types, eda, event_used = extract_eda_features(
            signal,
            fs,
            events,
            labels,
            dicom,
            dataset[i][0],  # subject name
            1  # run number
        )

        results["names"].append(dataset[i][0])
        results["amplitudes"].append(amps)
        results["latencys"].append(ints)
        results["types"].append(types)

        # Plot only first valid file
        if plot_first and not plotted and amps:
            peak_indices_reconstructed = [
                int((events[i] + ints[j]) * fs)
                for j, i in enumerate(event_used)
            ]
            plot_eda(eda, events, labels, peak_indices_reconstructed, amps, fs, f"{dataset[i][0]}.png")
            plotted = True
        if plot_all:
            peak_indices_reconstructed = [
                int((events[i] + ints[j]) * fs)
                for j, i in enumerate(event_used)
            ]
            plot_eda(eda, events, labels, peak_indices_reconstructed, amps, fs, f"{dataset[i][0]}.png")
            plotted = True

    df = pd.DataFrame(results)

    df = df.explode(['amplitudes', 'latencys', 'types'])

    df = df.rename(columns={
        'amplitudes': 'amplitude',
        'latencys': 'latency',
        'types': 'type'
    })

    df["group"] = df["names"].str[0]

    df["amplitude"] = pd.to_numeric(df["amplitude"], errors="coerce")
    df["latency"] = pd.to_numeric(df["latency"], errors="coerce")

    return  df

root1 = r"C:\Users\paulo\PycharmProjects\code\Paulo"

df1=main(root1,plot_first=True)
df1.to_csv("run1.csv",index=False)




"""
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
histograms_run1(df1)
descriptive_stats(df1,export_latex=True)
df_to_latex(normality_table(df1))
df_to_latex(mannwhitney_table(df1))
"""
