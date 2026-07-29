import bioread
import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk
import pandas as pd
from pathlib import Path
from scipy.stats import shapiro, mannwhitneyu
import seaborn as sns
from scipy.stats import wilcoxon

def list_pairs(root_dir):
    root = Path(root_dir)
    pairs = []

    for subject_folder in root.iterdir():
        if not subject_folder.is_dir():
            continue

        name = subject_folder.name.upper()
        if not (name.startswith("C") or name.startswith("S")):
            continue

        acq = next(subject_folder.glob(f"*run3*.acq"), None)
        txt = next(subject_folder.glob(f"*run3*.txt"), None)

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
    types = []
    events = []
    rx_time = []
    correct = []
    with open(filepath) as f:

        for _ in range(15):
            next(f)

        for i in f:
            if int(i.split()[0]) == 0:
                types.append(int(i.strip().split("\t")[0]))
                events.append(float(i.strip().split("\t")[2]))
                rx_time.append(None)
                correct.append(None)
            elif i.strip().split("\t")[5] == ' time_out ':
                continue
            else:
                types.append(int(i.strip().split("\t")[0]))
                events.append(float(i.strip().split("\t")[2]))
                rx_time.append(float(i.strip().split("\t")[5]))
                correct.append(float(i.strip().split("\t")[6]))

    if events:
        events.append(events[-1] + 20)
        types.append(0)
    return types, events, rx_time, correct

def extract_eda_features(signal, fs, events, labels, rx_time, correct, dicom, subj):
    run=3

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
    """Extract peak amplitudes and intervals."""
    eda, _ = nk.eda_process(signal, sampling_rate=fs)

    eda_phasic = eda["EDA_Phasic"]
    peak_indices = np.where(eda["SCR_Peaks"] == 1)[0]

    amplitudes, latencys, types, rx, corr = [], [], [], [], []
    event_indices_used = []  # NEW

    for i in range(len(events) - 1):  # Each interval of events

        index = []

        for idx in peak_indices:
            t = idx / fs
            if events[i] <= t <= events[i + 1]:
                index.append(idx)

        if not index:
            continue
        elif labels[i] == 0:
            continue
        else:
            values = eda_phasic.iloc[index].values
            max_idx = np.argmax(values)

            peak_index = index[max_idx]
            peak_value = values[max_idx]
            amplitudes.append(round(peak_value, 4))
            latencys.append(round(peak_index / fs - events[i], 4))
            types.append(labels[i])
            rx.append(rx_time[i])
            corr.append(correct[i])

        event_indices_used.append(i)  # 🔑 track alignment

    return amplitudes, latencys, types, eda, event_indices_used, rx, corr

def plot_eda(eda, events, labels, peaks, amplitudes, fs, filename):
    """Plot EDA with peaks and events."""
    plt.figure(figsize=(12, 5))
    plt.plot(eda.index / fs, eda["EDA_Phasic"], label="EDA")

    for e, label in zip(events, labels):
        if label == 0:
            color = "red"
        elif label == 1:
            color = "green"
        else:
            color = "blue"
        plt.axvline(e, color=color, linestyle="--")

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

def main(folder, plot_first=True, plot_all=False):

    dataset = list_pairs(folder)
    dicom = {
        (row.Subject, row.Run): row.DicomCount
        for _, row in pd.read_csv(r"/code/Ze_kitchen/max_dicom_per_run.csv").iterrows()
    }

    results = {
        "names": [],
        "amplitudes": [],
        "latencys": [],
        "typef": [],
        "rxf": [],
        "correctf": []
    }

    plotted = False

    for i in range(len(dataset)):

        print(f"Processing: {dataset[i][0]}")

        data = safe_read_acq(dataset[i][1])
        if data is None:
            continue

        labels, events, rx_time, correct = parse_log_file(dataset[i][2])

        channel = data.channels[0]
        signal = channel.data
        fs = channel.samples_per_second

        amps, ints, types, eda, event_used, rx, corr = extract_eda_features(signal, fs, events, labels, rx_time, correct, dicom, dataset[i][0])

        results["names"].append(dataset[i][0])
        results["amplitudes"].append(amps)
        results["latencys"].append(ints)
        results["typef"].append(types)
        results["rxf"].append(rx)
        results["correctf"].append(corr)

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

    return results

# -----------------------------
# Run
# -----------------------------

def prepare_dataframe(results):

    df = pd.DataFrame(results)

    df = df.explode([
        'amplitudes',
        'latencys',
        'typef',
        'rxf',
        'correctf'
    ])

    df = df.rename(columns={
        'amplitudes': 'amplitude',
        'latencys': 'latency',
        'typef': 'type',
        'rxf': 'rx_time',
        'correctf': 'correct'
    })

    # -----------------------
    # Convert groups
    # -----------------------

    df["group"] = df["names"].str[0]

    # -----------------------
    # Convert types
    # -----------------------

    type_map = {
        1: "easy",
        2: "hard"
    }

    df["type"] = pd.to_numeric(
        df["type"],
        errors="coerce"
    )

    df["type"] = df["type"].map(type_map)

    # -----------------------
    # Numeric conversion
    # -----------------------

    numeric_cols = [
        "amplitude",
        "latency",
        "rx_time",
        "correct"
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

root1 = r"C:\Users\paulo\PycharmProjects\code\Paulo"

df1=prepare_dataframe(main(root1,plot_first=False))
df1.to_csv("run3.csv",index=False)

"""
descriptive_stats(df1, export_latex=True)
histograms(df1)
normality_table(df1)
boxplots(df1)
mannwhitney_groups(df1)
wilcoxon_easy_vs_hard(df1)
amplitude_correct_vs_wrong(df1)
correctness_boxplot(df1)
latency_vs_rx_wilcoxon(df1)
boxplot_latency_vs_rxtime(df1)
"""