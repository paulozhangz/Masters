import bioread
import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk
import pandas as pd
from scipy.stats import shapiro
from pathlib import Path
from scipy.stats import mannwhitneyu
import seaborn as sns
from scipy.stats import wilcoxon

# -----------------------------
# Utility functions
# -----------------------------

def list_pairs(root_dir,c):
    root = Path(root_dir)
    pairs = []

    for subject_folder in root.iterdir():
        if not subject_folder.is_dir():
            continue

        name = subject_folder.name.upper()
        if not (name.startswith("C") or name.startswith("S")):
            continue

        acq = next(subject_folder.glob(f"*run{c}*.acq"), None)
        txt = next(subject_folder.glob(f"*run{c}*.txt"), None)

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
        for _ in range(14):
            next(f)

        for line in f:
            parts = line.split()
            if len(parts) == 3:
                if parts[0]=="rest":
                    continue
                else:
                    if parts[0]=="???":
                        labels.append("Answer")
                    else:
                        labels.append(parts[0])
                    events.append(float(parts[2]))

            elif len(parts) == 6:
                labels.append(parts[3])
                events.append(float(parts[5]))

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

    amplitudes, intervals, types = [], [], []
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
        intervals.append(round(peak_index / fs - events[i], 4))
        types.append(labels[i])

        event_indices_used.append(i)  # 🔑 track alignment

    return amplitudes, intervals, types, eda, event_indices_used


def plot_eda(eda, events, labels, peaks, amplitudes, fs, filename):
    """Plot EDA with peaks and events."""
    plt.figure(figsize=(12, 5))
    plt.plot(eda.index / fs, eda["EDA_Phasic"], label="EDA")
    used_labels = set()

    for e, label in zip(events, labels):
        if label == "Answer":
            color = "red"
        elif label == "CountHeartBeats":
            color= "green"
        elif label == "CountVolumes":
            color = "blue"
        else:
            color = "yellow"
        legend_label = label if label not in used_labels else "_nolegend_"
        used_labels.add(label)
        plt.axvline(e, color=color, linestyle="--", label=legend_label)

    peak_times = [p / fs for p in peaks]
    plt.scatter(peak_times, amplitudes, color="red", label="Peaks")


    plt.xlabel("Time (s)")
    plt.ylabel("EDA (µS)")
    plt.title("EDA Analysis")
    plt.savefig(filename)
    plt.close()

# -----------------------------
# Main pipeline
# -----------------------------

def main(folder, k, plot_first=False,plot_all=False):

    dataset = list_pairs(folder,k)
    dicom = {
        (row.Subject, row.Run): row.DicomCount
        for _, row in pd.read_csv(r"/code/Ze_kitchen/max_dicom_per_run.csv").iterrows()
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
            2  # run number
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
            plot_eda(eda, events, labels, peak_indices_reconstructed, amps, fs, f"{dataset[i][0]}run{k}.png")
            plotted = True
        if plot_all:
            peak_indices_reconstructed = [
                int((events[i] + ints[j]) * fs)
                for j, i in enumerate(event_used)
            ]
            plot_eda(eda, events, labels, peak_indices_reconstructed, amps, fs, f"{dataset[i][0]}run{k}.png")

    df = pd.DataFrame(results)

    # explode safely
    df = df.explode(['amplitudes', 'latencys', 'types'], ignore_index=True)

    df = df.rename(columns={
        'amplitudes': 'amplitude',
        'latencys': 'latency',
        'types': 'type'
    })

    df["group"] = df["names"].str[0]

    # force numeric (critical!)
    df["amplitude"] = pd.to_numeric(df["amplitude"], errors="coerce")
    df["latency"] = pd.to_numeric(df["latency"], errors="coerce")

    # drop broken rows
    df = df.dropna(subset=["amplitude", "latency", "group"])

    return df


# -----------------------------
# Run
# -----------------------------

def normality_table(df1):

    # -----------------------------
    # Rename columns
    # -----------------------------

    df1 = df1.rename(columns={
        'amplitudes': 'amplitude',
        'latencys': 'latency',
        'types': 'type'
    })

    # -----------------------------
    # Create C/S group
    # -----------------------------

    df1["group"] = df1["names"].str[0].str.upper()

    results = []

    # -----------------------------
    # Group by C/S and task type
    # -----------------------------

    for (group_name, t), group in df1.groupby(['group', 'type']):

        amp = pd.to_numeric(
            group['amplitude'],
            errors='coerce'
        ).dropna()

        inter = pd.to_numeric(
            group['latency'],
            errors='coerce'
        ).dropna()

        # -----------------------------
        # Shapiro amplitude
        # -----------------------------

        if len(amp) >= 3:
            w_amp, p_amp = shapiro(amp)
        else:
            w_amp, p_amp = np.nan, np.nan

        # -----------------------------
        # Shapiro interval
        # -----------------------------

        if len(inter) >= 3:
            w_int, p_int = shapiro(inter)
        else:
            w_int, p_int = np.nan, np.nan

        results.append({

            'group': group_name,
            'type': t,

            'amp_W': round(w_amp, 3),
            'amp_p': f"{p_amp:.2e}",

            'lat_W': round(w_int, 3),
            'lat_p': f"{p_int:.2e}"
        })

    result_df = pd.DataFrame(results)

    print(result_df)

    return result_df

def mannwhitney_run2(df):

    results = []

    # Ensure group column exists
    df["group"] = df["names"].str[0].str.upper()

    for t in df["type"].unique():

        subset = df[df["type"] == t]

        # -----------------------------
        # AMPLITUDE
        # -----------------------------

        c_amp = pd.to_numeric(
            subset[subset["group"] == "C"]["amplitude"],
            errors='coerce'
        ).dropna()

        s_amp = pd.to_numeric(
            subset[subset["group"] == "S"]["amplitude"],
            errors='coerce'
        ).dropna()

        if len(c_amp) > 0 and len(s_amp) > 0:
            u_amp, p_amp = mannwhitneyu(c_amp, s_amp)

        else:
            u_amp, p_amp = None, None

        # -----------------------------
        # INTERVAL
        # -----------------------------

        c_int = pd.to_numeric(
            subset[subset["group"] == "C"]["latency"],
            errors='coerce'
        ).dropna()

        s_int = pd.to_numeric(
            subset[subset["group"] == "S"]["latency"],
            errors='coerce'
        ).dropna()

        if len(c_int) > 0 and len(s_int) > 0:
            u_int, p_int = mannwhitneyu(c_int, s_int)

        else:
            u_int, p_int = None, None

        # -----------------------------
        # Significance labels
        # -----------------------------

        def sig_label(p):
            if p is None:
                return "NA"
            elif p < 0.001:
                return "***"
            elif p < 0.01:
                return "**"
            elif p < 0.05:
                return "*"
            else:
                return "ns"

        results.append({

            "type": t,

            "amp_U":
                round(u_amp, 3) if u_amp else None,

            "amp_p":
                round(p_amp, 4) if p_amp else None,

            "amp_sig":
                sig_label(p_amp),

            "int_U":
                round(u_int, 3) if u_int else None,

            "int_p":
                round(p_int, 4) if p_int else None,

            "int_sig":
                sig_label(p_int)
        })

    result_df = pd.DataFrame(results)

    print(result_df)

    return result_df

def compare_runs(df_all):

    results = []

    for t in df_all["type"].unique():

        # ---------- AMPLITUDE ----------
        amp_df = df_all[df_all["type"] == t]

        amp_pivot = amp_df.pivot_table(
            index="names",
            columns="run",
            values="amplitude",
            aggfunc="mean"
        ).dropna()

        if len(amp_pivot) >= 2:

            res_amp = wilcoxon(
                amp_pivot["run2"],
                amp_pivot["run4"]
            )

            w_amp = res_amp.statistic
            p_amp = res_amp.pvalue

        else:
            w_amp, p_amp = None, None

        # ---------- LATENCY ----------
        lat_df = df_all[df_all["type"] == t]

        lat_pivot = lat_df.pivot_table(
            index="names",
            columns="run",
            values="latency",
            aggfunc="mean"
        ).dropna()

        if len(lat_pivot) >= 2:

            res_lat = wilcoxon(
                lat_pivot["run2"],
                lat_pivot["run4"]
            )

            w_lat = res_lat.statistic
            p_lat = res_lat.pvalue

        else:
            w_lat, p_lat = None, None

        results.append({
            "type": t,

            "W_amplitude": round(w_amp, 3) if w_amp is not None else None,
            "p_amplitude": round(p_amp, 5) if p_amp is not None else None,
            "sig_amplitude":
                "Yes" if p_amp is not None and p_amp < 0.05 else "No",

            "W_latency": round(w_lat, 3) if w_lat is not None else None,
            "p_latency": round(p_lat, 5) if p_lat is not None else None,
            "sig_latency":
                "Yes" if p_lat is not None and p_lat < 0.05 else "No"
        })

    result_df = pd.DataFrame(results)

    print(result_df)

    return result_df

def boxplots_runs(df_all):

    # ---------- AMPLITUDE ----------
    plt.figure(figsize=(10, 5))

    sns.boxplot(
        data=df_all,
        x="type",
        y="amplitude",
        hue="run"
    )

    plt.title("Amplitude: Run2 vs Run4")
    plt.xlabel("Type")
    plt.ylabel("Amplitude")
    plt.savefig("boxplots_runs_amp.png")

    # ---------- LATENCY ----------
    plt.figure(figsize=(10, 5))

    sns.boxplot(
        data=df_all,
        x="type",
        y="latency",
        hue="run"
    )

    plt.title("Latency: Run2 vs Run4")
    plt.xlabel("Type")
    plt.ylabel("Latency")
    plt.savefig("boxplots_runs_lat.png")

def histograms(df,c):

    df = df.dropna(subset=["amplitude", "latency", "group"])

    # ---------- AMPLITUDE ----------
    plt.figure(figsize=(10, 5))

    sns.histplot(
        data=df,
        x="amplitude",
        hue="group",
        bins=25,
        kde=True,
        stat="density",
        common_norm=False,
        alpha=0.5
    )

    plt.title("Amplitude Distribution by Group")
    plt.xlabel("Amplitude")
    plt.ylabel("Density")

    plt.tight_layout()
    plt.savefig(f"histogram_amp{c}.png")
    plt.close()

    # ---------- LATENCY ----------
    plt.figure(figsize=(10, 5))

    sns.histplot(
        data=df,
        x="latency",
        hue="group",
        bins=25,
        kde=True,
        stat="density",
        common_norm=False,
        alpha=0.5
    )

    plt.title("Latency Distribution by Group")
    plt.xlabel("Latency")
    plt.ylabel("Density")

    plt.tight_layout()
    plt.savefig(f"histogram_lat{c}.png")
    plt.close()

def descriptive_stats(df, export_latex=False):

    result = (
        df.groupby(["group", "type"])
        .agg(
            amplitude_mean=("amplitude", "mean"),
            amplitude_std=("amplitude", "std"),
            amplitude_median=("amplitude", "median"),
            amplitude_min=("amplitude", "min"),
            amplitude_max=("amplitude", "max"),

            latency_mean=("latency", "mean"),
            latency_std=("latency", "std"),
            latency_median=("latency", "median"),
            latency_min=("latency", "min"),
            latency_max=("latency", "max"),

            n=("amplitude", "count")
        )
        .reset_index()
    )

    # round numeric columns
    numeric_cols = result.select_dtypes(include="number").columns
    result[numeric_cols] = result[numeric_cols].round(3)

    print(result)

    # ---------- LATEX EXPORT ----------
    if export_latex:

        latex_table = result.to_latex(
            index=False,
            float_format="%.3f",
            caption="Descriptive statistics for amplitude and latency",
            label="tab:descriptive_stats",
            bold_rows=False,
            escape=False
        )

        print(latex_table)

        return result, latex_table

    return result

def df_to_latex(df, caption="Table", label="tab:table", float_fmt="%.3f"):
    """
    Convert a pandas DataFrame into a clean LaTeX table string.
    """

    latex = df.to_latex(
        index=False,
        float_format=float_fmt,
        caption=caption,
        label=label,
        bold_rows=False,
        escape=False
    )

    print(latex)
    return latex

def compare_runs_by_group(df_all):
    """
    df_all must contain:
    run, group, names, type, amplitude, latency
    """

    results = []

    for g in ["C", "S"]:
        for t in df_all["type"].unique():

            subset = df_all[
                (df_all["group"] == g) &
                (df_all["type"] == t)
            ]

            # ----- amplitude -----
            pivot_amp = subset.pivot_table(
                index="names",
                columns="run",
                values="amplitude",
                aggfunc="mean"
            ).dropna()

            # ----- latency -----
            pivot_lat = subset.pivot_table(
                index="names",
                columns="run",
                values="latency",
                aggfunc="mean"
            ).dropna()

            # skip if insufficient pairs
            if len(pivot_amp) < 3 or len(pivot_lat) < 3:
                continue

            # Wilcoxon
            w_amp, p_amp = wilcoxon(
                pivot_amp["run2"],
                pivot_amp["run4"]
            )

            w_lat, p_lat = wilcoxon(
                pivot_lat["run2"],
                pivot_lat["run4"]
            )

            results.append({
                "group": g,
                "type": t,
                "W_amplitude": round(w_amp, 3),
                "p_amplitude": round(p_amp, 5),
                "sig_amp": "Yes" if p_amp < 0.05 else "No",
                "W_latency": round(w_lat, 3),
                "p_latency": round(p_lat, 5),
                "sig_lat": "Yes" if p_lat < 0.05 else "No"
            })

    return pd.DataFrame(results)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

root2 = r"C:\Users\paulo\PycharmProjects\code\Paulo"

# ---------- PREPARE DATAFRAMES ----------

df2 = pd.DataFrame(main(root2,2))
df4 = pd.DataFrame(main(root2,4))

df2.to_csv("run2.csv",index=False)
df4.to_csv("run4.csv",index=False)

"""
histograms(df2,2)
histograms(df4,4)

# add run labels
df2["run"] = "run2"
df4["run"] = "run4"

# combine
df_24 = pd.concat(
    [df2, df4],
    ignore_index=True
)

# numeric conversion
df_24["amplitude"] = pd.to_numeric(
    df_24["amplitude"],
    errors="coerce"
)

df_24["latency"] = pd.to_numeric(
    df_24["latency"],
    errors="coerce"
)

df_24 = df_24.dropna(subset=["type", "amplitude", "latency"])
df_24["type"] = df_24["type"].astype(str).str.strip()

df_to_latex(normality_table(df2))
df_to_latex(normality_table(df4))
df_to_latex(compare_runs(df_24))
boxplots_runs(df_24)
descriptive_stats(df2,export_latex=True)
descriptive_stats(df4,export_latex=True)

df_to_latex(compare_runs_by_group(df_24))
"""
