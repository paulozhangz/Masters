import bioread
import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk
import pandas as pd
from pathlib import Path
from scipy.stats import shapiro, mannwhitneyu
import seaborn as sns
from scipy.stats import wilcoxon

def plot_histograms(filepath):

    df = pd.read_csv(filepath)

    font=14
    # Rename S -> PD
    df["group"] = df["group"].replace({"S": "PD"})

    # -------------------------
    # Heart Rate
    # -------------------------

    plt.figure(figsize=(8, 6))

    sns.histplot(
        data=df,
        x="amplitude",
        hue="group",
        kde=True,
        stat="count",
        alpha=0.5,
        bins=15
    )

    plt.grid(True)
    plt.xlabel("Amplitude (μS)")
    plt.ylabel("Count")
    plt.title("Distribution of Amplitude (μS) by Group")
    plt.tight_layout()

    plt.savefig(f"histogram_amp.png", dpi=300)
    plt.close()

    # -------------------------
    # Respiration Rate
    # -------------------------

    plt.figure(figsize=(8, 6))

    sns.histplot(
        data=df,
        x="latency",
        hue="group",
        kde=True,
        stat="count",
        alpha=0.5,
        bins=15
    )

    plt.grid(True)
    plt.xlabel("Latency (s)")
    plt.ylabel("Count")
    plt.title("Distribution of Latency (s) by Group")
    plt.tight_layout()

    plt.savefig(f"histogram_lat.png", dpi=300)
    plt.close()

def boxplot(filepath, output="",col=""):

    # -------------------------------------
    # Load data
    # -------------------------------------

    df = pd.read_csv(filepath)

    # -------------------------------------
    # Create groups
    # -------------------------------------

    df["group"] = df["group"].replace({

        "S": "PD",

        "C": "C"

    })

    # -------------------------------------
    # Check column exists
    # -------------------------------------

    if col not in df.columns:

        raise ValueError(

            f"Column {col} not found."

        )

    # -------------------------------------
    # Remove invalid values
    # -------------------------------------

    df = df.dropna(

        subset=[col]

    )

    # -------------------------------------
    # Plot
    # -------------------------------------

    plt.figure(

        figsize=(8, 6)

    )

    sns.boxplot(

        data=df,

        x="group",

        y=col,

        hue="type"

    )

    font=14
    if col == "amplitude":
        plt.ylabel(
            "Amplitude (μS)", fontsize=font
        )
    elif col == "latency":
        plt.ylabel(

            "Latency (s)", fontsize=font

        )
    else:
        plt.ylabel(col, fontsize=font)

    plt.xlabel(
        "Group", fontsize=font
    )

    plt.title(
        f"{col} by Group and Task", fontsize=font
    )

    # -------------------------------------
    # Keep one legend
    # -------------------------------------

    handles, labels = (

        plt.gca()

        .get_legend_handles_labels()

    )

    n_tasks = df["type"].nunique()

    plt.legend(

        handles[:n_tasks],

        labels[:n_tasks],

        title="Task"

    )

    plt.tight_layout()

    plt.savefig(

        output,

        dpi=300

    )

    plt.show()

def run_analysis1(filepath):

    df = pd.read_csv(filepath)

    df["group"] = df["group"].replace({

        "S": "PD",

        "C": "C"

    })

    # Descriptive statistics
    summary = (
        df.groupby(["group", "type"])
        .agg(
            mean_amp=("amplitude", "mean"),
            sd_amp=("amplitude", "std"),
            mean_lat=("latency", "mean"),
            sd_lat=("latency", "std")
        )
        .reset_index()
    )

    # Add Shapiro-Wilk p-values
    summary["p_normal_amp"] = None
    summary["p_normal_lat"] = None

    for idx, row in summary.iterrows():

        group = row["group"]
        task = row["type"]

        subset = df[
            (df["group"] == group) &
            (df["type"] == task)
            ]

        # Shapiro requires at least 3 observations
        if len(subset) >= 3:
            _, p_hr = shapiro(subset["amplitude"])
            _, p_rr = shapiro(subset["latency"])

            summary.loc[idx, "p_normal_amp"] = p_hr
            summary.loc[idx, "p_normal_lat"] = p_rr

    print(summary)


def rank_biserial(U, n1, n2):
    """
    Rank-biserial correlation effect size.
    """
    return 1 - (2 * U) / (n1 * n2)

def mann_whitney_by_task(csv_file):
    """
    Mann-Whitney U test:
        Control vs PD
    performed separately for each task.

    Expected columns:
        names, amplitude, latency, type, group

    group:
        C -> Control
        S -> PD
    """

    # -------------------------
    # Load data
    # -------------------------
    df = pd.read_csv(csv_file)

    # Rename group
    df["group"] = df["group"].replace({"S": "PD"})

    tasks = sorted(df["type"].unique())

    for task in tasks:

        print("=" * 60)
        print(f"Task: {task}")
        print("=" * 60)

        subset = df[df["type"] == task]

        control = subset[subset["group"] == "C"]
        pd_group = subset[subset["group"] == "PD"]

        # -------------------------
        # Amplitude
        # -------------------------
        amp_c = control["amplitude"].dropna()
        amp_pd = pd_group["amplitude"].dropna()

        if len(amp_c) > 0 and len(amp_pd) > 0:
            U_amp, p_amp = mannwhitneyu(
                amp_c,
                amp_pd,
                alternative="two-sided"
            )

            print("\nAmplitude")
            print(f"U = {U_amp:.3f}")
            print(f"p = {p_amp:.4f}")

        # -------------------------
        # Latency
        # -------------------------
        lat_c = control["latency"].dropna()
        lat_pd = pd_group["latency"].dropna()

        if len(lat_c) > 0 and len(lat_pd) > 0:
            U_lat, p_lat = mannwhitneyu(
                lat_c,
                lat_pd,
                alternative="two-sided"
            )

            print("\nLatency")
            print(f"U = {U_lat:.3f}")
            print(f"p = {p_lat:.4f}")

        print()


def boxplot_run3(filepath, col="amplitude", output="boxplot.png"):
    """
    Creates a boxplot comparing Control vs PD,
    separated by task type (easy/hard).

    Parameters
    ----------
    filepath : str
        CSV file

    col : str
        "amplitude" or "latency"

    output : str
        Output image filename
    """

    # -------------------------------------
    # Load data
    # -------------------------------------
    df = pd.read_csv(filepath)

    # -------------------------------------
    # Rename groups
    # -------------------------------------
    df["group"] = df["group"].replace({
        "S": "PD",
        "C": "C"
    })

    # -------------------------------------
    # Check column
    # -------------------------------------
    if col not in df.columns:
        raise ValueError(f"{col} not found.")

    # -------------------------------------
    # Remove missing values
    # -------------------------------------
    df = df.dropna(subset=[col])

    # -------------------------------------
    # Figure
    # -------------------------------------
    plt.figure(figsize=(8, 6))

    sns.boxplot(
        data=df,
        x="group",
        y=col,
        hue="type"
    )

    # -------------------------------------
    # Labels
    # -------------------------------------
    font = 14

    ylabel = {
        "amplitude": "Amplitude (μS)",
        "latency": "Latency (s)"
    }.get(col, col)

    plt.ylabel(ylabel, fontsize=font)
    plt.xlabel("Group", fontsize=font)
    plt.title(f"{ylabel} by Group and Task", fontsize=font)

    # -------------------------------------
    # Fix legend
    # -------------------------------------
    handles, labels = plt.gca().get_legend_handles_labels()
    n_types = df["type"].nunique()

    plt.legend(
        handles[:n_types],
        labels[:n_types],
        title="Task"
    )

    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.show()

def amplitude_correct_vs_wrong(filepath):
    """
    Compare SCR amplitude between correct and wrong trials
    within each Group (C/PD) and Task (type).

    Produces:
        - one boxplot per task
        - summary table with:
            mean ± SD
            Mann-Whitney U
            p-value
    """
    font=12
    # -----------------------------------------
    # Load
    # -----------------------------------------
    df = pd.read_csv(filepath)

    df["group"] = df["group"].replace({
        "S": "PD",
        "C": "C"
    })

    df["correct_label"] = df["correct"].replace({
        1: "Correct",
        0: "Wrong"
    })

    df = df.dropna(subset=["amplitude"])

    summary = []

    # -----------------------------------------
    # Loop over tasks
    # -----------------------------------------
    for task in sorted(df["type"].unique()):

        df_task = df[df["type"] == task]

        # ==========================
        # Boxplot
        # ==========================
        plt.figure(figsize=(8,6))

        sns.boxplot(
            data=df_task,
            x="group",
            y="amplitude",
            hue="correct_label"
        )

        handles, labels = plt.gca().get_legend_handles_labels()

        plt.legend(
            handles[:2],
            labels[:2],
            title="Response"
        )

        plt.ylabel("Amplitude (μS)",fontsize=font)
        plt.xlabel("Group",fontsize=font)
        plt.title(f"{task}: Correct vs Wrong",fontsize=font)

        plt.tight_layout()
        plt.savefig(f"Amplitude_{task}.png", dpi=300)
        plt.show()

        # ==========================
        # Statistics
        # ==========================
        for group in ["C", "PD"]:

            subset = df_task[df_task["group"] == group]

            correct = subset.loc[
                subset["correct"] == 1,
                "amplitude"
            ]

            wrong = subset.loc[
                subset["correct"] == 0,
                "amplitude"
            ]

            if len(correct) > 0 and len(wrong) > 0:

                U, p = mannwhitneyu(
                    correct,
                    wrong,
                    alternative="two-sided"
                )

            else:

                U = np.nan
                p = np.nan

            summary.append({

                "group": group,

                "task": task,

                "mean_correct": correct.mean(),

                "sd_correct": correct.std(),

                "mean_wrong": wrong.mean(),

                "sd_wrong": wrong.std(),

                "U": U,

                "p": p

            })

    summary = pd.DataFrame(summary)

    print(summary)

    return summary

from scipy.stats import wilcoxon

def amplitude_correct_vs_wrong2(filepath):
    """
    Compare SCR amplitude between correct and wrong trials
    within each Group (C/PD) and Task.

    Uses:
        Wilcoxon signed-rank test
        (paired by participant)

    Produces:
        - one boxplot per task
        - summary table
    """

    font = 12

    # -----------------------------------------
    # Load data
    # -----------------------------------------
    df = pd.read_csv(filepath)

    df["group"] = df["group"].replace({
        "S": "PD",
        "C": "C"
    })

    df["correct_label"] = df["correct"].replace({
        1: "Correct",
        0: "Wrong"
    })

    df = df.dropna(subset=["amplitude"])

    summary = []

    # -----------------------------------------
    # Loop over tasks
    # -----------------------------------------
    for task in sorted(df["type"].unique()):

        df_task = df[df["type"] == task]

        # ==========================
        # Boxplot
        # ==========================
        plt.figure(figsize=(8, 6))

        sns.boxplot(
            data=df_task,
            x="group",
            y="amplitude",
            hue="correct_label"
        )

        handles, labels = plt.gca().get_legend_handles_labels()

        plt.legend(
            handles[:2],
            labels[:2],
            title="Response"
        )

        plt.ylabel("Amplitude (μS)", fontsize=font)
        plt.xlabel("Group", fontsize=font)
        plt.title(f"{task}: Correct vs Wrong", fontsize=font)

        plt.tight_layout()
        plt.savefig(f"Amplitude_{task}.png", dpi=300)
        plt.show()

        # ==========================
        # Statistics
        # ==========================
        for group in ["C", "PD"]:

            subset = df_task[df_task["group"] == group]

            # Mean amplitude per participant
            paired = (
                subset
                .groupby(["names", "correct"])["amplitude"]
                .mean()
                .unstack()
            )

            # Keep only participants
            # having both correct and wrong trials
            paired = paired.dropna()

            if len(paired) >= 3:

                wrong = paired[0]
                correct = paired[1]

                try:
                    W, p = wilcoxon(
                        correct,
                        wrong,
                        alternative="two-sided"
                    )
                except ValueError:
                    W = np.nan
                    p = np.nan

                mean_correct = correct.mean()
                sd_correct = correct.std()

                mean_wrong = wrong.mean()
                sd_wrong = wrong.std()

            else:

                W = np.nan
                p = np.nan

                mean_correct = np.nan
                sd_correct = np.nan

                mean_wrong = np.nan
                sd_wrong = np.nan

            summary.append({

                "group": group,
                "task": task,

                "n_subjects": len(paired),

                "mean_correct": mean_correct,
                "sd_correct": sd_correct,

                "mean_wrong": mean_wrong,
                "sd_wrong": sd_wrong,

                "W": W,
                "p": p

            })

    summary = pd.DataFrame(summary)

    print("\nWilcoxon Signed-Rank Results")
    print(summary)

    return summary

def latency_vs_rx_time(filepath,
                       output_easy="latency_rt_easy.png",
                       output_hard="latency_rt_hard.png"):

    # -------------------------------------------------
    # Load
    # -------------------------------------------------

    df = pd.read_csv(filepath)

    df["group"] = df["group"].replace({
        "S": "PD",
        "C": "C"
    })

    df = df.dropna(subset=["latency", "rx_time"])
    df = df[df["rx_time"] > 0.5]

    summary = []

    # ============================================================
    # Plot Easy and Hard separately
    # ============================================================

    for task, outfile in zip(
            ["easy", "hard"],
            [output_easy, output_hard]):

        subset = df[df["type"] == task]

        plot_df = subset.melt(
            id_vars=["group"],
            value_vars=["latency", "rx_time"],
            var_name="Measure",
            value_name="Time"
        )

        plt.figure(figsize=(8,6))

        sns.boxplot(
            data=plot_df,
            x="Measure",
            y="Time",
            hue="group"
        )


        plt.ylabel("Time (s)", fontsize=14)
        plt.xlabel("")
        plt.title(f"{task.capitalize()} Questions", fontsize=15)

        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(handles[:2], labels[:2], title="Group")

        plt.tight_layout()
        plt.savefig(outfile, dpi=300)
        plt.show()

        # -------------------------------------------------------
        # Statistics
        # -------------------------------------------------------

        for grp in ["C", "PD"]:

            x = subset.loc[
                subset.group == grp,
                "latency"
            ]

            y = subset.loc[
                subset.group == grp,
                "rx_time"
            ]

            # Wilcoxon signed-rank test (paired data)
            try:
                W, p = wilcoxon(
                    x,
                    y,
                    alternative="two-sided"
                )
            except ValueError:
                # Happens if all paired differences are zero
                W = np.nan
                p = np.nan

            summary.append({

                "group": grp,

                "task": task,

                "mean_latency": x.mean(),

                "sd_latency": x.std(),

                "mean_rx_time": y.mean(),

                "sd_rx_time": y.std(),

                "p_normal_latency":
                    shapiro(x).pvalue if len(x) >= 3 else np.nan,

                "p_normal_rx_time":
                    shapiro(y).pvalue if len(y) >= 3 else np.nan,

                "Wilcoxon W": W,

                "p": p

            })

    summary = pd.DataFrame(summary)

    print(summary)

    return summary

def compare_runs_by_task(run2_file, run4_file, column):
    """
    Compare Run 2 vs Run 4 for each task using the Wilcoxon signed-rank test.

    Parameters
    ----------
    run2_file : str
    run4_file : str
    column : str
        Column to compare (e.g., amplitude, latency, mean_scl, ln_phasic_auc)

    Returns
    -------
    DataFrame with results.
    """

    run2 = pd.read_csv(run2_file)
    run4 = pd.read_csv(run4_file)

    run2["run"] = "Run2"
    run4["run"] = "Run4"

    tasks = sorted(set(run2["type"]).intersection(run4["type"]))

    results = []

    for task in tasks:

        r2 = run2[run2["type"] == task][["names", column]]
        r4 = run4[run4["type"] == task][["names", column]]

        # Keep only subjects present in both runs
        merged = pd.merge(
            r2,
            r4,
            on="names",
            suffixes=("_run2", "_run4")
        )

        merged = merged.dropna()

        if len(merged) < 2:
            continue

        stat, p = wilcoxon(
            merged[f"{column}_run2"],
            merged[f"{column}_run4"]
        )

        results.append({
            "task": task,
            "n": len(merged),
            "mean_run2": merged[f"{column}_run2"].mean(),
            "mean_run4": merged[f"{column}_run4"].mean(),
            "W": stat,
            "p": p
        })

    results = pd.DataFrame(results)

    print(results)

    return results

def boxplots_runs(run2_file, run4_file):
    """
    Creates two boxplots:
        1. Amplitude by Task and Run
        2. Latency by Task and Run
    """

    # -------------------------------------
    # Load data
    # -------------------------------------
    run2 = pd.read_csv(run2_file)
    run4 = pd.read_csv(run4_file)

    run2["run"] = "Run2"
    run4["run"] = "Run4"

    df = pd.concat([run2, run4], ignore_index=True)

    # -------------------------------------
    # Boxplot: Amplitude
    # -------------------------------------
    plt.figure(figsize=(8, 6))

    sns.boxplot(
        data=df,
        x="type",
        y="amplitude",
        hue="run"
    )


    plt.xlabel("Task", fontsize=14)
    plt.ylabel("Amplitude (μS)", fontsize=14)
    plt.title("SCR Amplitude: Run 2 vs Run 4", fontsize=14)

    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[:2], labels[:2], title="Run")

    plt.tight_layout()
    plt.savefig("Amplitude_RunComparison.png", dpi=300)
    plt.show()

    # -------------------------------------
    # Boxplot: Latency
    # -------------------------------------
    plt.figure(figsize=(8, 6))

    sns.boxplot(
        data=df,
        x="type",
        y="latency",
        hue="run"
    )


    plt.xlabel("Task", fontsize=14)
    plt.ylabel("Latency (s)", fontsize=14)
    plt.title("SCR Latency: Run 2 vs Run 4", fontsize=14)

    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[:2], labels[:2], title="Run")

    plt.tight_layout()
    plt.savefig("Latency_RunComparison.png", dpi=300)
    plt.show()

def mann_whitney_rxtime(csv_file):
    """
    Mann-Whitney U test:
        Control vs PD
    performed separately for each task.

    Expected columns:
        names, amplitude, latency, type, group

    group:
        C -> Control
        S -> PD
    """

    # -------------------------
    # Load data
    # -------------------------
    df = pd.read_csv(csv_file)

    # Rename group
    df["group"] = df["group"].replace({"S": "PD"})

    tasks = sorted(df["type"].unique())

    for task in tasks:

        print("=" * 60)
        print(f"Task: {task}")
        print("=" * 60)

        subset = df[df["type"] == task]

        control = subset[subset["group"] == "C"]
        pd_group = subset[subset["group"] == "PD"]

        # -------------------------
        # rx_time
        # -------------------------
        lat_c = control["rx_time"].dropna()
        lat_pd = pd_group["rx_time"].dropna()

        if len(lat_c) > 0 and len(lat_pd) > 0:
            U_lat, p_lat = mannwhitneyu(
                lat_c,
                lat_pd,
                alternative="two-sided"
            )

            print("\nLatency")
            print(f"U = {U_lat:.3f}")
            print(f"p = {p_lat:.4f}")

        print()

run1=r"C:\Users\paulo\PycharmProjects\code\Ze_kitchen\EDA\run1.csv"
run2=r"C:\Users\paulo\PycharmProjects\code\Ze_kitchen\EDA\run2.csv"
run3=r"C:\Users\paulo\PycharmProjects\code\Ze_kitchen\EDA\run3.csv"
run4=r"C:\Users\paulo\PycharmProjects\code\Ze_kitchen\EDA\run4.csv"

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    mann_whitney_rxtime(run3)
    #plot_histograms(run4)
    #boxplots_runs(run2,run4)
    #boxplot_run3(run3,output="boxplot_amplitude.png",col="amplitude")
    #boxplot_run3(run3, output="boxplot_latency.png",col="latency")
    #amplitude_correct_vs_wrong2(run3)
    #latency_vs_rx_time(run3)
    #compare_runs_by_task(run2,run4,"amplitude")
    #compare_runs_by_task(run2, run4, "latency")
