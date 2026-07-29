import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import shapiro
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pingouin as pg
import scipy.stats as stats


def run_analysis_HrRr(filepath):

    df = pd.read_csv(filepath)

    # Create group column (C or S)
    df["group"] = df["subject"].str[0]

    # Descriptive statistics
    summary = (
        df.groupby(["group", "task"])
        .agg(
            mean_hr=("mean_hr", "mean"),
            sd_hr=("mean_hr", "std"),
            mean_rr=("mean_rr", "mean"),
            sd_rr=("mean_rr", "std")
        )
        .reset_index()
    )

    # Add Shapiro-Wilk p-values
    summary["p_normal_hr"] = None
    summary["p_normal_rr"] = None

    for idx, row in summary.iterrows():

        group = row["group"]
        task = row["task"]

        subset = df[
            (df["group"] == group) &
            (df["task"] == task)
            ]

        # Shapiro requires at least 3 observations
        if len(subset) >= 3:
            _, p_hr = shapiro(subset["mean_hr"])
            _, p_rr = shapiro(subset["mean_rr"])

            summary.loc[idx, "p_normal_hr"] = p_hr
            summary.loc[idx, "p_normal_rr"] = p_rr

    print(summary)

    import pingouin as pg

    df["group"] = df["subject"].str[0]
    df["group"] = df["group"].replace({"S": "PD"})

    aov_hr = pg.mixed_anova(
        data=df,
        dv="mean_hr",
        within="task",
        between="group",
        subject="subject"
    )

    print(aov_hr)

    aov_rr = pg.mixed_anova(
        data=df,
        dv="mean_rr",
        within="task",
        between="group",
        subject="subject"
    )

    print(aov_rr)

def boxplots(filepath):
    df=pd.read_csv(filepath)

    df["group"] = df["subject"].str[0]
    df["group"] = df["group"].replace({"S": "PD"})

    df["condition"] = df["group"] + "_" + df["task"]

    df["group"] = df["subject"].str[0]
    df["group"] = df["group"].replace({"S": "PD"})

    plt.figure(figsize=(8, 6))

    # Boxplot (keeps legend)
    sns.boxplot(
        data=df,
        x="group",
        y="mean_hr",
        hue="task"
    )

    # Stripplot (NO legend)
    sns.stripplot(
        data=df,
        x="group",
        y="mean_hr",
        hue="task",
        dodge=True,
        alpha=0.6,
        size=5,
        jitter=True,
        legend=False,  # <-- key fix
        color="black",
    )
    plt.ylabel("Heart Rate (bpm)")
    plt.title("Heart Rate by Group and Task")

    # Fix legend (keep only one)
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[:3], labels[:3], title="Task")

    plt.savefig("boxplot.png")

def plot_histograms(filepath):
    df = pd.read_csv(filepath)

    # Create group column from subject IDs
    df["group"] = df["subject"].str[0]

    # Rename S -> PD
    df["group"] = df["group"].replace({"S": "PD"})

    # -------------------------
    # Heart Rate
    # -------------------------

    plt.figure(figsize=(8, 6))

    sns.histplot(
        data=df,
        x="mean_hr",
        hue="group",
        kde=True,
        stat="count",
        alpha=0.5
    )

    plt.grid(True)
    plt.xlabel("Mean Heart Rate (bpm)")
    plt.ylabel("Count")
    plt.title("Distribution of Mean Heart Rate")
    plt.tight_layout()

    plt.savefig(f"hist_mean_hr.png", dpi=300)
    plt.close()

    # -------------------------
    # Respiration Rate
    # -------------------------

    plt.figure(figsize=(8, 6))

    sns.histplot(
        data=df,
        x="mean_rr",
        hue="group",
        kde=True,
        stat="count",
        alpha=0.5
    )

    plt.grid(True)
    plt.xlabel("Mean Respiration Rate (breaths/min)")
    plt.ylabel("Count")
    plt.title("Distribution of Mean Respiration Rate")
    plt.tight_layout()

    plt.savefig(f"hist_mean_rr.png", dpi=300)
    plt.close()

def cohens_d(a, b):
    return (np.mean(a) - np.mean(b)) / np.sqrt(
        (np.std(a, ddof=1) ** 2 + np.std(b, ddof=1) ** 2) / 2
    )

def boxplot(filepath, output="",col=""):

    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    # -------------------------------------
    # Load data
    # -------------------------------------

    df = pd.read_csv(filepath)

    # -------------------------------------
    # Create groups
    # -------------------------------------

    df["group"] = df["subject"].str[0]

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

        hue="task"

    )

    sns.stripplot(

        data=df,

        x="group",

        y=col,

        hue="task",

        dodge=True,

        alpha=0.6,

        size=5,

        jitter=True,

        color="black",

        legend=False

    )

    font=14
    plt.ylabel(

        col,fontsize=font

    )

    plt.xlabel(

        "Group",fontsize=font

    )

    plt.title(

        f"{col} by Group and Task",fontsize=font

    )

    # -------------------------------------
    # Keep one legend
    # -------------------------------------

    handles, labels = (

        plt.gca()

        .get_legend_handles_labels()

    )

    n_tasks = df["task"].nunique()

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

def hrv_summary_table(filepath):

    # ----------------------------------
    # Load data
    # ----------------------------------

    df = pd.read_csv(filepath)

    # ----------------------------------
    # Groups
    # ----------------------------------

    df["group"] = df["subject"].str[0]

    df["group"] = df["group"].replace({

        "S": "PD",

        "C": "C"

    })

    # ----------------------------------
    # HRV metrics to summarize
    # ----------------------------------

    metrics = [

        "sdnn",

        "rmssd",

        "sd1_sd2_ratio",

        "sdibi",

        "rmssd_resp",

        "sd1_sd2_ratio_resp"

    ]

    metrics = [

        m

        for m in metrics

        if m in df.columns

    ]

    results = []

    # ----------------------------------
    # Group × Task
    # ----------------------------------

    grouped = df.groupby(

        ["group", "task"]

    )

    for (group, task), subset in grouped:

        row = {

            "group": group,

            "task": task

        }

        for metric in metrics:

            values = subset[metric].dropna()

            if len(values) == 0:

                row[f"mean_{metric}"] = np.nan

                row[f"sd_{metric}"] = np.nan

                continue

            row[f"mean_{metric}"] = values.mean()

            row[f"sd_{metric}"] = values.std()


        results.append(row)

    result = pd.DataFrame(

        results

    )

    print(result.round(3))

    import pingouin as pg

    df["group"] = df["subject"].str[0]
    df["group"] = df["group"].replace({"S": "PD"})

    aov_hr = pg.mixed_anova(
        data=df,
        dv="rmssd",
        within="task",
        between="group",
        subject="subject"
    )

    print(aov_hr)

    aov_rr = pg.mixed_anova(
        data=df,
        dv="sdnn",
        within="task",
        between="group",
        subject="subject"
    )

    print(aov_rr)

    return result.round(3)

def mixed_anova_rmssd(filepath, dv):

    import pingouin as pg

    df = pd.read_csv(filepath)
    # ----------------------------------
    # Basic checks
    # ----------------------------------

    required_cols = ["subject", "group", "task", dv]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    data = df.dropna(subset=required_cols).copy()

    # ----------------------------------
    # Ensure correct types
    # ----------------------------------

    data["subject"] = data["subject"].astype(str)
    data["group"] = data["group"].astype(str)
    data["task"] = data["task"].astype(str)

    # ----------------------------------
    # Mixed ANOVA
    # ----------------------------------

    aov = pg.mixed_anova(

        dv=dv,
        within="task",
        between="group",
        subject="subject",
        data=data

    )

    return aov

import pandas as pd
import statsmodels.formula.api as smf


def compare_runs_with_task(file_run2,
                           file_run4,
                           col="mean_hr"):
    """
    Linear Mixed-Effects Model:
        DV ~ Group * Run * Task

    Random effect:
        Subject

    Returns:
        fitted model
    """

    # -------------------------
    # Load data
    # -------------------------
    run2 = pd.read_csv(file_run2)
    run4 = pd.read_csv(file_run4)

    run2["group"] = run2["subject"].str[0]
    run4["group"] = run4["subject"].str[0]

    run2["run"] = "Run2"
    run4["run"] = "Run4"

    # -------------------------
    # Keep only common subjects
    # -------------------------
    common_subjects = set(run2["subject"]).intersection(run4["subject"])
    run2 = run2[run2["subject"].isin(common_subjects)]
    run4 = run4[run4["subject"].isin(common_subjects)]

    # -------------------------
    # Combine
    # -------------------------
    df = pd.concat([run2, run4], ignore_index=True)

    # -------------------------
    # Mixed model
    # -------------------------
    model = smf.mixedlm(
        f"{col} ~ group * run * task",
        data=df,
        groups=df["subject"]
    )

    result = model.fit()

    print(result.summary())

    return result


def run_hrv_eda_lmm(filepath):
    """
    Linear mixed models relating HRV (RMSSD) to EDA.

    Fixed effects:
        - Group
        - Task
        - EDA metric

    Random effect:
        - Subject intercept

    Parameters
    ----------
    filepath : str
        CSV file containing one row per subject/task.

    Returns
    -------
    model_scl
    model_auc
    """

    df = pd.read_csv(filepath)

    # -----------------------------
    # Prepare variables
    # -----------------------------

    df["group"] = df["subject"].str[0]
    df["group"] = df["group"].replace({"S": "PD",
                                       "C": "Control"})

    df["group"] = df["group"].astype("category")
    df["task"] = df["task"].astype("category")
    # Log-transform phasic AUC
    df["ln_phasic_auc"] = np.log1p(df["phasic_auc"])

    # -----------------------------
    # Model 1
    # RMSSD ~ Mean SCL
    # -----------------------------

    model_scl = smf.mixedlm(
    "rmssd ~ task + group * mean_scl",
            data=df,
            groups=df["subject"]
        ).fit()

    print("\n")
    print("=" * 60)
    print("MODEL 1: RMSSD ~ Mean SCL")
    print("=" * 60)
    print(model_scl.summary())

    # -----------------------------
    # Model 2
    # RMSSD ~ Phasic AUC
    # -----------------------------

    model_auc = smf.mixedlm(
    "rmssd ~ task + group * phasic_auc",
            data=df,
            groups=df["subject"]
        ).fit()

    print("\n")
    print("=" * 60)
    print("MODEL 2: RMSSD ~ Phasic AUC")
    print("=" * 60)
    print(model_auc.summary())

    return model_scl, model_auc

def run_sdnn_eda_lmm(filepath):
    """
    Linear mixed models relating HRV (RMSSD) to EDA.

    Fixed effects:
        - Group
        - Task
        - EDA metric

    Random effect:
        - Subject intercept

    Parameters
    ----------
    filepath : str
        CSV file containing one row per subject/task.

    Returns
    -------
    model_scl
    model_auc
    """

    df = pd.read_csv(filepath)

    # -----------------------------
    # Prepare variables
    # -----------------------------

    df["group"] = df["subject"].str[0]
    df["group"] = df["group"].replace({"S": "PD",
                                       "C": "Control"})

    df["group"] = df["group"].astype("category")
    df["task"] = df["task"].astype("category")
    # Log-transform phasic AUC
    df["ln_phasic_auc"] = np.log1p(df["phasic_auc"])

    # -----------------------------
    # Model 1
    # RMSSD ~ Mean SCL
    # -----------------------------

    model_scl = smf.mixedlm(
    "sdnn ~ task + group * mean_scl",
            data=df,
            groups=df["subject"]
        ).fit()

    print("\n")
    print("=" * 60)
    print("MODEL 1: sdnn ~ Mean SCL")
    print("=" * 60)
    print(model_scl.summary())

    # -----------------------------
    # Model 2
    # RMSSD ~ Phasic AUC
    # -----------------------------

    model_auc = smf.mixedlm(
    "sdnn ~ task + group * phasic_auc",
            data=df,
            groups=df["subject"]
        ).fit()

    print("\n")
    print("=" * 60)
    print("MODEL 2: sdnn ~ Phasic AUC")
    print("=" * 60)
    print(model_auc.summary())

    return model_scl, model_auc

def boxplots_auc(filepath):
    """
    Creates two boxplots:
        1. ln(phasic_auc)
        2. ln(phasic_auc) after removing values >3 SD from the mean

    Returns
    -------
    summary : DataFrame
        group, task,
        mean_ln_auc,
        sd_ln_auc,
        p_normal_ln_auc
    """
    font=14
    # -------------------------------------------------
    # Load data
    # -------------------------------------------------
    df = pd.read_csv(filepath)

    df["group"] = df["subject"].str[0]
    df["group"] = df["group"].replace({"S": "PD"})

    # Remove impossible values
    df = df[df["phasic_auc"] > 0].copy()

    # -------------------------------------------------
    # Log transform
    # -------------------------------------------------
    df["ln_phasic_auc"] = np.log(df["phasic_auc"])

    # =================================================
    # BOXPLOT 1
    # =================================================
    plt.figure(figsize=(8,6))

    sns.boxplot(
        data=df,
        x="group",
        y="ln_phasic_auc",
        hue="task"
    )

    sns.stripplot(
        data=df,
        x="group",
        y="ln_phasic_auc",
        hue="task",
        dodge=True,
        color="black",
        alpha=0.6,
        size=5,
        legend=False
    )

    plt.ylabel("ln(Phasic AUC)",fontsize=font)
    plt.title("Log-transformed Phasic AUC",fontsize=font)
    plt.xlabel("Group",fontsize=font)

    handles, labels = plt.gca().get_legend_handles_labels()
    n_tasks = len(df["task"].unique())
    plt.legend(handles[:n_tasks], labels[:n_tasks], title="Task")

    plt.tight_layout()
    plt.savefig("boxplot_ln_auc.png", dpi=300)
    plt.show()

    # =================================================
    # Remove 3 SD outliers
    # =================================================
    mean = df["ln_phasic_auc"].mean()
    sd = df["ln_phasic_auc"].std()

    df_clean = df[
        np.abs(df["ln_phasic_auc"] - mean) <= 3 * sd
    ].copy()

    # =================================================
    # BOXPLOT 2
    # =================================================
    plt.figure(figsize=(8,6))

    sns.boxplot(
        data=df_clean,
        x="group",
        y="ln_phasic_auc",
        hue="task"
    )

    sns.stripplot(
        data=df_clean,
        x="group",
        y="ln_phasic_auc",
        hue="task",
        dodge=True,
        color="black",
        alpha=0.6,
        size=5,
        legend=False
    )

    plt.ylabel("ln(Phasic AUC)",fontsize=font)
    plt.title("Log-transformed Phasic AUC (3 SD outliers removed)",fontsize=font)
    plt.xlabel("Group",fontsize=font)

    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[:n_tasks], labels[:n_tasks], title="Task")

    plt.tight_layout()
    plt.savefig("auc_after1.png", dpi=300)
    plt.show()

    # =================================================
    # Summary statistics
    # =================================================
    summary = (
        df_clean
        .groupby(["group", "task"])
        .agg(
            mean_ln_auc=("ln_phasic_auc", "mean"),
            sd_ln_auc=("ln_phasic_auc", "std"),
            mean_scl=("mean_scl", "mean"),
            sd_scl=("mean_scl", "std"),
        )
        .reset_index()
    )

    summary["p_normal_ln_auc"] = np.nan
    summary["p_normal_scl"] = None

    for idx, row in summary.iterrows():

        group = row["group"]
        task = row["task"]

        subset = df_clean[
            (df_clean["group"] == group) &
            (df_clean["task"] == task)
            ]

        _, p_ln_auc = shapiro(subset["ln_phasic_auc"])
        _, p_scl = shapiro(subset["mean_scl"])

        summary.loc[idx, "p_normal_ln_auc"] = p_ln_auc
        summary.loc[idx, "p_normal_scl"] = p_scl

    print(summary)

    print("\n" + "=" * 60)
    print("Linear Mixed Model: ln(phasic_auc)")
    print("=" * 60)

    model_auc = smf.mixedlm(
        "ln_phasic_auc ~ group * task",
        data=df_clean,
        groups=df_clean["subject"]
    )

    result_auc = model_auc.fit()

    print(result_auc.summary())

    # -------------------------
    # Residual normality
    # -------------------------
    w, p = shapiro(result_auc.resid)

    print(f"\nResidual Shapiro-Wilk: W = {w:.4f}, p = {p:.4f}")

    print("\n" + "=" * 60)
    print("Linear Mixed Model: Mean SCL")
    print("=" * 60)

    model_scl = smf.mixedlm(
        "mean_scl ~ group * task",
        data=df_clean,
        groups=df_clean["subject"]
    )

    result_scl = model_scl.fit()

    print(result_scl.summary())

    w, p = shapiro(result_scl.resid)

    print(f"\nResidual Shapiro-Wilk: W = {w:.4f}, p = {p:.4f}")


    return summary, result_auc, result_scl


def compare_runs_lmm(file_run2,
                     file_run4,
                     col="phasic_auc"):
    """
    Linear Mixed Model

        DV ~ group * run * task

    Random effect:
        subject

    Parameters
    ----------
    col : str
        "phasic_auc" or "mean_scl"

    Returns
    -------
    result : fitted MixedLM
    """

    # -------------------------------------------------
    # Load
    # -------------------------------------------------
    run2 = pd.read_csv(file_run2)
    run4 = pd.read_csv(file_run4)

    # -------------------------------------------------
    # Groups
    # -------------------------------------------------
    for df in (run2, run4):

        df["group"] = df["subject"].str[0]
        df["group"] = df["group"].replace({"S": "PD"})

    # -------------------------------------------------
    # Remove impossible AUC values
    # -------------------------------------------------
    if col == "phasic_auc":

        run2 = run2[run2["phasic_auc"] > 0].copy()
        run4 = run4[run4["phasic_auc"] > 0].copy()

        run2["value"] = np.log(run2["phasic_auc"])
        run4["value"] = np.log(run4["phasic_auc"])

        ylabel = "ln(phasic_auc)"

    else:

        run2["value"] = run2[col]
        run4["value"] = run4[col]

        ylabel = col

    # -------------------------------------------------
    # Keep only subjects present in both runs
    # -------------------------------------------------
    common = set(run2.subject).intersection(run4.subject)

    run2 = run2[run2.subject.isin(common)]
    run4 = run4[run4.subject.isin(common)]

    run2 = run2[run2["phasic_auc"] > 0].copy()
    run4 = run4[run4["phasic_auc"] > 0].copy()

    # -------------------------------------------------
    # Add run label
    # -------------------------------------------------
    run2["run"] = "Run2"
    run4["run"] = "Run4"

    # -------------------------------------------------
    # Combine
    # -------------------------------------------------
    df = pd.concat([run2, run4], ignore_index=True)

    # -------------------------------------------------
    # Mixed model
    # -------------------------------------------------
    model = smf.mixedlm(
        "value ~ group * run * task",
        data=df,
        groups=df["subject"]
    )

    result = model.fit()

    print("=" * 60)
    print(f"Linear Mixed Model: {ylabel}")
    print("=" * 60)

    print(result.summary())

    # -------------------------------------------------
    # Residual normality
    # -------------------------------------------------
    W, p = shapiro(result.resid)

    print()
    print(f"Residual Shapiro-Wilk: W = {W:.4f}, p = {p:.4f}")


    return result




run1=r"C:\Users\paulo\PycharmProjects\code\Ze_kitchen\Despair\result_run1.csv"
run2=r"C:\Users\paulo\PycharmProjects\code\Ze_kitchen\Despair\result_run2.csv"
run3=r"C:\Users\paulo\PycharmProjects\code\Ze_kitchen\Despair\result_run3.csv"
run4=r"C:\Users\paulo\PycharmProjects\code\Ze_kitchen\Despair\result_run4.csv"

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    run_analysis_HrRr(run2)
    hrv_summary_table(run2)
    run_analysis_HrRr(run4)
    hrv_summary_table(run4)


