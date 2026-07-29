import numpy as np
import neurokit2 as nk
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import traceback
import bioread
from sklearn.linear_model import LinearRegression


def list_pairs(root_dir, n):

    root = Path(root_dir)

    pairs = []
    names = []
    log = []
    eda = []

    # -------------------------------------------------------------
    # Possible patterns
    # -------------------------------------------------------------

    patterns = {
        1: ["HGR1AP", "R1HGAP"],
        2: ["HBR1AP", "R1HBAP","HBR2AP","R2HBAP","ARR2AP"],
        3: ["HATR1AP", "R1ATAP","ARR3AP","ATR1AP","AMR1AP","ARR1AP","R3ARAP"],
        4: ["HBR2AP", "R2HBAP","HBR4AP","ARR4AP"],
    }

    # -------------------------------------------------------------
    # Validate n
    # -------------------------------------------------------------

    if n not in patterns:
        raise ValueError("n must be 1, 2, 3 or 4")

    possible_patterns = patterns[n]

    # -------------------------------------------------------------
    # Iterate subjects
    # -------------------------------------------------------------

    for subject_folder in root.iterdir():

        if not subject_folder.is_dir():
            continue

        name = subject_folder.name.upper()

        if not (
            name.startswith("C")
            or name.startswith("S")
        ):
            continue

        acq = next(subject_folder.glob(f"*run{n}*.acq"), None)
        txt = next(subject_folder.glob(f"*run{n}*.txt"), None)
        log.append(txt)
        eda.append(acq)


        # ---------------------------------------------------------
        # Find ONLY subfolder
        # ---------------------------------------------------------

        subfolders = [
            f for f in subject_folder.iterdir()
            if f.is_dir()
        ]

        if len(subfolders) == 0:
            continue

        if len(subfolders) > 1:

            print(
                f"WARNING: Multiple folders inside "
                f"{subject_folder}"
            )

            continue

        data_folder = subfolders[0]

        found = False

        # ---------------------------------------------------------
        # Try all possible naming conventions
        # ---------------------------------------------------------

        for base_pattern in possible_patterns:

            # -----------------------------------------------------
            # PRIORITY 1 -> AP2
            # -----------------------------------------------------

            puls_files = list(
                data_folder.glob(
                    f"*{base_pattern}2.puls"
                )
            )

            resp_files = list(
                data_folder.glob(
                    f"*{base_pattern}2.resp"
                )
            )

            # -----------------------------------------------------
            # PRIORITY 2 -> AP
            # -----------------------------------------------------

            if (
                len(puls_files) == 0
                or len(resp_files) == 0
            ):

                puls_files = list(
                    data_folder.glob(
                        f"*{base_pattern}.puls"
                    )
                )

                resp_files = list(
                    data_folder.glob(
                        f"*{base_pattern}.resp"
                    )
                )

            # -----------------------------------------------------
            # If files found -> store and stop searching
            # -----------------------------------------------------

            if (
                len(puls_files) > 0
                and len(resp_files) > 0
            ):

                pairs.append(
                    (
                        puls_files[0],
                        resp_files[0]
                    )
                )

                names.append(name)

                print("\n----------------------------------")
                print(f"SUBJECT: {name}")
                print(f"PULS: {puls_files[0].name}")
                print(f"RESP: {resp_files[0].name}")

                found = True

                break

        # ---------------------------------------------------------
        # Nothing found
        # ---------------------------------------------------------

        if not found:

            print(f"Missing files for {name}")

    return pairs, names, log, eda

def parse_log_file_1(filepath):
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

def parse_log_file_2(filepath):
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

def parse_log_file_3(filepath):
    TYPE_MAP = {
        0: "fixation_cross",
        1: "easy",
        2: "hard",
    }
    types = []
    events = []

    last_type = None

    with open(filepath) as f:
        for _ in range(15):
            next(f)

        for line in f:
            parts = line.strip().split("\t")

            if len(parts) < 3:
                continue

            try:
                block_type = int(parts[0])
                time = float(parts[2])
            except:
                continue

            # skip time_out rows
            if "time_out" in line:
                continue

            label = TYPE_MAP.get(block_type, "unknown")

            # only store if changed
            if label != last_type:
                types.append(label)
                events.append(time)
                last_type = label

    return types, events

def read_siemens_physio(filepath,run,subj,dicom, fs=400):
    """
    Read and clean Siemens physiological log files (.puls or .resp).

    Parameters
    ----------
    filepath : str
        Path to physiological log file.

    fs : int
        Sampling frequency in Hz.

    Returns
    -------
    signal : np.ndarray
        Cleaned physiological signal.

    time : np.ndarray
        Time vector in seconds.

    metadata : dict
        Timing metadata extracted from footer.
    """

    # ---------------------------------------------------------------------
    # READ FILE
    # ---------------------------------------------------------------------

    with open(filepath, "r") as f:
        line1 = f.readline().strip()

    # ---------------------------------------------------------------------
    # SPLIT FIRST LINE INTO TOKENS
    # ---------------------------------------------------------------------

    tokens = line1.split()

    # ---------------------------------------------------------------------
    # REMOVE EVERYTHING AFTER:
    # '5002', 'MSGTYPE', '301', '6002'
    # ---------------------------------------------------------------------

    pattern_end = ["5002", "MSGTYPE", "301", "6002"]

    idx_last = None

    for i in range(len(tokens) - 3):

        if tokens[i:i + 4] == pattern_end:
            idx_last = i

    if idx_last is not None:

        tokens = tokens[:idx_last]


    # ---------------------------------------------------------------------
    # REMOVE EVERYTHING BEFORE FIRST:
    # '5002', 'MSGTYPE', '300', '6002'
    # ---------------------------------------------------------------------

    pattern_start = ["5002", "MSGTYPE", "300", "6002"]

    idx_first = None

    for i in range(len(tokens) - 3):

        if tokens[i:i + 4] == pattern_start:
            idx_first = i
            break

    if idx_first is not None:
        tokens = tokens[idx_first + 4:]

    # ---------------------------------------------------------------------
    # REMOVE INTERNAL MESSAGE BLOCKS
    # ---------------------------------------------------------------------

    cleaned_tokens = []

    i = 0

    while i < len(tokens) - 3:

        current_block = tokens[i:i + 4]

        if current_block == pattern_start:
            i += 4
            continue

        elif current_block == pattern_end:
            i += 4
            continue

        cleaned_tokens.append(tokens[i])
        i += 1

    # append remaining tokens
    cleaned_tokens.extend(tokens[i:])

    tokens = cleaned_tokens

    # ---------------------------------------------------------------------
    # KEEP ONLY NUMERIC TOKENS
    # ---------------------------------------------------------------------

    numeric_tokens = []

    for token in tokens:

        try:
            float(token)
            numeric_tokens.append(token)

        except ValueError:
            pass

    nums = np.array(numeric_tokens, dtype=float)

    # ---------------------------------------------------------------------
    # REMOVE TRIGGER VALUES (5000)
    # ---------------------------------------------------------------------

    nums = nums[nums != 5000]

    # ---------------------------------------------------------------------
    # BUILD SIGNAL
    # ---------------------------------------------------------------------

    signal = nums.copy()

    dicom_count = dicom.get((subj, f"Run{run}"))

    if dicom_count is None:
        print(f"No DicomCount found for {subj} Run{run}")
    else:
        dicom_count = int(dicom_count)

        signal_duration = len(signal) / fs  # seconds

        if signal_duration > dicom_count:

            excess_seconds = signal_duration - dicom_count
            excess_samples = int(round(excess_seconds * fs))

            print(
                f"{subj} Run{run}: trimming "
                f"{excess_samples} samples "
                f"({excess_seconds:.2f} s)"
            )

            signal = signal[:-excess_samples]

        elif signal_duration < dicom_count:

            print(
                f"{subj} Run{run}: physiological recording shorter "
                f"({signal_duration:.2f} s) than DicomCount "
                f"({dicom_count})"
            )

    return signal

def read_resp(filename,run,subj,dicom,fs=400):
    # -------------------------------------------------------------------------
    # READ FIRST LINE
    # -------------------------------------------------------------------------

    with open(filename, "r") as f:
        first_line = f.readline().strip()


    # -------------------------------------------------------------------------
    # SPLIT INTO TOKENS
    # -------------------------------------------------------------------------

    tokens = first_line.split()


    # -------------------------------------------------------------------------
    # FIND START OF RESPIRATION DATA
    #
    # Data begin after:
    # 5002 MSGTYPE 400 6002
    # -------------------------------------------------------------------------

    start_pattern = ["5002", "MSGTYPE", "400", "6002"]

    start_idx = None

    for i in range(len(tokens) - 3):

        if tokens[i:i+4] == start_pattern:
            start_idx = i + 4
            break


    if start_idx is None:
        raise ValueError("Could not find start of respiration data.")


    # -------------------------------------------------------------------------
    # KEEP ONLY TOKENS AFTER HEADER
    # -------------------------------------------------------------------------

    data_tokens = tokens[start_idx:]


    # -------------------------------------------------------------------------
    # CONVERT TO NUMBERS ONLY
    # -------------------------------------------------------------------------

    numbers = []

    for token in data_tokens:

        try:
            numbers.append(int(token))

        except ValueError:
            pass


    # -------------------------------------------------------------------------
    # EXTRACT RESPIRATION SIGNAL
    #
    # Expected sequence:
    #
    #   RESP 33554432 98304 67108864 163840
    #
    # Example:
    #
    #   1440 33554432 98304 67108864 163840
    #
    # BUT trigger values (5000) may appear between packets:
    #
    #   1507 33554432 98304 67108864 163840
    #   5000
    #   1507 33554432 98304 67108864 163840
    #
    # We skip the trigger and continue reading packets normally.
    # -------------------------------------------------------------------------

    resp_signal = []

    i = 0

    while i <= len(numbers) - 5:

        # -------------------------------------------------------------
        # Skip trigger values
        # -------------------------------------------------------------

        if numbers[i] == 5000:
            i += 1
            continue

        # -------------------------------------------------------------
        # Check if next 4 values match the expected fixed pattern
        # -------------------------------------------------------------

        if (
            numbers[i + 1] == 33554432 and
            numbers[i + 2] == 98304 and
            numbers[i + 3] == 67108864 and
            numbers[i + 4] == 163840
        ):

            # First value is the respiration signal
            resp_signal.append(numbers[i])

            # Move to next packet
            i += 5

        else:
            # If packet structure is broken, move forward by 1
            i += 1


    signal = np.array(resp_signal)

    dicom_count = dicom.get((subj, f"Run{run}"))

    if dicom_count is None:
        print(f"No DicomCount found for {subj} Run{run}")
    else:
        dicom_count = int(dicom_count)

        signal_duration = len(signal) / fs  # seconds

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
                f"{subj} Run{run}: physiological recording shorter "
                f"({signal_duration:.2f} s) than DicomCount "
                f"({dicom_count})"
            )

    return signal

def clean_rr_intervals(rr):

    rr = np.asarray(rr)

    mask = np.ones(
        len(rr),
        dtype=bool
    )

    # --------------------
    # Physiological limits
    # --------------------

    mask &= (rr >= 400)
    mask &= (rr <= 1500)

    # --------------------
    # First 4 intervals:
    # median ± 2 SD
    # --------------------

    n_history = 4

    if len(rr) >= n_history:

        first = rr[:n_history]

        med = np.median(first)

        std = np.std(first)

        if std > 0:

            lower = med - 2 * std

            upper = med + 2 * std

            mask[:n_history] &= (
                (first >= lower)
                &
                (first <= upper)
            )

    # --------------------
    # Rolling 2 SD filter
    # --------------------

    for i in range(n_history, len(rr)):

        # use previously accepted RR intervals only
        history = rr[:i][mask[:i]]

        history = history[-n_history:]

        if len(history) < 2:
            continue

        mean_hist = np.mean(history)

        std_hist = np.std(history)

        if std_hist < 1:
            continue

        lower = mean_hist - 2 * std_hist

        upper = mean_hist + 2 * std_hist

        if rr[i] < lower or rr[i] > upper:

            mask[i] = False

    # --------------------
    # Large beat-to-beat jumps
    # --------------------

    rr_diff = np.abs(
        np.diff(rr)
    )

    jump_mask = np.ones(
        len(rr),
        dtype=bool
    )

    jump_mask[1:] &= rr_diff < 200

    mask &= jump_mask

    rr_clean = rr[mask]

    return rr_clean, mask

def compute_hrv_metrics(rr_ms):
    """
    rr_ms : array-like
        Inter-beat intervals in milliseconds.

    Returns
    -------
    dict
    """

    rr_ms = np.asarray(rr_ms)

    if len(rr_ms) < 10:
        return None

    diff_rr = np.diff(rr_ms)

    if len(diff_rr) < 2:
        return None

    mean_ibi = np.mean(rr_ms)

    sdnn = np.std(rr_ms, ddof=1)

    rmssd = np.sqrt(
        np.mean(diff_rr**2)
    )

    ln_rmssd = np.log(rmssd) if rmssd > 0 else np.nan

    sdsd = np.std(
        diff_rr,
        ddof=1
    )

    pnn20 = (
        np.sum(
            np.abs(diff_rr) > 20
        )
        /
        len(diff_rr)
    ) * 100

    sd1 = np.sqrt(0.5) * sdsd

    sd2_sq = (
        2 * sdnn**2
        - 0.5 * sdsd**2
    )

    sd2 = (
        np.sqrt(sd2_sq)
        if sd2_sq > 0
        else np.nan
    )

    sd1_sd2_ratio = (
        sd1 / sd2
        if sd2 > 0
        else np.nan
    )

    return {

        "mean_ibi": mean_ibi,

        "sdnn": sdnn,

        "rmssd": rmssd,

        "ln_rmssd": ln_rmssd,

        "pnn20": pnn20,

        "sd1": sd1,

        "sd1_sd2_ratio": sd1_sd2_ratio

    }

def extract_eda_features(filepath, events, labels,
                         dicom, subj, run):


    data=bioread.read_file(filepath)
    channel = data.channels[0]
    signal = channel.data
    fs = channel.samples_per_second

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


    # ----------------------------------
    # Clean EDA
    # ----------------------------------

    try:

        eda_clean = nk.eda_clean(
            signal,
            sampling_rate=fs
        )

        eda_signals, _ = nk.eda_process(
            eda_clean,
            sampling_rate=fs
        )

    except Exception:

        return []

    results = []

    # ----------------------------------
    # Loop epochs
    # ----------------------------------

    for j in range(len(events)):

        start_sec = events[j]

        if j < len(events) - 1:

            end_sec = events[j + 1]

        else:

            end_sec = len(signal) / fs

        label = labels[j]

        start_idx = round(
            start_sec * fs
        )

        end_idx = round(
            end_sec * fs
        )

        if end_idx <= start_idx:

            continue

        if (
            end_idx - start_idx
            < fs * 10
        ):

            continue

        seg = eda_signals.iloc[
            start_idx:end_idx
        ]

        if len(seg) == 0:

            continue

        # ----------------------------
        # Tonic
        # ----------------------------

        scl = seg[
            "EDA_Tonic"
        ].values

        mean_scl = np.nanmean(
            scl
        )

        std_scl = np.nanstd(
            scl
        )

        # slope

        try:

            scl_slope = np.polyfit(

                np.arange(
                    len(scl)
                ),

                scl,

                1

            )[0]

        except Exception:

            scl_slope = np.nan

        # ----------------------------
        # Phasic
        # ----------------------------

        phasic = seg[
            "EDA_Phasic"
        ].values

        phasic_auc = np.trapezoid(
            np.abs(phasic)
        )

        # SCR peaks

        scr_mask = seg[
            "SCR_Peaks"
        ].values.astype(bool)

        scr_count = np.sum(
            scr_mask
        )

        duration = (
            end_sec
            - start_sec
        )

        scr_rate = (
            scr_count
            / duration
            * 60
        )

        # amplitudes

        if scr_count > 0:

            scr_amplitudes = seg.loc[

                scr_mask,

                "SCR_Amplitude"

            ].values

            scr_amp_mean = np.nanmean(
                scr_amplitudes
            )

            scr_amp_max = np.nanmax(
                scr_amplitudes
            )

        else:

            scr_amp_mean = 0

            scr_amp_max = 0

        # ----------------------------
        # Signal quality
        # ----------------------------

        eda_std = np.nanstd(

            seg[
                "EDA_Clean"
            ]

        )

        # ----------------------------
        # Store
        # ----------------------------

        results.append({

            "subject": subj,

            "task": label,

            "mean_scl": mean_scl,

            "std_scl": std_scl,

            "scl_slope": scl_slope,

            "scr_count": scr_count,

            "scr_rate": scr_rate,

            "scr_amp_mean": scr_amp_mean,

            "scr_amp_max": scr_amp_max,

            "phasic_auc": phasic_auc,

            "eda_std": eda_std

        })

    return results

def compute_rsa(rr_times, rr_values, resp_signal, fs, start_idx, end_idx):
    """
    RSA via respiration–RR coupling (regression slope)
    """

    if len(rr_values) < 5:
        return np.nan

    # -----------------------------
    # Build RR time series
    # -----------------------------
    rr_time = rr_times / fs

    rr_series = np.interp(
        np.arange(start_idx, end_idx) / fs,
        rr_time,
        rr_values
    )

    # -----------------------------
    # Resp segment
    # -----------------------------
    resp_seg = resp_signal[start_idx:end_idx]

    if len(resp_seg) != len(rr_series):
        min_len = min(len(resp_seg), len(rr_series))
        resp_seg = resp_seg[:min_len]
        rr_series = rr_series[:min_len]

    if np.std(resp_seg) == 0 or np.std(rr_series) == 0:
        return np.nan

    # -----------------------------
    # Regression (RSA index)
    # -----------------------------
    X = resp_seg.reshape(-1, 1)
    y = rr_series

    model = LinearRegression().fit(X, y)

    rsa = model.coef_[0]

    return rsa

def quality_control(rr_clean, mask, rmssd):

    failures = []

    # ------------------------
    # Artifact fraction
    # ------------------------

    artifact_fraction = 1 - mask.mean()

    if artifact_fraction > 0.15:

        failures.append("artifact_fraction")

    # ------------------------
    # CV of RR intervals
    # ------------------------

    cv = np.std(rr_clean) / np.mean(rr_clean)

    if cv > 0.20:

        failures.append("cv")

    # ------------------------
    # RMSSD
    # ------------------------

    if rmssd > 120:

        failures.append("rmssd")

    # ------------------------
    # Reject if >=2 failures
    # ------------------------

    reject = len(failures) >= 2

    metrics = {

        "artifact_fraction": artifact_fraction,

        "cv": cv,

        "n_failures": len(failures),

        "failed_metrics": ",".join(failures),

        "reject": reject

    }

    return reject, metrics

def process_subject(i, lst, names, log, eda_data, run, dicom):

    fs = 400

    subject = names[i]

    if run == 1:
        labels, events = parse_log_file_1(log[i])

    elif run == 3:
        labels, events = parse_log_file_3(log[i])

    else:
        labels, events = parse_log_file_2(log[i])

    print(f"Processing {subject}")

    puls_signal = read_siemens_physio(
        lst[i][0],
        run,
        subject,
        dicom
    )

    resp_signal = read_resp(
        lst[i][1],
        run,
        subject,
        dicom
    )

    eda_signal = extract_eda_features(
        eda_data[i],
        events,
        labels,
        dicom,
        subject,
        run
    )

    eda_dict = {

        idx: row

        for idx, row

        in enumerate(eda_signal)

    }

    # -------------------------------------
    # Sanity checks
    # -------------------------------------

    if len(puls_signal) < 1000:
        return []

    if np.all(np.isnan(puls_signal)):
        return []

    if np.std(puls_signal) == 0:
        return []

    results = []

    plots=0
    max_plots = 0  #PLOTS FOR HRV

    resp_clean = nk.rsp_clean(
        resp_signal,
        sampling_rate=fs
    )

    resp_signals, resp_info = nk.rsp_process(
        resp_clean,
        sampling_rate=fs
    )

    resp_clean_series = resp_signals["RSP_Clean"].values


    # -------------------------------------
    # Loop epochs
    # -------------------------------------

    for j in range(len(events)):

        start_sec = events[j]

        if j < len(events)-1:

            end_sec = events[j+1]

        else:

            end_sec = len(puls_signal)/fs

        label = labels[j]

        start_idx = round(start_sec*fs)

        end_idx = round(end_sec*fs)

        # -------------------------------------
        # RESP processing
        # -------------------------------------

        resp_rate = resp_signals[
            "RSP_Rate"
        ].values[start_idx:end_idx]

        # -------------------------------------
        # PPG processing
        # -------------------------------------

        ppg_seg = puls_signal[start_idx:end_idx]

        if len(ppg_seg) < fs*10:
            continue

        try:

            seg_signals, seg_info = nk.ppg_process(
                ppg_seg,
                sampling_rate=fs,
                method="elgendi"
            )

        except Exception:

            continue

        peaks_before = np.asarray(
            seg_info["PPG_Peaks"]
        )

        plot_before=seg_signals["PPG_Clean"]

        peaks_before_y = seg_signals[
            "PPG_Clean"
        ].iloc[peaks_before]

        # -------------------------------------
        # Peak correction
        # -------------------------------------

        _, peaks = nk.signal_fixpeaks(
            peaks_before,
            sampling_rate=fs,
            method="Kubios",
            iterative=True
        )

        rr = np.diff(peaks) / fs * 1000



        hrv_before = compute_hrv_metrics(rr)  #BEFORE CORRECTION

        if hrv_before is None:
            continue

        rmssd_before = hrv_before["rmssd"]

        if rmssd_before > 120:

            rr_clean, mask = clean_rr_intervals(rr)

            if len(rr_clean) < 10:
                continue

            hrv = compute_hrv_metrics(rr_clean)

            if hrv is None:
                continue

            rmssd_after = hrv["rmssd"]

            reject, qc = quality_control(

                rr_clean,

                mask,

                rmssd_after

            )

            improved = (
                    rmssd_after < rmssd_before < 120
                    and
                    rmssd_after < 0.6 * rmssd_before
            )

            peaks_rsa = peaks[1:][mask]

            if improved and plots < max_plots:

                print("plotting")

                t_seg = np.arange(len(ppg_seg)) / fs

                fig, ax = plt.subplots(
                    2,
                    1,
                    figsize=(12, 8)
                )

                # --------------------------
                # PPG + peaks
                # --------------------------

                ax[0].plot(
                    t_seg,
                    plot_before,
                )

                ax[0].scatter(
                    peaks_before / fs,
                    peaks_before_y,
                    color="red",
                    s=20
                )

                ax[0].set_ylabel("PPG")

                ax[0].set_title(
                    f"{subject} | {label} | RMSSD={rmssd_before:.1f}"
                )

                # --------------------------
                # RR intervals
                # --------------------------

                rr_time = np.cumsum(rr) / 1000

                ax[1].plot(
                    rr_time,
                    rr,
                    marker="o"
                )

                ax[1].set_ylabel(
                    "RR (ms)"
                )

                ax[1].set_xlabel(
                    "Time (s)"
                )

                plt.tight_layout()

                plt.savefig(
                    f"high_rmssd_{subject}_{label}_{j}_before.png",
                    dpi=300
                )

                plt.close()

                #AFTER CORRECTION

                t_seg = np.arange(len(seg_signals["PPG_Clean"])) / fs
                peaks_after = peaks[1:][mask]

                fig, ax = plt.subplots(
                    2,
                    1,
                    figsize=(12, 8)
                )

                # --------------------------
                # PPG + peaks
                # --------------------------

                ax[0].plot(
                    t_seg,
                    seg_signals["PPG_Clean"]
                )

                ax[0].scatter(
                    peaks_after / fs,
                    seg_signals["PPG_Clean"].iloc[peaks_after],
                    color="red",
                    s=20
                )

                ax[0].set_title(
                    f"{subject} | {label} | RMSSD={rmssd_after:.1f}"
                )

                ax[0].set_ylabel("PPG")

                # --------------------------
                # RR intervals
                # --------------------------

                rr_time = peaks[1:][mask] / fs

                ax[1].plot(
                    rr_time,
                    rr_clean,
                    marker="o"
                )

                ax[1].set_ylabel(
                    "RR (ms)"
                )

                ax[1].set_xlabel(
                    "Time (s)"
                )

                plt.tight_layout()

                plt.savefig(
                    f"high_rmssd_{subject}_{label}_{j}_after.png",
                    dpi=300
                )

                plt.close()

                plots = plots + 1

            if reject:
                print(

                    f"{subject} | {label} "
    
                    f"rejected: {qc['failed_metrics']}"

                )

                continue      #If correction is needed

        else:
            rr_clean=rr
            hrv=hrv_before
            peaks_rsa = peaks[1:]

        rsa_epoch = compute_rsa(
            peaks_rsa,
            rr_clean,
            resp_clean_series,
            fs,
            start_idx,
            end_idx
        )

        # -------------------------------------
        # HR from cleaned RR
        # -------------------------------------

        heart_rate = 60000 / rr_clean

        mean_hr = np.mean(
            heart_rate
        )

        mean_rr = np.nanmean(resp_rate)

        # -------------------------------------
        # Skip invalid epochs
        # -------------------------------------

        if np.isnan(mean_hr):

            continue


        # -------------------------------------
        # Store
        # -------------------------------------

        eda_features = {

            k: v

            for k, v in eda_dict.get(j, {}).items()

            if k not in ["subject", "task"]

        }

        results.append({

            "subject": subject,

            "task": label,

            "mean_hr": mean_hr,

            "mean_rr": mean_rr,

            **hrv,

            **eda_features,

        # -----------------------
        # NEW RSA METRICS
        # -----------------------
            "rsa_mean": rsa_epoch


        })

    return results

def main(folder, run):

    # -------------------------------------------------------------
    # GET FILE PAIRS
    # -------------------------------------------------------------

    lst, names, log, eda_data = list_pairs(
        folder,
        run
    )

    dicom = {

        (row.Subject, row.Run): row.DicomCount

        for _, row in pd.read_csv(

            r"C:\Users\paulo\PycharmProjects\code\Ze_kitchen\max_dicom_per_run.csv"

        ).iterrows()

    }

    # -------------------------------------------------------------
    # STORE RESULTS
    # -------------------------------------------------------------

    all_results = []

    # -------------------------------------------------------------
    # LOOP SUBJECTS
    # -------------------------------------------------------------

    for i in range(len(lst)):

        try:

            res = process_subject(

                i,

                lst,

                names,

                log,

                eda_data,

                run,

                dicom

            )

        except Exception as e:

            tb = traceback.extract_tb(e.__traceback__)
            filename, lineno, func, text = tb[-1]  # last call = where it actually failed

            print(f"Skipping {names[i]}: ERROR {e}")
            print(f"  → Line {lineno}: {text}")

            continue

        all_results.extend(res)

    # -------------------------------------------------------------
    # CREATE DATAFRAME
    # -------------------------------------------------------------

    if len(all_results) == 0:

        return pd.DataFrame()

    df = pd.DataFrame(all_results)

    # -------------------------------------------------------------
    # Remove invalid rows
    # -------------------------------------------------------------

    required_cols = [

        "mean_hr",

        "mean_rr",

        "rmssd",

        "ln_rmssd",

        "mean_scl"

    ]

    for col in required_cols:
        print(col, df[col].dtype)

        if col in df.columns:

            df = df[

                np.isfinite(df[col])

            ]

    # -------------------------------------------------------------
    # Aggregate by subject/task
    # -------------------------------------------------------------

    aggregation = {

        # -------------------
        # HRV
        # -------------------

        "mean_hr": "mean",

        "mean_ibi": "mean",

        "sdnn": "mean",

        "rmssd": "mean",

        "ln_rmssd": "mean",

        "pnn20": "mean",

        "sd1": "mean",

        "sd1_sd2_ratio": "mean",

        "artifact_fraction": "mean",

        # -------------------
        # RESP
        # -------------------

        "mean_rr": "mean",

        # -------------------
        # EDA tonic
        # -------------------

        "mean_scl": "mean",

        "std_scl": "mean",

        "scl_slope": "mean",

        # -------------------
        # EDA phasic
        # -------------------

        "scr_count": "mean",

        "scr_rate": "mean",

        "scr_amp_mean": "mean",

        "scr_amp_max": "mean",

        "phasic_auc": "mean",

        # NEW RSA

        "rsa_mean": "mean",
        "rsa_sd": "mean",

    }

    # only aggregate columns that exist

    aggregation = {
        k: v
        for k, v in aggregation.items()
        if k in df.columns
    }

    result = (
        df
        .groupby(
            ["subject", "task"]
        )
        .agg(
            aggregation
        )
        .reset_index()
    )

    return result

def plot_section(folder, run):

    # -------------------------------------------------------------
    # GET FILE PAIRS
    # -------------------------------------------------------------
    lst, names, log, eda_data = list_pairs(
        folder,
        run
    )

    dicom = {
        (row.Subject, row.Run): row.DicomCount
        for _, row in pd.read_csv(
            r"C:\Users\paulo\PycharmProjects\PythonProject\code\Ze_kitchen\max_dicom_per_run.csv"
        ).iterrows()
    }

    i = 0
    subject = names[i]

    if run == 1:
        labels, events = parse_log_file_1(log[i])

    elif run == 3:
        labels, events = parse_log_file_3(log[i])

    else:
        labels, events = parse_log_file_2(log[i])

    puls_signal = read_siemens_physio(
        lst[i][0],
        run,
        subject,
        dicom
    )

    resp_signal = read_resp(
        lst[i][1],
        run,
        subject,
        dicom
    )


    fs = 400
    ppg_clean=nk.ppg_clean(puls_signal,fs)
    rsp_clean=nk.rsp_clean(resp_signal,fs)

    # -------------------------------------------------------------
    # Sampling frequency
    # -------------------------------------------------------------


    duration = 20  # seconds

    n_samples = min(
        int(duration * fs),
        len(puls_signal),
        len(resp_signal)
    )

    t = np.arange(n_samples) / fs

    # -------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------
    plt.figure(figsize=(12, 8))
    plt.plot(t, ppg_clean[:n_samples])
    plt.plot(t, rsp_clean[:n_samples])
    plt.ylabel('Amplitude')
    plt.xlabel('Time (s)')
    plt.title(f"PPG and RESP example")
    plt.grid(True)

    plt.savefig("example3.png")



root1=r"C:\Users\paulo\PycharmProjects\code\Paulo"

for i in [1,2,3,4]:
    df=main(root1,i)
    df.to_csv(f"run{i}.csv",index=False)
