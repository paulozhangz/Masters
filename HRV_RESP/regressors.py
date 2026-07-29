import numpy as np
import neurokit2 as nk
import pandas as pd
from pathlib import Path
import bioread
from nilearn.glm.first_level import spm_hrf
from scipy.signal import fftconvolve


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
                f"{subj} Run{run} pulse: trimming "
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
                f"{subj} Run{run} resp: trimming "
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

    rr = np.asarray(rr).copy()

    mask = np.ones(len(rr), dtype=bool)

    # physiological limits
    mask &= (rr >= 300)
    mask &= (rr <= 2000)

    for i in range(2, len(rr)-2):

        local = rr[i-2:i+3]

        med = np.median(local)

        if abs(rr[i] - med) > 0.20 * med:
            mask[i] = False

    rr_clean = rr[mask]

    return rr_clean, mask

def extract_eda_regressor(filepath,
                          dicom,
                          subj,
                          run):

    data = bioread.read_file(filepath)

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
                f"{subj} Run{run} eda: trimming "
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

    eda_clean = nk.eda_clean(
        signal,
        sampling_rate=fs
    )

    eda_signals, _ = nk.eda_process(
        eda_clean,
        sampling_rate=fs
    )

    tonic = eda_signals["EDA_Tonic"]

    phasic = eda_signals["EDA_Phasic"]

    duration = int(dicom[(subj, f"Run{run}")])

    old_t = np.arange(len(tonic)) / fs

    new_t = np.arange(duration)

    tonic_1hz = np.interp(
        new_t,
        old_t,
        tonic
    )

    phasic_1hz = np.interp(
        new_t,
        old_t,
        phasic
    )

    return tonic_1hz, phasic_1hz

def extract_hrv_regressor(
        puls_signal,
        subject,
        run,
        dicom,
        fs=400,
        window=60,
        step=1
):
    signals, info = nk.ppg_process(
        puls_signal,
        sampling_rate=fs,
        method="elgendi"
    )

    _, peaks = nk.signal_fixpeaks(
        info["PPG_Peaks"],
        sampling_rate=fs,
        iterative=True
    )

    peaks = np.asarray(peaks)

    # Compute RR intervals (ms)
    rr = np.diff(peaks) / fs * 1000

    # Each RR interval is assigned the time of the second beat
    rr_times = peaks[1:] / fs

    # Clean RR intervals
    rr_clean, mask = clean_rr_intervals(rr)

    # Keep only valid intervals and their timestamps
    rr = rr_clean
    rr_times = rr_times[mask]

    times = []
    rmssd = []

    t = window

    while t < rr_times[-1]:

        window_mask = (
                (rr_times >= t - window)
                &
                (rr_times <= t)
        )

        current = rr[window_mask]

        if len(current) > 5:

            diff_rr = np.diff(current)

            value = np.sqrt(
                np.mean(diff_rr ** 2)
            )

        else:

            value = np.nan

        rmssd.append(value)

        times.append(t)

        t += step

    times = np.asarray(times)

    rmssd = np.asarray(rmssd)

    valid = ~np.isnan(rmssd)

    times = times[valid]

    rmssd = rmssd[valid]

    duration = int(
        dicom[(subject, f"Run{run}")]
    )

    final_t = np.arange(duration)

    rmssd_1hz = np.interp(
        final_t,
        times,
        rmssd,
        left=rmssd[0],
        right=rmssd[-1]
    )

    return rmssd_1hz

def extract_resp_regressor(
        resp_signal,
        subj,
        run,
        dicom,
        fs=400
):

    duration = int(
        dicom[(subj, f"Run{run}")]
    )

    resp_clean = nk.rsp_clean(
        resp_signal,
        sampling_rate=fs
    )

    resp_signals, _ = nk.rsp_process(
        resp_clean,
        sampling_rate=fs
    )

    resp_rate = resp_signals[
        "RSP_Rate"
    ].values

    old_t = np.arange(
        len(resp_rate)
    ) / fs

    new_t = np.arange(
        duration
    )

    resp_1hz = np.interp(

        new_t,

        old_t,

        resp_rate

    )

    return resp_1hz

def hrf_convolve(signal):

    hrf = spm_hrf(

        t_r=1,

        oversampling=1

    )

    conv = fftconvolve(

        signal,

        hrf

    )

    return conv[:len(signal)]

from scipy.stats import zscore

def normalize(signal):

    signal = np.asarray(signal)

    signal = np.nan_to_num(
        signal,
        nan=np.nanmean(signal)
    )

    if np.std(signal) == 0:

        return signal

    return zscore(signal)

def save_bv_single(
        signal,
        subject,
        run,
        predictor_name
):

    n = len(signal)

    filename = (
        f"{subject}"
        f"_Run{run}"
        f"_{predictor_name}.sdm"
    )

    with open(filename, "w") as f:

        f.write("FileVersion:\t\t1\n\n")

        f.write("NrOfPredictors:\t\t2\n")

        f.write(f"NrOfDataPoints:\t\t{n}\n")

        f.write("IncludesConstant:\t\t1\n")

        f.write("FirstConfoundPredictor:\t\t2\n\n")

        # predictor colors
        f.write(
            "125 200 125\t125 125 125\n"
        )

        # names
        f.write(
            f'"{predictor_name}"\t"Constant"\n'
        )

        # values

        for x in signal:

            f.write(
                f"{x:.6f}\t1.000000\n"
            )

    print(f"Saved {filename}")

def save_bv_eda(
        phasic,
        tonic,
        subject,
        run
):

    n = len(phasic)

    filename = (
        f"{subject}"
        f"_Run{run}"
        "_EDA.sdm"
    )

    with open(filename, "w") as f:

        f.write("FileVersion:\t\t1\n\n")

        f.write("NrOfPredictors:\t\t3\n")

        f.write(f"NrOfDataPoints:\t\t{n}\n")

        f.write("IncludesConstant:\t\t1\n")

        f.write("FirstConfoundPredictor:\t\t3\n\n")

        # colors

        f.write(
            "255 50 50\t"
            "50 255 50\t"
            "255 255 255\n"
        )

        # names

        f.write(
            '"SCR_Phasic"\t'
            '"SCR_Tonic"\t'
            '"Constant"\n'
        )

        for p, t in zip(phasic, tonic):

            f.write(

                f"{p:.3f}\t"

                f"{t:.3f}\t"

                "1.0\n"

            )

    print(f"Saved {filename}")

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

            r"C:\Users\paulo\PycharmProjects\PythonProject\code\Ze_kitchen\max_dicom_per_run.csv"

        ).iterrows()

    }

    for i in range(len(lst)):
        subject = names[i]

        try:
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


            tonic, phasic = extract_eda_regressor(

                eda_data[i],

                dicom,

                subject,

                run

            )

            rmssd = extract_hrv_regressor(
                puls_signal,
                subject,
                run,
                dicom,
                fs=400
            )

            resp = extract_resp_regressor(

                resp_signal,

                subject,

                run,

                dicom,

                fs=400

            )
        except Exception as e:
            print(e)
            continue

        tonic = hrf_convolve(tonic)

        phasic = hrf_convolve(phasic)

        rmssd = hrf_convolve(rmssd)

        resp = hrf_convolve(resp)

        tonic = normalize(tonic)

        phasic = normalize(phasic)

        rmssd = normalize(rmssd)

        resp = normalize(resp)

        save_bv_single(
            rmssd,
            subject,
            run,
            "HRV"
        )

        save_bv_single(
            resp,
            subject,
            run,
            "RESP"
        )

        save_bv_eda(

            phasic,

            tonic,

            subject,

            run

        )

root=r"C:\Users\paulo\PycharmProjects\PythonProject\code\Paulo"

main(root,2)
