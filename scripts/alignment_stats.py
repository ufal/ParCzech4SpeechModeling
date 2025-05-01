from pathlib import Path
import pandas as pd
from tqdm import tqdm
import math
from multiprocessing import Pool


def get_segments(align_df):
    segments = []
    segment = {"start_id": -1}
    word_id = 0

    for i, row in align_df.iterrows():
        if row["relative_word_id"] is not None:
            if segment["start_id"] == -1:
                segment["start_id"] = i
            if row["relative_word_id"] < word_id:
                segment["end_id"] = i
                segments.append(segment)
                segment = {"start_id": i}
            word_id = row["relative_word_id"]
    return segments


def get_duration(align_df, segments, threshold):
    duration = 0

    for i, s in enumerate(segments):
        score = align_df.iloc[s["start_id"]:s["end_id"]]["edit_distance"].mean()

        if score < threshold:
            last_idx = s["end_id"]
            first_idx = s["start_id"]
            while math.isnan(align_df.iloc[last_idx]["end"]):
                last_idx -= 1
            while math.isnan(align_df.iloc[first_idx]["start"]):
                first_idx += 1
            segment_duration = align_df.iloc[last_idx]["end"] - align_df.iloc[first_idx]["start"]

            duration += segment_duration

    return duration

def solve(align_file, thresholds):
    align_df = pd.read_parquet(align_file)
    align_df["relative_word_id"] = None
    align_df.loc[~align_df.token_id.isna(), "relative_word_id"] = (
        align_df.loc[~align_df.token_id.isna(), "token_id"]
        .str.split(".").str[-1].str.replace("w", "").astype(int)
    )
    segments = get_segments(align_df)
    result = {
        threshold: get_duration(align_df, segments, threshold)
        for threshold in thresholds
    }
    return result

def solve_wrapper(params):
    return solve(params["align_df"], params["thresholds"])

if __name__ == "__main__":
    align_dir = Path("/lnet/troja/work/people/stankov/parczech4speechmodeling/alignment")
    default_model_name = "default_whisperlv2_with_numerals"
    norm_model_name = "default_whisperlv2_no_numerals"
    n_cores = 30

    thresholds = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 1, 2, 5]
    total_durations = {threshold: 0 for threshold in thresholds}

    jobs = []
    for align_file in Path(align_dir, norm_model_name).glob("*.parquet"):
        jobs.append({
            "align_df": align_file.as_posix(),
            "thresholds": tuple(thresholds)
        })


    with Pool(processes=n_cores) as pool:
        results = list(tqdm(pool.imap(solve_wrapper, jobs), total=len(jobs)))

    for threshold in thresholds:
        total_duration = sum(result[threshold] for result in results) / 3600
        print(f"Total duration for threshold {threshold}: {total_duration:.2f} hours")

