import multiprocessing as mp
import re
from functools import partial
from pathlib import Path

import hydra
import pandas as pd
import torchaudio
import yaml
from tqdm import tqdm


def process_audio_row(row, mp3_folder, audio_format, part_audio_folder, overwrite_flag):
    segment_filename = Path(
        part_audio_folder,
        f"{row['vert']}_{row['seg_id']}_{row['start_token_id']}_{row['end_token_id']}.{audio_format}"
    )

    if segment_filename.exists() and not overwrite_flag:
        return segment_filename.as_posix()

    vertical = row["vert"]
    mp3_path = Path(
        mp3_folder,
        vertical[:4], vertical[4:6], vertical[6:8],
        vertical.replace(".tsv", ".mp3")
    )

    try:
        audio, sr = torchaudio.load(mp3_path)
        segment = audio[:, int(row["start"] * sr) : int(row["end"] * sr)]
        torchaudio.save(
            segment_filename,
            segment,
            sr,
            format=audio_format,
            encoding="PCM_S",
            bits_per_sample=16
        )
    except Exception as e:
        print(f"Error processing {mp3_path}: {str(e)}")
        return None

    return segment_filename.as_posix()


def parallel_process_audios(part_df, cfg, part_audio_folder):
    num_workers = max(mp.cpu_count() - 2, 2)

    # Create a partial function with fixed parameters
    process_func = partial(
        process_audio_row,
        mp3_folder=cfg.mp3_folder,
        audio_format=cfg.audio_format,
        part_audio_folder=part_audio_folder,
        overwrite_flag=cfg.overwrite
    )

    rows = part_df.to_dict("records")

    with mp.Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_func, rows),
            desc="Processing audios",
            total=len(part_df)
        ))

    return results


@hydra.main(config_path="../configs", config_name="create_hf_dataset_part")
def main(cfg):
    tqdm.pandas()

    trg_parquet_file = Path(cfg.parquet_file).stem
    dfs = []
    for parquet in Path(cfg.data_folder, cfg.inp_version, cfg.model_name).glob("*.parquet"):
        tmp_df = pd.read_parquet(parquet)
        tmp_df["parquet"] = parquet.stem
        dfs.append(tmp_df)

    df = pd.concat(dfs, ignore_index=True)

    with open(cfg.speakers_mapping_yaml) as file:
        speaker_mapping = yaml.safe_load(file)

    df["true_char_avg_dur"] = df["dur"] / df["n_true_chars"]
    df["split"] = df["speakers"].map(speaker_mapping)
    df["split"] = df["split"].fillna("train")

    pattern = "|".join(re.escape(p) for p in cfg.punc_list)
    regex_pattern = f"[{pattern}]"  # Or f'({pattern})' if using alternation

    df["norm_true_text"] = (
        df["true_text"]
        .str.strip()
        .str.lower()
        .str.replace(regex_pattern, "", regex=True)
    )
    df["speaker_text_cnt"] = df.groupby(["norm_true_text", "speakers"])["norm_true_text"].transform("count")

    clean_df = df.query(cfg.filter.replace("\n", " ").strip())
    part_df = clean_df[clean_df.parquet == trg_parquet_file].copy(deep=True).reset_index(drop=True)

    part_audio_folder = Path(
        cfg.output_folder,
        cfg.out_version,
        "audio",
        trg_parquet_file
    )
    part_audio_folder.mkdir(parents=True, exist_ok=True)

    part_metadata_folder = Path(
        cfg.output_folder,
        cfg.out_version,
        "metadata",
    )
    part_metadata_folder.mkdir(parents=True, exist_ok=True)

    audios = parallel_process_audios(
        part_df,
        cfg,
        part_audio_folder
    )
    part_df["audio_path"] = audios
    part_df[["audio_path"] + cfg.save_columns].to_parquet(
        Path(part_metadata_folder,trg_parquet_file + ".parquet"), index=False
    )
    print("Done.")


if __name__ == "__main__":
    main()
