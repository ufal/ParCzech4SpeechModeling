
from multiprocessing import Pool
from pathlib import Path

import hydra
import pandas as pd
from parczech.alignment_utils import align
from parczech.util import get_all_mp3_files
from tqdm import tqdm


def process_job(job):
    print(f"Processing {job['vert_path']} and {job['recognized_path']}")
    result = align(**job)
    return result

@hydra.main(config_path="../configs", config_name="align")
def main(cfg):
    mp3_files = get_all_mp3_files(Path(cfg.recognized_base_path, f"{cfg.year}", f"{cfg.month}"))
    vertical_files = [
        mp3.as_posix().replace(cfg.recognized_base_path, cfg.vertical_base_path).replace(".mp3", ".tsv")
        for mp3 in mp3_files
    ]
    punctuations = set(cfg.punctuations)

    for model_name in cfg.model_names:
        recognized_files = [
            mp3.as_posix().replace(".mp3", f"__{model_name}.tsv")
            for mp3 in mp3_files
        ]
        job_lst = []

        for vertical_file, recognized_file in zip(vertical_files, recognized_files):
            if not Path(recognized_file).exists() or not Path(vertical_file).exists():
                raise ValueError(f"File {recognized_file} does not exist.")

            job = {
                "vert_path": vertical_file,
                "recognized_path": recognized_file,
                "vert_columns": cfg.vert_columns,
                "gap_char": cfg.gap_char,
                "punctuations": punctuations,
                "default_edit_distance": cfg.default_edit_distance
            }
            job_lst.append(job)

        result_dfs = []
        with Pool(processes=cfg.n_cores - 1) as pool:
            result_dfs = list(tqdm(pool.imap(process_job, job_lst),
                          total=len(job_lst),
                          desc=f"Aligning {model_name}"))

        final_df = pd.concat(result_dfs, ignore_index=True)
        output_dir = Path(cfg.output_directory, model_name)
        output_dir.mkdir(parents=True, exist_ok=True)

        final_df.to_parquet(Path(output_dir, f"{cfg.year}_{cfg.month}.parquet"))
        print(f"Saved {model_name} results to {output_dir}/{cfg.year}_{cfg.month}.parquet")


if __name__ == "__main__":
    main()
