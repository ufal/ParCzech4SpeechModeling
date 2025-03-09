from pathlib import Path

import hydra
import pandas as pd
from parczech.tar_utils import download_tar_file, extract_tar_file
from tqdm import tqdm


@hydra.main(config_path="../configs", config_name="recognize")
def main(cfg):
    Path(cfg.output_folder).mkdir(parents=True, exist_ok=True)

    merged_df = pd.merge(
        pd.read_csv(cfg.meta_data_df, sep="\t"),
        pd.read_csv(cfg.urls_df, sep="\t"),
        on="archiveFileName",
    )
    recognizers = [hydra.utils.instantiate(recognizer) for recognizer in cfg.recognizers]
    debug_mode = cfg.n_debug is not None and cfg.n_debug > 0

    for i, link in tqdm(enumerate(cfg.links), total=len(cfg.links), desc="Processing links"):
        if debug_mode and  i > 0:
            print("Debug mode. Processing only 1 link.")
            break

        tar_path = download_tar_file(link, cfg.output_folder, cfg.overwrite_tar)
        extract_tar_file(tar_path)
        files_lst = merged_df[merged_df["repositoryUrl"] == link]["filePath"].tolist()

        for j, f in tqdm(enumerate(files_lst), total=len(files_lst), desc=f"Processing {Path(link).stem}"):
            if debug_mode and j > cfg.n_debug:
                print(f"Debug mode. Processing only {cfg.n_debug} files.")
                break

            for recognizer in recognizers:
                file_path = Path(cfg.output_folder, f)
                output_path = Path(cfg.output_folder, file_path.parent, f"{file_path.stem}__{recognizer.name}.tsv")

                if output_path.exists() and not cfg.overwrite_recognized:
                    print(f"File {output_path.as_posix()} already exists. Skipping recognition.")
                    continue
                predicted_vertical = recognizer(file_path)
                predicted_vertical_df = pd.DataFrame(predicted_vertical)

                predicted_vertical_df.to_csv(output_path, sep="\t", index=False,)

    print(f"Done. Debug mode is {debug_mode}.")


if __name__ == "__main__":
    main()
