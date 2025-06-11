from pathlib import Path

import hydra


@hydra.main(config_path="../configs", config_name="create_version_with_stats")
def main(cfg):
    Path(cfg.output_directory).mkdir(parents=True, exist_ok=True)
    print(f"Overwriting: {cfg.overwrite}")

    for model_name in cfg.model_names:
        print(f"Processing {model_name}...")
        parquet_dir = Path(cfg.alignment_dir, model_name)
        if not parquet_dir.exists():
            raise ValueError(f"Directory {parquet_dir} does not exist.")

        Path(cfg.output_directory, model_name).mkdir(parents=True, exist_ok=True)
        output_parquet = Path(cfg.output_directory, model_name, cfg.parquet_file)
        if output_parquet.exists() and not cfg.overwrite:
            print(f"File {output_parquet} already exists. Skipping.")
            continue

        segmenter = hydra.utils.instantiate(cfg.segmenter)
        segments_df = segmenter.extract_segments(Path(parquet_dir, cfg.parquet_file))
        segments_df.to_parquet(output_parquet, index=False)

    print("Done.")


if __name__ == "__main__":
    main()
