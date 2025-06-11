from pathlib import Path

import hydra
from parczech.slurm_utils import push_job, setup_slurm


@hydra.main(config_path="../configs", config_name="create_version_with_stats_parallel")
def main(cfg):
    file_sets = [
        set(f.stem for f in Path(cfg.alignment_dir, model_name).glob("*.parquet"))
        for model_name in cfg.model_names
    ]
    first = file_sets[0]
    if not all(first == file_set for file_set in file_sets[1:]):
        raise ValueError("Files in the directories do not match.")

    template = setup_slurm(
        slurm_scripts_dir=cfg.slurm_scripts_dir,
        slurm_log_dir=cfg.slurm_log_dir,
        template_path=cfg.template,
    )

    for file in Path(cfg.alignment_dir, cfg.model_names[0]).glob("*.parquet"):
        model_names = "[" + ",".join(cfg.model_names) + "]"

        script = template.format(
            job_name=cfg.job_name,
            job_id=file.stem,
            cpus=cfg.cpus,
            mem=cfg.mem,
            script=cfg.script,
            params=f'parquet_file={file.name} model_names="{model_names}" overwrite={cfg.overwrite} output_directory="{cfg.output_directory}"',
        )

        push_job(
            script_dir=cfg.slurm_scripts_dir,
            script_name=f"{file.stem}.sh",
            script_content=script,
        )


if __name__ == "__main__":
    main()