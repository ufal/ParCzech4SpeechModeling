from pathlib import Path

import hydra
from parczech.slurm_utils import push_job, setup_slurm


@hydra.main(config_path="../configs", config_name="infer_asr_parallel")
def main(cfg):
    template = setup_slurm(
        slurm_scripts_dir=cfg.bash_script_dir,
        slurm_log_dir=cfg.slurm_logs_dir,
        template_path=cfg.template_script,
    )

    for parquet_file in Path(cfg.parquet_dir).glob("*.parquet"):
        script_name = f"recognize_{parquet_file.stem}.sh"
        if Path(cfg.params.output_dir, cfg.params.target_class.split(".")[-1], parquet_file.name).exists():
            print(f"Output file {parquet_file.name} already exists. Skipping.")
            continue

        script_params = "df_path={dp} num_workers={nw} model._target_={mt} output_dir={od}".format(  # noqa: UP032
            dp=parquet_file.as_posix(),
            nw=cfg.cpus - 2,
            mt=cfg.params.target_class,
            od=cfg.params.output_dir,
        )

        instance = template.format(
            job_name=f"{cfg.job_name}",
            job_id=parquet_file.stem,
            cpus=cfg.cpus,
            mem=cfg.mem,
            n_gpus=cfg.n_gpus,
            script=cfg.script,
            params=script_params,
        )

        push_job(
            script_dir=cfg.bash_script_dir,
            script_name=script_name,
            script_content=instance,
        )
    print("Done.")

if __name__ == "__main__":
    main()
