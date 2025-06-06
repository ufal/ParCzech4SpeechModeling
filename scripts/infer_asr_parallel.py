import subprocess
from pathlib import Path

import hydra


@hydra.main(config_path="../configs", config_name="infer_asr_parallel")
def main(cfg):
    Path(cfg.bash_script_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.slurm_logs_dir).mkdir(parents=True, exist_ok=True)

    with open(cfg.template_script) as f:
        template = f.read()


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
        with open(Path(cfg.bash_script_dir, script_name), "w") as f:
            f.write(instance)

        cmd = ["sbatch", Path(cfg.bash_script_dir, script_name).as_posix()]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        print(result.stdout, end="")
    print("Done.")

if __name__ == "__main__":
    main()
