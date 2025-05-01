import subprocess
from pathlib import Path

import hydra


@hydra.main(config_path="../configs", config_name="align_parallel")
def main(cfg):
    Path(cfg.bash_script_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.slurm_logs_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.output_directory).mkdir(parents=True, exist_ok=True)

    with open(cfg.template_script) as f:
        template = f.read()

    job_id = 0

    for year_dir in Path(cfg.recognized_base_path).iterdir():
        for month_dir in year_dir.iterdir():
                year = year_dir.name
                month = month_dir.name

                script_name = f"recognize_{year}{month}.sh"

                instance = template.format(
                    id=job_id,
                    year=year,
                    month=month,
                    model_names=cfg.model_names,
                    output_directory=cfg.output_directory,
                    slurm_logs_dir=cfg.slurm_logs_dir,
                )
                with open(Path(cfg.bash_script_dir, script_name), "w") as f:
                    f.write(instance)
                cmd = ["sbatch", Path(cfg.bash_script_dir, script_name).as_posix()]
                result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
                print(result.stdout, end="")
                job_id += 1


if __name__ == "__main__":
    main()
