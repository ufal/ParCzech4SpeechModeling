from pathlib import Path

import hydra
from parczech.slurm_utils import push_job, setup_slurm


@hydra.main(config_path="../configs", config_name="align_parallel")
def main(cfg):
    Path(cfg.output_directory).mkdir(parents=True, exist_ok=True)
    template = setup_slurm(
        slurm_scripts_dir=cfg.bash_script_dir,
        slurm_log_dir=cfg.slurm_logs_dir,
        template_path=cfg.template_script,
    )

    for year_dir in Path(cfg.recognized_base_path).iterdir():
        for month_dir in year_dir.iterdir():
            year = year_dir.name
            month = month_dir.name

            id = f"{year}_{month}"
            script_name = f"{id}.sh"
            params = "year={y} month={m} model_names={mn} output_directory={od}".format(  # noqa: UP032
                y=year,
                m=month,
                mn=cfg.model_names,
                od=cfg.output_directory,
            )

            instance = template.format(
                job_id=id,
                job_name=cfg.job_name,
                cpus=cfg.cpus,
                mem=cfg.mem,
                script=cfg.script,
                params=params,
            )

            push_job(
                script_dir=cfg.bash_script_dir,
                script_name=script_name,
                script_content=instance,
            )
    print("Done.")


if __name__ == "__main__":
    main()
