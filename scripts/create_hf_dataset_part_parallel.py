from pathlib import Path

import hydra
from parczech.slurm_utils import push_job, setup_slurm


@hydra.main(config_path="../configs", config_name="create_hf_dataset_part_parallel")
def main(cfg):
    template = setup_slurm(
        slurm_scripts_dir=cfg.slurm_scripts_dir,
        slurm_log_dir=cfg.slurm_log_dir,
        template_path=cfg.template,
    )

    for file in Path(cfg.segment_dir).glob("*.parquet"):
        # params = f"parquet_file={file.name} model_name={cfg.model_name} overwrite={cfg.overwrite}"
        params = "out_version={ov} parquet_file={pf} inp_version={iv} model_name={mn} overwrite={ow}".format(  # noqa: UP032
            ov=cfg.params.out_version,
            pf=file.name,
            iv=cfg.params.inp_version,
            mn=cfg.params.model_name,
            ow=cfg.params.overwrite,
        )
        if cfg.params.get("filter", None) is not None:
            params += f' \'filter="{cfg.params.filter}"\''

        script = template.format(
            job_name=f"{cfg.job_name}",
            job_id=file.stem,
            cpus=cfg.cpus,
            mem=cfg.mem,
            script=cfg.script,
            params=params
        )

        push_job(
            script_dir=cfg.slurm_scripts_dir,
            script_name=f"{file.stem}.sh",
            script_content=script,
        )
    print("Done.")


if __name__ == "__main__":
    main()