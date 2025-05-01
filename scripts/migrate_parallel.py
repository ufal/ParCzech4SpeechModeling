import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

from hydra import main
from omegaconf import DictConfig


@main(config_path="../configs", config_name="migrate_parallel")
def main(cfg: DictConfig) -> None:
    # Read the sbatch template file.
    slurm_logs_dir = Path(cfg.slurm_logs_dir, "migrate")
    slurm_logs_dir.mkdir(parents=True, exist_ok=True)
    scripts_dir = Path(cfg.scripts_dir, "migrate")
    scripts_dir.mkdir(parents=True, exist_ok=True)

    with open(cfg.template) as template_file:
        template_str = template_file.read()

    for idx, directory in enumerate(cfg.directories):
        # Fill the template using the directory and other job arguments defined in the config.
        input_dir = Path(cfg.inp_base_dir, directory)
        output_dir = Path(cfg.out_base_dir, directory)

        if cfg.get("start_id", None) is not None:
            idx = cfg.start_id + idx

        if not input_dir.exists():
            raise ValueError(f"Directory {input_dir} does not exist. ")

        filled_script = template_str.format(
            id=idx,
            srcdir=input_dir.as_posix(),
            destdir=output_dir.as_posix(),
            slurm_logs_dir=slurm_logs_dir,
            debug=cfg.debug,
        )

        script_path = Path(scripts_dir, f"sbatch_{idx}.sh")

        with open(script_path, "w") as f:
            f.write(filled_script)

        # Submit the job using sbatch.
        cmd = ["sbatch", script_path.as_posix()]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        print(result.stdout, end="")
        if result.returncode != 0:
            print(f"Failed to submit job for {directory}")

if __name__ == "__main__":
    main()
