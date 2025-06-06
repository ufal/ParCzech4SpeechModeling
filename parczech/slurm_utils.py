import subprocess
from pathlib import Path


def push_job(script_dir, script_name, script_content):
    if not Path(script_dir).is_path():
        Path(script_dir).mkdir(parents=True, exist_ok=True)

    with open(Path(script_dir, script_name), "w") as f:
        f.write(script_content)

    script_file = Path(script_dir, script_name).as_posix()

    cmd = ["sbatch", script_file]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    print(result.stdout, end="")


def setup_slurm(slurm_scripts_dir, slurm_log_dir, template_path):
    Path(slurm_scripts_dir).mkdir(parents=True, exist_ok=True)
    Path(slurm_log_dir).mkdir(parents=True, exist_ok=True)
    with open(template_path) as f:
        return f.read()
