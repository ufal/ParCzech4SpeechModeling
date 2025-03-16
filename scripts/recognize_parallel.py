import subprocess
import time

from pathlib import Path

import hydra
import pandas as pd


def divide_into_chunks(lst, n_parts):
    if n_parts == -1:
        return [[x] for x in lst]
    parts = [[] for i in range(n_parts)]
    for i, item in enumerate(lst):
        parts[i % n_parts].append(item)
    return parts

def get_best_gpu(priority_list):
    """Check if a job with specified GPU memory can start immediately."""
    for gpu in priority_list:
        try:
            cmd = ["sinfo", "-o", "%N %G", "--state=idle"]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
            if gpu in result.stdout:
                return gpu
        except Exception as e:
            print(f"Error checking job submission: {e}")
            continue
    return None


@hydra.main(config_path="../configs", config_name="recognize_parallel")
def main(cfg):
    Path(cfg.bash_script_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.slurm_logs_dir).mkdir(parents=True, exist_ok=True)

    metadata_df = pd.read_csv(cfg.metadata_df, sep="\t").sort_values("cntFiles", ascending=False)
    unique_links = metadata_df["repositoryUrl"].unique()

    with open(cfg.template_script) as f:
        template = f.read()

    used_gpus = []
    used_gpus_index = 0
    for i, link_chunk in enumerate(divide_into_chunks(unique_links, cfg.n_jobs)):
        if cfg.n_debug is not None and cfg.n_debug > 0 and i > cfg.n_debug:
            print(f"Debug mode. Processing only {cfg.n_debug} link.")
            break
        script_name = f"recognize_{i}.sh"
        links = '"[' + ",".join(link_chunk) + ']"'
        gpu = get_best_gpu(cfg.gpu_priority)

        # If no GPU is available, request for already requested GPU
        # since will be available in 20 hours
        if gpu is None:
            gpu = used_gpus[used_gpus_index]
            used_gpus_index = (used_gpus_index + 1) % len(used_gpus)
        else:
            used_gpus.append(gpu)

        instance = template.format(
            id=i,
            gpu=gpu,
            links=links,
            n_debug=cfg.n_debug,
            config=cfg.job_config,
            slurm_logs_dir=cfg.slurm_logs_dir,
            output_folder=cfg.output_folder,
            overwrite_recognized=cfg.overwrite_recognized,
        )
        with open(Path(cfg.bash_script_dir, script_name), "w") as f:
            f.write(instance)

        cmd = ["sbatch", Path(cfg.bash_script_dir, script_name).as_posix()]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        print(result.stdout, end="")
        time.sleep(2)
    print(f"Done. Debug mode is {cfg.n_debug}.")

if __name__ == "__main__":
    main()
