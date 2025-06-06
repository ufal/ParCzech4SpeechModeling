import hydra
import pandas as pd
from parczech.slurm_utils import push_job, setup_slurm


def divide_into_chunks(lst, n_parts):
    if n_parts == -1:
        return [[x] for x in lst]
    parts = [[] for i in range(n_parts)]
    for i, item in enumerate(lst):
        parts[i % n_parts].append(item)
    return parts


@hydra.main(config_path="../configs", config_name="recognize_parallel")
def main(cfg):
    template = setup_slurm(
        slurm_scripts_dir=cfg.bash_script_dir,
        slurm_log_dir=cfg.slurm_logs_dir,
        template_path=cfg.template_script,
    )

    metadata_df = pd.read_csv(cfg.metadata_df, sep="\t").sort_values("cntFiles", ascending=False)
    unique_links = metadata_df["repositoryUrl"].unique()

    for i, link_chunk in enumerate(divide_into_chunks(unique_links, cfg.n_jobs)):
        if cfg.n_debug is not None and cfg.n_debug > 0 and i > cfg.n_debug:
            print(f"Debug mode. Processing only {cfg.n_debug} link.")
            break
        script_name = f"recognize_{i}.sh"

        script_params = "overwrite_recognized={ore} links={l} output_folder={of} n_debug={nd}".format(
            ore=cfg.params.overwrite_recognized,
            l='"[' + ",".join(link_chunk) + ']"',
            of=cfg.params.output_folder,
            nd=cfg.params.n_debug,
        )

        instance = template.format(
            job_name=cfg.job_name,
            job_id=i,
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

    print(f"Done. Debug mode is {cfg.n_debug}.")

if __name__ == "__main__":
    main()
