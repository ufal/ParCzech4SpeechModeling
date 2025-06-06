# ParCzech4SpeechModeling

We introduce **ParCzech4Speech 1.0**, a processed version of the ParCzech 4.0 corpus, tailored for speech modeling tasks. The largest variant of the dataset contains 2,695 hours of aligned audio-text data. We combined the sound recordings of Czech parliamentary speeches with their official transcripts. Audio was processed using WhisperX and Wav2Vec 2.0 to extract automated word-level alignments.

---

## Installation

You would need Python 3.10.16.

```
git clone https://github.com/ufal/ParCzech4SpeechModeling.git
cd parczech
pip install -r requirements.txt
pip install -e .
```
---

## Repository Structure

The repository is organized as follows:

- `scripts/` — Scripts for downloading, processing, aligning, and recognizing the ParCzech4 dataset.
- `parczech/` — Common utility functions used in the scripts.
- `configs/` — Configuration files for the recognition and alignment scripts.
- `assets/` — Templates and example files.
- `notebooks/` — Jupyter notebooks for exploring the source and processed datasets, and for further experimentation.

Scripts, configs, and assets are designed to support **parallel processing** of the dataset. For instance, `align.py` is paired with `align_parallel.py`, which is used to submit multiple SLURM jobs for parallel execution of `align.py`. This pattern is used for other scripts as well. All scripts are parametrized using **Hydra**, with configuration files located in the `configs/` directory. Configs for both parallel and single-job execution are stored in the same directory and can be distinguished by the script names they are associated with.

---

## Main Scripts

Below is a list of the core (non-parallel) scripts for processing the ParCzech4 dataset. Parallel versions are wrappers for launching jobs on a SLURM cluster.

- `recognize.py` — Performs ASR on the dataset using WhisperX and Wav2Vec 2.0.
- `align.py` — Aligns recognized texts with official transcripts at the word level.
- `alignment_stats.py` — Computes statistics on alignment quality for a specified version of the dataset (segmented or unsegmented). Note: only a subset of data is used at this stage.
- `create_hf_dataset.py` — Creates a Hugging Face dataset by extracting aligned segments and applying corpus-level filtering.
- `infer_asr.py` — Runs an ASR model on the final dataset for quality verification and removal of misaligned segments.
- `get_duration.py` — Computes durations of dataset segments.

### Pipeline Order

Run the pipeline in the following order:

1. `recognize.py`
2. `align.py`
3. `alignment_stats.py`
4. `create_hf_dataset.py`
5. `infer_asr.py`

---

## Assets

The main assets include:

- `cpu_job.template`
- `gpu_job.template`

These templates are used to create SLURM jobs for parallel scripts. You can specify resource requirements and logging directories in the templates.


