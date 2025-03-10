# ParCzech4SpeechModeling

## Recognition of ParCzech4 using WhisperX
Use the following scripts:

* [recognize.py](/scripts/recognize.py) - For given links downloads `.tar` files, extracts them into the common directory and runs the recognition for a recognizers specified in [recognize.yaml](/configs/recognize.yaml). For debug purposes you can set `n_debug` to a number greater than 0, this corresponds to the number of files in the first link that will be recognized by all recognizers in the config. Required parameters are `links` and `output_folder`.
* [recognize_parallel.py](/scripts/recognize_parallel.py) - This is a wrapper for `recognize.py` that given all links creates `n_jobs` slurm jobs with `recognize.py` script with its own config [recognize_parallel.yaml](/configs/recognize_parallel.yaml). Recognition is done on the GPU cluster. Use `n_debug` for debugging purposes, it will be propagated to `recognize.py`. The default number of slurm jobs is specified by `n_jobs` in the config, with -1 corresponding to the number of links (each `recognize.py` job will process one link). Required parameter is `job_name` that will be used as a prefix for the slurm jobs. 

The results will be stored in the `output_folder` specified in the [recognize_parallel.yaml](/configs/recognize_parallel.yaml), that will be propagated to `recognize.py`. As for now it is `/lnet/troja/work/people/stankov/parczech4speechmodeling`. There you can find:

* `.tar` files for each downloaded quarter.
* `audioPSP-meta.audioFile.tsv` and `audioPSP-meta.quarterArchive.tsv`, two files with the metadata about the original ParCzech4 dataset (audio information only).
* `audio` directory with all audio files and transcripts. Relative path to the audio can be find in `audioPSP-meta.audioFile.tsv` in the column `filePath`. The recognized texts will share the same relative path and will be stored in `.tsv` files. The suffix in the naming displays the recognizer type.
