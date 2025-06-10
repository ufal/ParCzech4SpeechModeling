import multiprocessing
import os
import warnings

import pandas as pd
from pydub import AudioSegment
from tqdm import tqdm

# Suppress Pydub warning about ffmpeg
warnings.filterwarnings("ignore", message="Couldn't find ffmpeg")

def get_mp3_duration(args):
    """Get duration of a single MP3 file in seconds.
    Args should be a tuple of (prefix, file_path)"""
    prefix, file_path = args
    full_path = os.path.join(prefix, file_path) if prefix else file_path
    try:
        audio = AudioSegment.from_mp3(full_path)
        return audio.duration_seconds
    except Exception as e:
        print(f"Error processing {full_path}: {str(e)}")
        return 0

def parallel_get_durations(file_paths, prefix, num_workers=None):
    """Get durations of multiple MP3 files in parallel."""
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()

    # Prepare arguments with prefix
    args = [(prefix, fp) for fp in file_paths]

    with multiprocessing.Pool(processes=num_workers) as pool:
        results = list(tqdm(pool.imap(get_mp3_duration, args), total=len(file_paths)))

    return results

def format_duration(seconds):
    """Format duration in seconds to HH:MM:SS"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"

def calculate_total_duration(mp3_files, prefix="", num_workers=None):
    """
    Calculate total duration of all MP3 files in the list.

    Args:
        mp3_files: List of paths to MP3 files
        prefix: Path prefix to prepend to each file path
        num_workers: Number of parallel processes to use (default: CPU count)

    Returns:
        Total duration in seconds and formatted string (HH:MM:SS)
    """
    durations = parallel_get_durations(mp3_files, prefix, num_workers)
    total_seconds = sum(durations)
    return total_seconds, format_duration(total_seconds)

if __name__ == "__main__":
    meta_tsv = "/lnet/troja/work/people/stankov/parczech4speechmodeling/audioPSP-meta.audioFile.tsv"

    meta_df = pd.read_csv(meta_tsv, sep="\t")
    total_seconds, formatted_duration = calculate_total_duration(meta_df["filePath"].tolist(), prefix="/lnet/troja/work/people/stankov/parczech4speechmodeling/", num_workers=30)
    print(f"Total duration: {total_seconds} seconds")
    print(f"Formatted duration: {formatted_duration}")
