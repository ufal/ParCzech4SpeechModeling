import tarfile
from pathlib import Path

import requests


def download_tar_file(url, output_folder, overwrite):
    """
    Downloads a .tar file from a given URL and saves it to the specified output folder.

    Args:
        url (str): The URL of the .tar file to download.
        output_folder (str): The folder where the downloaded file will be saved.

    Returns:
        str: The path to the downloaded file.
    """

    # Extract the filename from the URL

    # Download the file
    try:
        output_file = Path(output_folder, f"{Path(url).stem}.tar")
        if output_file.exists() and not overwrite:
            print(f"File {output_file.as_posix()} already exists. Skipping download.")
            return output_file

        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an error for bad status codes

        # Write the file to disk
        with open(output_file, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"Download completed. File saved to: {output_file}")
        return output_file

    except requests.exceptions.RequestException as e:
        print(f"Failed to download {output_file}: {e}")
        return None


def extract_tar_file(tar_path):
    """
    Extracts a .tar inside its folder.

    Args:
        tar_path (str): Path to the .tar file.
    """
    # Ensure the output folder exists
    extraction_dir = Path(tar_path).parent

    try:
        # Open the .tar file
        with tarfile.open(tar_path, "r") as tar:
            tar_files = set(tar.getnames())

            # Check if any files are missing in extraction directory
            missing_files = [
                f for f in tar_files 
                if not (extraction_dir / f).exists()
            ]

            if not missing_files:
                print(f"Files already extracted to {extraction_dir}")
                return

            # Extract only if files are missing
            print(f"Extracting {tar_path} to {extraction_dir}")

            tar.extractall(path=tar_path.parent)  # Extract all files
    except tarfile.TarError as e:
        print(f"Failed to extract {tar_path}: {e}")
        raise e
