import tarfile
import requests
from pathlib import Path
import hydra
from tqdm import tqdm


def download_tar_file(url, output_folder):
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
        if output_file.exists():
            print(f"File already exists. Skipping download.")
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
    Extracts a .tar file to the specified output folder.

    Args:
        tar_path (str): Path to the .tar file.

    Returns:
        bool: True if extraction is successful, False otherwise.
    """
    # Ensure the output folder exists


    try:
        # Open the .tar file
        with tarfile.open(tar_path, "r") as tar:
            tar.extractall(path=tar_path.parent)  # Extract all files
            print("Extraction completed successfully.")
    except tarfile.TarError as e:
        print(f"Failed to extract {tar_path}: {e}")
        raise e


@hydra.main(config_path="../configs", config_name="download_data")
def main(cfg):
    if cfg.output_folder is None:
        Path(cfg.output_folder).mkdir(parents=True, exist_ok=True)
    
    for link in tqdm(cfg.links):
        print(f"Downloading {link}...")
        p = download_tar_file(link, cfg.output_folder)
        extract_tar_file(p)
    

if __name__ == "__main__":
    main()
