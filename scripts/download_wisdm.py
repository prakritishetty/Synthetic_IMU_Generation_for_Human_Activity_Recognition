#!/usr/bin/env python3
import os
import urllib.request
import tarfile

def main():
    url = "https://www.cis.fordham.edu/wisdm/includes/datasets/latest/WISDM_ar_latest.tar.gz"
    download_dir = "./raw_data"
    tar_path = os.path.join(download_dir, "WISDM_ar_latest.tar.gz")
    extract_dir = os.path.join(download_dir, "WISDM")

    os.makedirs(download_dir, exist_ok=True)

    print(f"Downloading WISDM dataset from {url}...")
    # This might take a minute depending on your internet speed
    urllib.request.urlretrieve(url, tar_path)
    print("Download complete.")

    print(f"Extracting to {extract_dir}...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=extract_dir)
    print("Extraction complete.")

    # Clean up the zipped file to save space
    os.remove(tar_path)
    print(f"Done! Raw data is ready at: {extract_dir}")

if __name__ == "__main__":
    main()