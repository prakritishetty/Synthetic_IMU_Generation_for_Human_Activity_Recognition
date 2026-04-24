import os
import urllib.request
import tarfile

def download_wisdm(output_dir="raw_data/WISDM"):
    url = "https://www.cis.fordham.edu/wisdm/includes/datasets/latest/WISDM_ar_latest.tar.gz"
    os.makedirs(output_dir, exist_ok=True)
    tar_path = os.path.join(output_dir, "WISDM_ar_latest.tar.gz")
    
    if not os.path.exists(tar_path):
        print(f"Downloading WISDM from {url}...")
        urllib.request.urlretrieve(url, tar_path)
        print("Download complete.")
    else:
        print("Archive already exists. Skipping download.")
        
    print("Extracting archive...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=output_dir)
        
    wisdm_txt_path = os.path.join(output_dir, "WISDM_ar_v1.1", "WISDM_ar_v1.1_raw.txt")
    if os.path.exists(wisdm_txt_path):
        print(f"Success! WISDM raw data located at: {wisdm_txt_path}")
    else:
        print(f"Could not find the extracted txt file at {wisdm_txt_path}. Please check extraction manually.")

if __name__ == "__main__":
    download_wisdm()
