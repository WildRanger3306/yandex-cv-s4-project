import os
import requests
import zipfile
from pathlib import Path
from tqdm import tqdm

DATA_DIR = Path("data/raw")
ZIP_URL = "https://code.s3.yandex.net/deep-learning-cv/images.zip?etag=c508a8c704db94187fe6e0b8219479e9-4"
ZIP_PATH = DATA_DIR / "images.zip"
EXTRACT_PATH = DATA_DIR

def download_file(url: str, dest: Path):
    print(f"Downloading from {url} to {dest}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))
    
    with open(dest, 'wb') as file, tqdm(
        desc=dest.name,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)
    print("Download complete.")

def extract_zip(zip_path: Path, dest: Path):
    print(f"Extracting {zip_path} to {dest}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dest)
    print("Extraction complete.")

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    if not ZIP_PATH.exists():
        download_file(ZIP_URL, ZIP_PATH)
    else:
        print(f"{ZIP_PATH} already exists. Skipping download.")
        
    extract_zip(ZIP_PATH, EXTRACT_PATH)
    print("Phase 1: Downloading and extraction completed.")

if __name__ == "__main__":
    main()
