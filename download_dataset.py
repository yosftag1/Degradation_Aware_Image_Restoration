"""Download and prepare DIV2K dataset."""

import os
import urllib.request
import zipfile
from pathlib import Path
import argparse


def download_div2k(dataset_dir, split="all", verbose=True):
    """Download DIV2K dataset.
    
    Args:
        dataset_dir: Directory to save dataset
        split: "train", "valid", or "all"
        verbose: Print download progress
    """
    os.makedirs(dataset_dir, exist_ok=True)
    
    base_url = "http://data.cv.snu.ac.kr:8008/datasets/DIV2K"
    
    splits = {
        "train": "DIV2K_train_HR.zip",
        "valid": "DIV2K_valid_HR.zip",
    }
    
    if split == "all":
        splits_to_download = list(splits.keys())
    else:
        splits_to_download = [split]
    
    for split_name in splits_to_download:
        filename = splits[split_name]
        url = f"{base_url}/{filename}"
        filepath = os.path.join(dataset_dir, filename)
        
        if os.path.exists(filepath.replace(".zip", "")):
            print(f"✓ {filename} already extracted, skipping download")
            continue
        
        if os.path.exists(filepath):
            print(f"✓ {filename} already downloaded")
        else:
            print(f"Downloading {filename}...")
            try:
                urllib.request.urlretrieve(
                    url, filepath,
                    reporthook=_download_progress if verbose else None
                )
                print(f"✓ Downloaded {filename}")
            except Exception as e:
                print(f"✗ Failed to download {filename}: {e}")
                continue
        
        # Extract
        print(f"Extracting {filename}...")
        try:
            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                zip_ref.extractall(dataset_dir)
            print(f"✓ Extracted {filename}")
            os.remove(filepath)  # Remove zip after extraction
        except Exception as e:
            print(f"✗ Failed to extract {filename}: {e}")


def _download_progress(block_num, block_size, total_size):
    """Print download progress."""
    downloaded = block_num * block_size
    percent = min(100, int(100.0 * downloaded / total_size))
    print(f"\r  Progress: {percent}%", end="", flush=True)


def verify_dataset(dataset_dir):
    """Verify DIV2K dataset structure."""
    print("\nVerifying dataset structure...")
    
    train_dir = os.path.join(dataset_dir, "DIV2K_train_HR")
    valid_dir = os.path.join(dataset_dir, "DIV2K_valid_HR")
    
    train_count = len(list(Path(train_dir).glob("*.png"))) if os.path.exists(train_dir) else 0
    valid_count = len(list(Path(valid_dir).glob("*.png"))) if os.path.exists(valid_dir) else 0
    
    print(f"Training images: {train_count} (expected 800)")
    print(f"Validation images: {valid_count} (expected 100)")
    
    if train_count == 800 and valid_count == 100:
        print("✓ Dataset verification successful!")
        return True
    else:
        print("✗ Dataset verification failed!")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download DIV2K dataset")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="./data/DIV2K",
        help="Directory to save DIV2K dataset"
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "valid", "all"],
        default="all",
        help="Dataset split to download"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Only verify existing dataset"
    )
    
    args = parser.parse_args()
    
    if args.verify:
        verify_dataset(args.dataset_dir)
    else:
        print(f"Downloading DIV2K dataset to {args.dataset_dir}...")
        download_div2k(args.dataset_dir, split=args.split)
        verify_dataset(args.dataset_dir)
