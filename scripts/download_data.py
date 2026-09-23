"""Download Top Tagging Reference Dataset from Zenodo."""

import argparse
import os
import sys
import urllib.request
import zipfile

ZENODO_URL = "https://zenodo.org/record/4169448/files/top_tagging_2.6M.h5.zip"
ZENODO_MD5 = "d1f840f1c4a58b56e25b8fc1a10dcb34"
EXPECTED_SIZE = 3_699_696_000  # approx 3.4 GB uncompressed
CHUNK_SIZE = 8 * 1024 * 1024  # 8 MB


def download_file(url: str, dest: str) -> None:
    """Download a file with progress reporting.

    Parameters
    ----------
    url : str
        Download URL.
    dest : str
        Destination file path.
    """
    print(f"Downloading {url}")
    print(f"  -> {dest}")
    urllib.request.urlretrieve(url, dest, reporthook=_progress)
    print()


def _progress(block_count: int, block_size: int, total_size: int) -> None:
    downloaded = block_count * block_size
    percent = min(100, int(100 * downloaded / total_size))
    sys.stdout.write(f"\r  {percent}% ({downloaded / 1e9:.1f} GB)")
    sys.stdout.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Top Tagging dataset")
    parser.add_argument(
        "--dest", type=str, default="data/top_tagging",
        help="Destination directory (default: data/top_tagging)",
    )
    parser.add_argument(
        "--keep-zip", action="store_true",
        help="Keep the downloaded zip file after extraction",
    )
    args = parser.parse_args()

    os.makedirs(args.dest, exist_ok=True)
    zip_path = os.path.join(args.dest, "top_tagging_2.6M.h5.zip")
    h5_path = os.path.join(args.dest)

    # Check if already downloaded
    existing_files = [f for f in os.listdir(h5_path) if f.endswith(".h5")]
    if existing_files:
        total = sum(
            os.path.getsize(os.path.join(h5_path, f)) for f in existing_files
        )
        if total > 3_000_000_000:
            print(f"Dataset already exists in {h5_path} ({total / 1e9:.1f} GB)")
            return

    download_file(ZENODO_URL, zip_path)

    print("Extracting (this may take a while)...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(args.dest)

    if not args.keep_zip:
        os.remove(zip_path)

    extracted = os.listdir(args.dest)
    print(f"Done. Extracted {len(extracted)} files to {args.dest}")


if __name__ == "__main__":
    main()
