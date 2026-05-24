"""Download all indices from Hugging Face datasets."""
import os
from huggingface_hub import hf_hub_download, snapshot_download

# Script is now in scripts/ directory, so go up one level to reach root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_DIR = os.path.join(BASE_DIR, "index")
FAISS_DIR = os.path.join(INDEX_DIR, "faiss_index")
os.makedirs(FAISS_DIR, exist_ok=True)

TEXT_REPO = "KirantiLoh/steam-text-index"
IMAGE_REPO = "Frmeta/tk-tbi-steam-index-img-only-hnsw"


def download_text_index():
    print("Downloading text index (BM25 + FAISS)...")
    snapshot_download(
        repo_id=TEXT_REPO,
        repo_type="dataset",
        local_dir=INDEX_DIR,
        local_dir_use_symlinks=False,
        ignore_patterns=[".gitattributes"],
    )
    print("Text index downloaded.")


def download_image_index():
    print("Downloading image HNSW index...")
    hf_hub_download(
        repo_id=IMAGE_REPO,
        filename="steam_games.index",
        repo_type="dataset",
        local_dir=INDEX_DIR,
        local_dir_use_symlinks=False,
    )
    hf_hub_download(
        repo_id=IMAGE_REPO,
        filename="metadata.parquet",
        repo_type="dataset",
        local_dir=INDEX_DIR,
        local_dir_use_symlinks=False,
    )
    print("Image HNSW index downloaded.")


if __name__ == "__main__":
    download_text_index()
    download_image_index()
    print("All indices downloaded to", INDEX_DIR)
