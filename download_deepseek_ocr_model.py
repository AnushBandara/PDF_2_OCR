from pathlib import Path
from huggingface_hub import snapshot_download


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "deepseek-ai/DeepSeek-OCR"

INPUT_PDFS_DIR = Path("input_pdfs")
OUTPUT_MARKDOWN_DIR = Path("output_markdown")
TEMP_IMAGES_DIR = Path("temp_images")
LOGS_DIR = Path("logs")
MODELS_DIR = Path("models")

MODEL_SAVE_DIR = MODELS_DIR / "deepseek-ocr"


# ============================================================
# CREATE PROJECT FOLDERS
# ============================================================

def create_project_folders():
    folders = [
        INPUT_PDFS_DIR,
        OUTPUT_MARKDOWN_DIR,
        TEMP_IMAGES_DIR,
        LOGS_DIR,
        MODELS_DIR,
        MODEL_SAVE_DIR,
    ]

    print("Creating required project folders...")

    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)
        print(f"Ready: {folder.resolve()}")

    print("All folders are ready.")


# ============================================================
# CHECK MODEL AVAILABILITY
# ============================================================

def is_model_already_downloaded():
    """
    Checks whether the DeepSeek-OCR model files already exist locally.
    """

    required_files = [
        "config.json",
    ]

    for file_name in required_files:
        file_path = MODEL_SAVE_DIR / file_name

        if not file_path.exists():
            return False

    model_weight_files = list(MODEL_SAVE_DIR.glob("*.safetensors"))

    if len(model_weight_files) == 0:
        return False

    return True


# ============================================================
# DOWNLOAD MODEL FILES
# ============================================================

def download_model_snapshot():
    if is_model_already_downloaded():
        print("\nDeepSeek-OCR model already exists locally.")
        print("Skipping download.")
        print(f"Model location: {MODEL_SAVE_DIR.resolve()}")
        return

    print("\nDeepSeek-OCR model not found locally.")
    print("Downloading DeepSeek-OCR model files...")

    snapshot_download(
        repo_id=MODEL_NAME,
        local_dir=MODEL_SAVE_DIR,
        local_dir_use_symlinks=False,
        resume_download=True
    )

    print("Model files downloaded successfully.")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("DeepSeek-OCR Model Downloader")
    print("=" * 70)

    create_project_folders()
    download_model_snapshot()

    print("\n" + "=" * 70)
    print("Setup completed.")
    print(f"Model folder: {MODEL_SAVE_DIR.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    main()