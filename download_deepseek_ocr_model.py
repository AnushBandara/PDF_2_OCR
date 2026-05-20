from pathlib import Path
from transformers import AutoModel, AutoTokenizer


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "deepseek-ai/DeepSeek-OCR"

# Project folders
INPUT_PDFS_DIR = Path("input_pdfs")
OUTPUT_MARKDOWN_DIR = Path("output_markdown")
TEMP_IMAGES_DIR = Path("temp_images")
LOGS_DIR = Path("logs")
MODELS_DIR = Path("models")

# Model save location
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
# DOWNLOAD AND SAVE MODEL
# ============================================================

def download_tokenizer():
    print("\nDownloading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True
    )

    tokenizer.save_pretrained(MODEL_SAVE_DIR)

    print("Tokenizer saved successfully.")


def download_model():
    print("\nDownloading model...")

    model = AutoModel.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        use_safetensors=True
    )

    model.save_pretrained(MODEL_SAVE_DIR)

    print("Model saved successfully.")


def main():
    print("=" * 70)
    print("DeepSeek-OCR Model Downloader")
    print("=" * 70)

    create_project_folders()

    download_tokenizer()
    download_model()

    print("\n" + "=" * 70)
    print("Download completed successfully.")
    print(f"Model saved at: {MODEL_SAVE_DIR.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    main()