import shutil
import subprocess
import sys
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

# Folders to delete
# IMPORTANT:
# - models/ is NOT included because it contains the downloaded DeepSeek-OCR model.
# - templates/ is NOT included because it contains the FastAPI UI HTML files.
# - static/ is NOT included because it may contain UI assets.
FOLDERS_TO_DELETE = [
    Path("input_pdfs"),
    Path("output_markdown"),
    Path("temp_images"),
    Path("logs"),
]

# Script to run after cleanup
SETUP_SCRIPT = Path("download_deepseek_ocr_model.py")


# ============================================================
# SAFETY CHECK
# ============================================================

def confirm_project_root():
    """
    Basic safety check to make sure this script is being run
    from the correct project folder.
    """

    required_files = [
        Path("download_deepseek_ocr_model.py"),
        Path("pdf_to_markdown_pipeline.py"),
    ]

    missing_files = []

    for file_path in required_files:
        if not file_path.exists():
            missing_files.append(str(file_path))

    if missing_files:
        raise RuntimeError(
            "This does not look like the correct project root folder.\n"
            "Missing required files:\n"
            + "\n".join(missing_files)
            + "\n\nPlease run this script from the PDF_2_OCR project root."
        )


# ============================================================
# DELETE FOLDERS
# ============================================================

def delete_folder(folder_path: Path):
    if folder_path.exists():
        print(f"Deleting: {folder_path.resolve()}")
        shutil.rmtree(folder_path)
        print(f"Deleted: {folder_path}")
    else:
        print(f"Skipping, not found: {folder_path}")


def delete_project_folders():
    print("=" * 70)
    print("Deleting generated project folders")
    print("=" * 70)

    for folder in FOLDERS_TO_DELETE:
        delete_folder(folder)

    print("Folder cleanup completed.")


# ============================================================
# RUN SETUP SCRIPT
# ============================================================

def run_download_setup_script():
    if not SETUP_SCRIPT.exists():
        raise FileNotFoundError(
            f"{SETUP_SCRIPT} was not found. "
            "Please make sure download_deepseek_ocr_model.py exists in the project root."
        )

    print("\n" + "=" * 70)
    print(f"Running: {SETUP_SCRIPT}")
    print("=" * 70)

    result = subprocess.run(
        [sys.executable, str(SETUP_SCRIPT)],
        check=False
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"{SETUP_SCRIPT} failed with exit code {result.returncode}"
        )

    print("\nSetup script completed successfully.")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("Project Folder Reset Tool")
    print("=" * 70)

    confirm_project_root()
    delete_project_folders()
    run_download_setup_script()

    print("\n" + "=" * 70)
    print("Project reset completed successfully.")
    print("models/ folder was kept safely.")
    print("templates/ and static/ folders were kept safely.")
    print("Fresh input/output/log/temp folders were recreated.")
    print("=" * 70)


if __name__ == "__main__":
    main()