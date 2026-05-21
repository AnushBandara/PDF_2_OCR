import shutil
import subprocess
import sys
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

# Folders to delete
# IMPORTANT: models/ is NOT included here
FOLDERS_TO_DELETE = [
    Path("input_pdfs"),
    Path("output_markdown"),
    Path("temp_images"),
    Path("logs"),
]

# Script to run after cleanup
SETUP_SCRIPT = Path("download_deepseek_ocr_model.py")


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

    delete_project_folders()
    run_download_setup_script()

    print("\n" + "=" * 70)
    print("Project reset completed successfully.")
    print("models/ folder was kept safely.")
    print("=" * 70)


if __name__ == "__main__":
    main()