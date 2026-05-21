import os
import json
import time
import traceback
from pathlib import Path

import fitz  # PyMuPDF
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


# ============================================================
# CONFIGURATION
# ============================================================

PDF_INPUT_DIR = Path("input_pdfs")
IMAGE_OUTPUT_ROOT = Path("temp_images")
MARKDOWN_OUTPUT_DIR = Path("output_markdown")
LOG_DIR = Path("logs")

MODEL_PATH = Path("models/deepseek-ocr")

DPI = 300

# Options:
# "test"     = does not run DeepSeek-OCR. Good for Mac pipeline testing.
# "deepseek" = runs real DeepSeek-OCR. Use this on the NVIDIA GPU PC.
OCR_MODE = "test"

# Options:
# "gpu"  = use NVIDIA CUDA GPU. Use this on RTX 4090 PC.
# "cpu"  = use CPU. Useful for Mac testing with OCR_MODE="test".
# "auto" = use CUDA if available, otherwise CPU.
DEVICE_MODE = "cpu"

DELETE_IMAGES_AFTER_OCR = True

PROMPT = "<image>\n<|grounding|>Convert the document to markdown. "
BASE_SIZE = 1024
IMAGE_SIZE = 640
CROP_MODE = True

CUDA_DEVICE = "0"


# ============================================================
# CREATE REQUIRED FOLDERS
# ============================================================

PDF_INPUT_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
MARKDOWN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

PROGRESS_LOG_PATH = LOG_DIR / "progress.json"
ERROR_LOG_PATH = LOG_DIR / "errors.log"
OCR_TEMP_ROOT = LOG_DIR / "ocr_temp"
PAGE_MARKDOWN_ROOT = LOG_DIR / "page_markdown"

OCR_TEMP_ROOT.mkdir(parents=True, exist_ok=True)
PAGE_MARKDOWN_ROOT.mkdir(parents=True, exist_ok=True)


# ============================================================
# PROGRESS LOG FUNCTIONS
# ============================================================

def load_progress():
    if PROGRESS_LOG_PATH.exists():
        with open(PROGRESS_LOG_PATH, "r", encoding="utf-8") as file:
            return json.load(file)

    return {
        "completed_pages": {},
        "completed_pdfs": []
    }


def save_progress(progress):
    temp_path = PROGRESS_LOG_PATH.with_suffix(".tmp")

    with open(temp_path, "w", encoding="utf-8") as file:
        json.dump(progress, file, indent=2)

    temp_path.replace(PROGRESS_LOG_PATH)


def is_page_completed(progress, pdf_name, page_number):
    return page_number in progress.get("completed_pages", {}).get(pdf_name, [])


def mark_page_completed(progress, pdf_name, page_number):
    if pdf_name not in progress["completed_pages"]:
        progress["completed_pages"][pdf_name] = []

    if page_number not in progress["completed_pages"][pdf_name]:
        progress["completed_pages"][pdf_name].append(page_number)
        progress["completed_pages"][pdf_name].sort()

    save_progress(progress)


def mark_pdf_completed(progress, pdf_name):
    if pdf_name not in progress["completed_pdfs"]:
        progress["completed_pdfs"].append(pdf_name)

    save_progress(progress)


# ============================================================
# ERROR LOGGING
# ============================================================

def log_error(message):
    with open(ERROR_LOG_PATH, "a", encoding="utf-8") as file:
        file.write("\n")
        file.write("=" * 100)
        file.write("\n")
        file.write(message)
        file.write("\n")


# ============================================================
# DEVICE SELECTION
# ============================================================

def get_compute_device():
    mode = DEVICE_MODE.lower().strip()

    if mode not in ["gpu", "cpu", "auto"]:
        raise ValueError(
            f"Invalid DEVICE_MODE: {DEVICE_MODE}. Use 'gpu', 'cpu', or 'auto'."
        )

    if mode == "gpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_DEVICE

        if not torch.cuda.is_available():
            raise RuntimeError(
                "DEVICE_MODE is set to 'gpu', but CUDA is not available.\n"
                "For Mac testing, use OCR_MODE='test' and DEVICE_MODE='cpu'.\n"
                "For RTX 4090 PC, install CUDA-supported PyTorch and NVIDIA drivers."
            )

        device = torch.device("cuda")
        dtype = torch.bfloat16
        print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        return device, dtype

    if mode == "cpu":
        device = torch.device("cpu")
        dtype = torch.float32
        print("Using CPU mode.")
        return device, dtype

    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_DEVICE
        device = torch.device("cuda")
        dtype = torch.bfloat16
        print(f"Auto mode selected CUDA GPU: {torch.cuda.get_device_name(0)}")
        return device, dtype

    device = torch.device("cpu")
    dtype = torch.float32
    print("Auto mode selected CPU because CUDA is not available.")
    return device, dtype


# ============================================================
# PDF IMAGE CONVERSION
# ============================================================

def convert_pdf_page_to_image(pdf_document, page_index, image_path, dpi):
    page = pdf_document.load_page(page_index)

    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)

    pixmap = page.get_pixmap(matrix=matrix, alpha=False)
    pixmap.save(str(image_path))


def get_pdf_page_count(pdf_path):
    with fitz.open(pdf_path) as document:
        return document.page_count


# ============================================================
# MODEL LOADING
# ============================================================

def load_ocr_engine():
    mode = OCR_MODE.lower().strip()

    if mode not in ["test", "deepseek"]:
        raise ValueError(
            f"Invalid OCR_MODE: {OCR_MODE}. Use 'test' or 'deepseek'."
        )

    if mode == "test":
        print("=" * 80)
        print("OCR_MODE is set to TEST.")
        print("DeepSeek-OCR model will NOT be loaded.")
        print("This mode tests PDF conversion, logging, resume, and markdown rebuild.")
        print("=" * 80)

        device = torch.device("cpu")
        return None, None, device

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Local model folder not found: {MODEL_PATH.resolve()}\n"
            f"Please run download_deepseek_ocr_model.py first."
        )

    device, dtype = get_compute_device()

    print("=" * 80)
    print("Loading DeepSeek-OCR model from local folder...")
    print(f"Model path: {MODEL_PATH.resolve()}")
    print(f"Device: {device}")
    print(f"Data type: {dtype}")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_PATH),
        trust_remote_code=True,
        local_files_only=True
    )

    model = AutoModel.from_pretrained(
        str(MODEL_PATH),
        trust_remote_code=True,
        use_safetensors=True,
        local_files_only=True
    )

    model = model.eval().to(device).to(dtype)

    print("Model loaded successfully.")

    return tokenizer, model, device


# ============================================================
# OCR FUNCTION
# ============================================================

def run_ocr_on_image(model, tokenizer, image_path, output_dir, pdf_name, page_number):
    output_dir.mkdir(parents=True, exist_ok=True)

    mode = OCR_MODE.lower().strip()

    if mode == "test":
        return (
            "[TEST MODE]\n\n"
            "DeepSeek-OCR was skipped for this page.\n\n"
            f"PDF: {pdf_name}\n\n"
            f"Page: {page_number}\n\n"
            f"Image path: `{image_path}`\n\n"
            "This confirms that PDF page conversion, page markdown saving, "
            "progress logging, image cleanup, and final markdown rebuilding are working."
        )

    with torch.inference_mode():
        result = model.infer(
            tokenizer,
            prompt=PROMPT,
            image_file=str(image_path),
            output_path=str(output_dir),
            base_size=BASE_SIZE,
            image_size=IMAGE_SIZE,
            crop_mode=CROP_MODE,
            save_results=True,
            test_compress=True
        )

    return result


# ============================================================
# PAGE MARKDOWN FUNCTIONS
# ============================================================

def get_page_markdown_path(pdf_name, page_number):
    page_md_dir = PAGE_MARKDOWN_ROOT / pdf_name
    page_md_dir.mkdir(parents=True, exist_ok=True)

    return page_md_dir / f"page_{page_number:05d}.md"


def save_page_markdown_atomic(page_markdown_path, page_number, markdown_content):
    temp_path = page_markdown_path.with_suffix(".tmp")

    with open(temp_path, "w", encoding="utf-8") as file:
        file.write(f"## Page {page_number}\n\n")
        file.write(str(markdown_content).strip())
        file.write("\n")

    temp_path.replace(page_markdown_path)


def rebuild_final_markdown(pdf_name, total_pages):
    final_markdown_path = MARKDOWN_OUTPUT_DIR / f"{pdf_name}.md"
    temp_final_path = final_markdown_path.with_suffix(".tmp")

    with open(temp_final_path, "w", encoding="utf-8") as final_file:
        final_file.write(f"# {pdf_name}\n\n")

        for page_number in range(1, total_pages + 1):
            page_markdown_path = get_page_markdown_path(pdf_name, page_number)

            if not page_markdown_path.exists():
                final_file.write("\n\n---\n\n")
                final_file.write(f"## Page {page_number}\n\n")
                final_file.write("[PAGE OCR NOT COMPLETED]\n")
                continue

            with open(page_markdown_path, "r", encoding="utf-8") as page_file:
                page_content = page_file.read().strip()

            final_file.write("\n\n---\n\n")
            final_file.write(page_content)
            final_file.write("\n")

    temp_final_path.replace(final_markdown_path)

    print(f"Final markdown rebuilt: {final_markdown_path}")


# ============================================================
# CLEANUP FUNCTIONS
# ============================================================

def safe_delete_file(file_path):
    try:
        if file_path.exists():
            file_path.unlink()
    except Exception:
        log_error(
            f"""
Failed to delete file.

File: {file_path}
Time: {time.strftime("%Y-%m-%d %H:%M:%S")}

Error:
{traceback.format_exc()}
"""
        )


def cleanup_empty_folder(folder_path):
    try:
        if folder_path.exists() and not any(folder_path.iterdir()):
            folder_path.rmdir()
    except Exception:
        pass


def clear_gpu_cache_if_needed(device):
    if device is not None and device.type == "cuda":
        torch.cuda.empty_cache()


# ============================================================
# SINGLE PDF PROCESSING
# ============================================================

def process_single_pdf(pdf_path, model, tokenizer, device, progress):
    pdf_path = Path(pdf_path)
    pdf_name = pdf_path.stem

    print("\n" + "=" * 80)
    print(f"Processing PDF: {pdf_name}")
    print("=" * 80)

    total_pages = get_pdf_page_count(pdf_path)
    print(f"Total pages: {total_pages}")

    if pdf_name in progress.get("completed_pdfs", []):
        print(f"PDF already completed. Rebuilding final markdown only: {pdf_name}")
        rebuild_final_markdown(pdf_name, total_pages)
        return

    pdf_image_dir = IMAGE_OUTPUT_ROOT / pdf_name
    pdf_image_dir.mkdir(parents=True, exist_ok=True)

    pdf_ocr_temp_dir = OCR_TEMP_ROOT / pdf_name
    pdf_ocr_temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        with fitz.open(pdf_path) as pdf_document:
            for page_index in tqdm(range(total_pages), desc=f"OCR {pdf_name}"):
                page_number = page_index + 1
                page_markdown_path = get_page_markdown_path(pdf_name, page_number)

                if is_page_completed(progress, pdf_name, page_number) and page_markdown_path.exists():
                    continue

                image_path = pdf_image_dir / f"page_{page_number:05d}.png"
                page_ocr_output_dir = pdf_ocr_temp_dir / f"page_{page_number:05d}"

                try:
                    if not image_path.exists():
                        convert_pdf_page_to_image(
                            pdf_document=pdf_document,
                            page_index=page_index,
                            image_path=image_path,
                            dpi=DPI
                        )

                    markdown_result = run_ocr_on_image(
                        model=model,
                        tokenizer=tokenizer,
                        image_path=image_path,
                        output_dir=page_ocr_output_dir,
                        pdf_name=pdf_name,
                        page_number=page_number
                    )

                    save_page_markdown_atomic(
                        page_markdown_path=page_markdown_path,
                        page_number=page_number,
                        markdown_content=markdown_result
                    )

                    mark_page_completed(
                        progress=progress,
                        pdf_name=pdf_name,
                        page_number=page_number
                    )

                    if DELETE_IMAGES_AFTER_OCR:
                        safe_delete_file(image_path)

                    clear_gpu_cache_if_needed(device)

                except Exception:
                    error_message = f"""
Page-level error.

PDF: {pdf_name}
Page: {page_number}
Time: {time.strftime("%Y-%m-%d %H:%M:%S")}

Error:
{traceback.format_exc()}
"""
                    log_error(error_message)
                    print(f"Error on {pdf_name}, page {page_number}. Logged and continuing.")

        completed_page_count = len(progress["completed_pages"].get(pdf_name, []))

        rebuild_final_markdown(pdf_name, total_pages)

        if completed_page_count == total_pages:
            mark_pdf_completed(progress, pdf_name)
            print(f"Completed PDF: {pdf_name}")

            if DELETE_IMAGES_AFTER_OCR:
                cleanup_empty_folder(pdf_image_dir)
        else:
            print(f"PDF partially completed: {pdf_name}")
            print(f"Completed pages: {completed_page_count}/{total_pages}")
            print("Final markdown was rebuilt with placeholders for incomplete pages.")

    except Exception:
        error_message = f"""
PDF-level error.

PDF: {pdf_name}
Time: {time.strftime("%Y-%m-%d %H:%M:%S")}

Error:
{traceback.format_exc()}
"""
        log_error(error_message)
        print(f"PDF-level error on {pdf_name}. Logged and continuing.")


# ============================================================
# SHARED PIPELINE RUNNER
# ============================================================

def run_pipeline():
    progress = load_progress()

    tokenizer, model, device = load_ocr_engine()

    pdf_files = sorted(PDF_INPUT_DIR.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found inside: {PDF_INPUT_DIR.resolve()}")
        return

    print(f"Found {len(pdf_files)} PDF files.")

    for pdf_path in pdf_files:
        process_single_pdf(
            pdf_path=pdf_path,
            model=model,
            tokenizer=tokenizer,
            device=device,
            progress=progress
        )

    print("\nAll available PDFs processed.")


# ============================================================
# FASTAPI ENTRY FUNCTION
# ============================================================

def run_pipeline_from_api():
    """
    This function is used by FastAPI.

    It runs the same OCR pipeline, but without depending on the
    command-line main() function.

    FastAPI can call this function in a background thread.
    """
    print("=" * 80)
    print("PDF to Markdown OCR Pipeline Started From FastAPI")
    print("=" * 80)

    print(f"OCR_MODE: {OCR_MODE}")
    print(f"DEVICE_MODE: {DEVICE_MODE}")
    print(f"DPI: {DPI}")
    print(f"DELETE_IMAGES_AFTER_OCR: {DELETE_IMAGES_AFTER_OCR}")

    run_pipeline()

    print("=" * 80)
    print("PDF to Markdown OCR Pipeline Finished From FastAPI")
    print("=" * 80)


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():
    print("=" * 80)
    print("PDF to Markdown OCR Pipeline")
    print("=" * 80)

    print(f"OCR_MODE: {OCR_MODE}")
    print(f"DEVICE_MODE: {DEVICE_MODE}")
    print(f"DPI: {DPI}")
    print(f"DELETE_IMAGES_AFTER_OCR: {DELETE_IMAGES_AFTER_OCR}")

    run_pipeline()


if __name__ == "__main__":
    main()