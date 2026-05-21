import json
import shutil
import threading
import zipfile
from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from pdf_to_markdown_pipeline import (
    PDF_INPUT_DIR,
    MARKDOWN_OUTPUT_DIR,
    LOG_DIR,
    PROGRESS_LOG_PATH,
    get_pdf_page_count,
    run_pipeline_from_api,
)


# ============================================================
# FASTAPI SETUP
# ============================================================

app = FastAPI(title="PDF to Markdown OCR UI")

templates = Jinja2Templates(directory="templates")

STATIC_DIR = Path("static")
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


# ============================================================
# GLOBAL JOB STATE
# ============================================================

job_state: Dict[str, Any] = {
    "running": False,
    "message": "Idle",
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def ensure_required_folders():
    PDF_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    MARKDOWN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def read_progress():
    if not PROGRESS_LOG_PATH.exists():
        return {
            "completed_pages": {},
            "completed_pdfs": []
        }

    with open(PROGRESS_LOG_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def get_pdf_status():
    progress = read_progress()

    pdf_status = []

    for pdf_path in sorted(PDF_INPUT_DIR.glob("*.pdf")):
        pdf_name = pdf_path.stem

        try:
            total_pages = get_pdf_page_count(pdf_path)
        except Exception:
            total_pages = 0

        completed_pages = progress.get("completed_pages", {}).get(pdf_name, [])
        completed_count = len(completed_pages)

        percentage = 0
        if total_pages > 0:
            percentage = round((completed_count / total_pages) * 100, 2)

        output_md_path = MARKDOWN_OUTPUT_DIR / f"{pdf_name}.md"

        pdf_status.append({
            "pdf_name": pdf_name,
            "file_name": pdf_path.name,
            "total_pages": total_pages,
            "completed_pages": completed_count,
            "percentage": percentage,
            "is_completed": pdf_name in progress.get("completed_pdfs", []),
            "download_ready": output_md_path.exists(),
            "download_url": f"/download/{pdf_name}" if output_md_path.exists() else None,
        })

    return pdf_status


def clear_folder_contents(folder_path: Path):
    folder_path.mkdir(parents=True, exist_ok=True)

    for item in folder_path.iterdir():
        if item.is_file() or item.is_symlink():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)


def clear_runtime_files():
    """
    Clears uploaded PDFs, markdown outputs, temporary images, and logs.
    Does NOT delete models/, templates/, static/, or source code.
    """
    clear_folder_contents(PDF_INPUT_DIR)
    clear_folder_contents(MARKDOWN_OUTPUT_DIR)
    clear_folder_contents(LOG_DIR)

    temp_images_dir = Path("temp_images")
    clear_folder_contents(temp_images_dir)


def create_markdown_zip():
    """
    Creates a ZIP file containing all generated markdown files.
    """
    ensure_required_folders()

    markdown_files = sorted(MARKDOWN_OUTPUT_DIR.glob("*.md"))

    if not markdown_files:
        return None

    zip_path = MARKDOWN_OUTPUT_DIR / "all_markdown_files.zip"

    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for markdown_file in markdown_files:
            zip_file.write(
                markdown_file,
                arcname=markdown_file.name
            )

    return zip_path


def background_ocr_job():
    try:
        job_state["running"] = True
        job_state["message"] = "OCR conversion is running..."

        run_pipeline_from_api()

        job_state["message"] = "OCR conversion completed."
    except Exception as error:
        job_state["message"] = f"OCR conversion failed: {error}"
    finally:
        job_state["running"] = False


# ============================================================
# ROUTES
# ============================================================

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    ensure_required_folders()

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )


@app.post("/upload")
async def upload_pdfs(files: list[UploadFile] = File(...)):
    ensure_required_folders()

    uploaded_files = []

    for file in files:
        if not file.filename:
            continue

        if not file.filename.lower().endswith(".pdf"):
            continue

        save_path = PDF_INPUT_DIR / file.filename

        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        uploaded_files.append(file.filename)

    return JSONResponse({
        "message": "PDF files uploaded successfully.",
        "uploaded_files": uploaded_files
    })


@app.post("/start")
def start_conversion():
    ensure_required_folders()

    if job_state["running"]:
        return JSONResponse({
            "message": "OCR conversion is already running.",
            "running": True
        })

    thread = threading.Thread(target=background_ocr_job, daemon=True)
    thread.start()

    return JSONResponse({
        "message": "OCR conversion started.",
        "running": True
    })


@app.post("/clear")
def clear_all_files():
    ensure_required_folders()

    if job_state["running"]:
        return JSONResponse(
            {
                "message": "Cannot clear files while OCR conversion is running.",
                "cleared": False
            },
            status_code=409
        )

    clear_runtime_files()

    job_state["message"] = "All uploaded, generated, temporary, and log files were cleared."

    return JSONResponse({
        "message": "All existing files cleared successfully.",
        "cleared": True
    })


@app.get("/progress")
def progress():
    ensure_required_folders()

    markdown_files = list(MARKDOWN_OUTPUT_DIR.glob("*.md"))

    return JSONResponse({
        "running": job_state["running"],
        "message": job_state["message"],
        "has_download_all": len(markdown_files) > 0,
        "download_all_url": "/download-all" if markdown_files else None,
        "pdfs": get_pdf_status()
    })


@app.get("/download/{pdf_name}")
def download_markdown(pdf_name: str):
    markdown_path = MARKDOWN_OUTPUT_DIR / f"{pdf_name}.md"

    if not markdown_path.exists():
        return JSONResponse(
            {"error": "Markdown file is not ready yet."},
            status_code=404
        )

    return FileResponse(
        path=markdown_path,
        filename=f"{pdf_name}.md",
        media_type="text/markdown"
    )


@app.get("/download-all")
def download_all_markdown():
    zip_path = create_markdown_zip()

    if zip_path is None:
        return JSONResponse(
            {"error": "No markdown files are available for download."},
            status_code=404
        )

    return FileResponse(
        path=zip_path,
        filename="all_markdown_files.zip",
        media_type="application/zip"
    )