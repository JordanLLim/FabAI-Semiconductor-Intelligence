from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from src.repository import WaferRepository

ROOT = Path(__file__).resolve().parents[1]
repo = WaferRepository(str(ROOT / "data/raw/LSWMD.pkl"))

app = FastAPI(
    title="FAB.AI",
    version="0.1.0",
    description="Semiconductor wafer intelligence command center built around WM-811K.",
)
app.mount("/static", StaticFiles(directory=ROOT / "app/static"), name="static")


@app.get("/")
def home():
    return FileResponse(ROOT / "app/static/index.html")


@app.get("/api/health")
def health():
    return {"status": "ok", "dataset_ready": repo.ready}


@app.get("/api/summary")
def summary():
    return repo.summary()


@app.get("/api/incidents/{label}")
def incident(label: str):
    result = repo.incident(label)
    if result is None:
        raise HTTPException(503, "WM-811K not installed. Place LSWMD.pkl in data/raw/.")
    return result


@app.get("/api/incidents")
def incidents():
    return repo.recent()
