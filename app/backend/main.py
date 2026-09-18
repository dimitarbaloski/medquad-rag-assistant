import os
from pathlib import Path

import uvicorn

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.routers.chat import router as chat_router


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DIST_DIR = PROJECT_ROOT / "frontend" / "dist"


app = FastAPI(
    title="Demo Chatbot API"
)


# -----------------------------
# API routes
# -----------------------------

app.include_router(chat_router)


@app.get("/api/health")
def health():
    return {
        "status": "healthy"
    }


# -----------------------------
# React static files
# -----------------------------

assets_directory = DIST_DIR / "assets"

if assets_directory.exists():

    app.mount(
        "/assets",
        StaticFiles(
            directory=assets_directory
        ),
        name="assets",
    )


# IMPORTANT:
# Keep this AFTER all /api routes
@app.get(
    "/{full_path:path}",
    include_in_schema=False,
)
def serve_react(full_path: str):

    requested_file = DIST_DIR / full_path

    # Example:
    # /vite.svg
    # /some-file.png
    if (
        full_path
        and requested_file.is_file()
    ):
        return FileResponse(
            requested_file
        )


    # Otherwise serve React
    index_file = DIST_DIR / "index.html"

    if not index_file.exists():

        raise HTTPException(
            status_code=500,
            detail=(
                "React production build was not found."
            ),
        )


    return FileResponse(
        index_file
    )


# -----------------------------
# Start server
# -----------------------------

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "DATABRICKS_APP_PORT",
            os.environ.get(
                "PORT",
                "8000",
            ),
        )
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
    )