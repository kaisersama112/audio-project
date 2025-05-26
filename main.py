import os
from fastapi import FastAPI
from starlette.responses import HTMLResponse
from starlette.staticfiles import StaticFiles

from config import settings
from routers import transcribe

from services.audio_service import audio_service
from utils.mysql_db import init_db
app = FastAPI()
app.include_router(transcribe.router, prefix="/api/v1", tags=["transcribe"])


@app.on_event("startup")
async def startup_event():
    init_db()
    audio_service.load_model()


TEMP_DIR = "temp_audio_files"
os.makedirs(TEMP_DIR, exist_ok=True)

@app.get("/status",summary="服务检查")
def status():
    return {
        "status":200,
        "message":"OK"
    }

app.mount("/temp_audio_files", StaticFiles(directory=TEMP_DIR), name="temp_audio_files")
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=settings.HOST,
        port=settings.PORT,
        workers=1)
