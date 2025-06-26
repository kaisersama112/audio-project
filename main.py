import os
from fastapi import FastAPI
from starlette.staticfiles import StaticFiles
from config import settings, DOWNLOAD_DIR
from curd.async_models import init_db_async

from routers import transcribe
from fastapi.middleware.cors import CORSMiddleware
from services.audio_service import audio_service

app = FastAPI()
app.include_router(transcribe.router, prefix="/api/v1", tags=["transcribe"])
origins = [
    "http://localhost:5173",  # 开发环境
    "http://your-production-domain.com"  # 生产环境
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


@app.on_event("startup")
async def startup_event():
    init_db_async()
    audio_service.load_model()


TEMP_DIR = "temp_audio_files"
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


@app.get("/status", summary="服务检查")
def status():
    return {
        "status": 200,
        "message": "OK"
    }


app.mount("/temp_audio_files", StaticFiles(directory=TEMP_DIR, check_dir=False), name="temp_audio_files")
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=settings.HOST,
                port=settings.PORT,
                workers=1)
