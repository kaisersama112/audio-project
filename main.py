from contextlib import asynccontextmanager

from fastapi import FastAPI
from config import settings
from routers import transcribe

from services.audio_service import audio_service
from utils.mysql_db import init_db

app = FastAPI()
app.include_router(transcribe.router, prefix="/api/v1", tags=["transcribe"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield
    # 这里可以添加关闭资源的逻辑


@app.on_event("startup")
async def startup_event():
    init_db()
    audio_service.load_model()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=settings.HOST,
        port=settings.PORT
    )
