import asyncio
import os
from fastapi import FastAPI
from starlette.staticfiles import StaticFiles
from config import settings, DOWNLOAD_DIR
from curd.async_models import init_db_async

from routers import transcribe
from fastapi.middleware.cors import CORSMiddleware
from services.audio_service import audio_service, task_queue_manager

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
    print("数据库初始化完成")
    audio_service.load_model()
    print("模型加载完成")
    asyncio.create_task(task_queue_manager.start())
    print("任务队列管理器已启动")

# 使用 uvicorn 的 shutdown 事件来清理资源
@app.on_event("shutdown")
async def shutdown_event():
    await task_queue_manager.stop()  # 假设 TaskQueueManager 有 stop 方法
    print("任务队列管理器已停止")

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
