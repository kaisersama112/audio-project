import asyncio
import os
import shutil
import time

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import FastAPI
from starlette.staticfiles import StaticFiles
from config import settings, DOWNLOAD_DIR
from curd.async_models import init_db_async, init_engine

from routers import transcribe
from fastapi.middleware.cors import CORSMiddleware
from services.audio_service import audio_service, task_queue_manager

# 初始化定时任务调度器
scheduler = AsyncIOScheduler()
app = FastAPI()
app.include_router(transcribe.router, prefix="/api/v1", tags=["transcribe"])
origins = [
    "http://localhost:5173",  # 开发环境
    "http://your-production-domain.com"  # 生产环境
]
# 新增：定时清理函数
async def cleanup_temp_dir():
    """清理TEMP_DIR下超过指定天数的文件夹（排除DOWNLOAD_DIR）"""
    days = settings.CLEANUP_TEMP_DAYS
    now = time.time()
    cutoff = now - days * 86400  # 转换为秒

    if not os.path.exists(TEMP_DIR):
        return

    for entry in os.scandir(TEMP_DIR):
        # 只处理目录且排除下载目录
        if entry.is_dir() and entry.path != DOWNLOAD_DIR:
            try:
                stat = entry.stat()
                if stat.st_ctime < cutoff:  # 比较创建时间
                    shutil.rmtree(entry.path)
                    print(f"已删除旧文件夹: {entry.path}")
            except Exception as e:
                print(f"删除文件夹 {entry.path} 失败: {e}")

async def cleanup_download_dir():
    """清理DOWNLOAD_DIR下超过指定小时的压缩包文件"""
    hours = settings.CLEANUP_DOWNLOAD_HOURS
    now = time.time()
    cutoff = now - hours * 3600  # 转换为秒

    if not os.path.exists(DOWNLOAD_DIR):
        return

    for entry in os.scandir(DOWNLOAD_DIR):
        # 只处理常见压缩格式文件
        if entry.is_file() and entry.name.lower().endswith(('.zip', '.rar', '.7z', '.tar.gz', '.gz')):
            try:
                stat = entry.stat()
                if stat.st_mtime < cutoff:  # 比较修改时间
                    os.remove(entry.path)
                    print(f"已删除旧压缩包: {entry.path}")
            except Exception as e:
                print(f"删除文件 {entry.path} 失败: {e}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


@app.on_event("startup")
async def startup_event():
    engine = init_engine()
    SessionLocal = init_db_async(engine)
    print("数据库初始化完成")
    audio_service.load_model()
    print("模型加载完成")
    asyncio.create_task(task_queue_manager.start())
    print("任务队列管理器已启动")
    # 新增：配置并启动定时任务
    # scheduler.add_job(cleanup_temp_dir, 'cron',  hour=2, minute=0)  # 每天凌晨2点执行
    # scheduler.add_job(cleanup_download_dir, 'interval', hours=1)  # 每小时执行一次
    # scheduler.start()
    # print("定时清理任务已启动")

# 使用 uvicorn 的 shutdown 事件来清理资源
@app.on_event("shutdown")
async def shutdown_event():
    await task_queue_manager.stop()  # 假设 TaskQueueManager 有 stop 方法
    # scheduler.shutdown()  # 新增：关闭调度器
    # print("任务队列管理器已停止")

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
