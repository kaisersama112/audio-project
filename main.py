from fastapi import FastAPI
from config import settings
from routers import transcribe

from services.audio_service import audio_service

app = FastAPI()
app.include_router(transcribe.router, prefix="/api/v1", tags=["transcribe"])


@app.on_event("startup")
async def startup_event():
    audio_service.load_model()





if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=settings.HOST,
        port=settings.PORT
    )
