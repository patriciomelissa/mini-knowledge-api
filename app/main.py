from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from app.api.routes import router
from app.utils import RAGService

rag_service = RAGService()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Initializing RAG service...")
    rag_service.initialize()
    yield
    # Shutdown (se precisares no futuro)
    print("Shutting down...")


app = FastAPI(
    title="Mini Knowledge API",
    lifespan=lifespan
)

app.include_router(router)


@app.get("/")
def root():
    return {"message": "API is running"}


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="debug"
    )