from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI

from app.api.routes import router
from app.utils import RAGService

rag_service = RAGService()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan event handler.

    This function manages startup and shutdown lifecycle events.

    During startup:
        - Initializes the RAG service.

    During shutdown:
        - Placeholder for future cleanup logic.

    Args:
        app (FastAPI): FastAPI application instance.

    Yields:
        AsyncGenerator[None, None]: Lifecycle control generator.
    """
    # Startup
    print("Initializing RAG service...")
    rag_service.initialize()
    yield
    # Shutdown (se precisares no futuro)
    print("Shutting down...")


app = FastAPI(title="Mini Knowledge API", lifespan=lifespan)

app.include_router(router)


@app.get("/")
def root() -> dict:
    """
    Root endpoint to verify if the API is running.

    Returns:
        dict: Simple health check message.
    """
    return {"message": "API is running"}


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app", host="0.0.0.0", port=8000, reload=True, log_level="debug"
    )
