from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from src.api.routes import router
from src.services.data_manager import HallucinationDataManager
from src.core.agent.memory import MemoryManager
import logging

logger = logging.getLogger(__name__)

app = FastAPI(title="Weave Query API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the router with prefix
app.include_router(router, prefix="/api/v1")


@app.get("/")
async def root():
    return {"message": "Welcome to Weave Query API"}


@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    if isinstance(exc, HTTPException):
        raise exc
    return {"detail": str(exc)}


@app.on_event("startup")
async def startup_event():
    """Initialize data and memory at startup"""
    try:
        # Initialize data manager
        data_manager = HallucinationDataManager()
        await data_manager.initialize()

        # Initialize memory manager (no async needed here)
        memory_manager = MemoryManager(
            cache_size=20,  # Keep 20 most recent messages in memory
            persist_after=5,  # Write to database every 5 messages
        )

        # Store in app state for access across requests
        app.state.data_manager = data_manager
        app.state.memory_manager = memory_manager

        logger.info("Successfully initialized data manager and memory at startup")
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
