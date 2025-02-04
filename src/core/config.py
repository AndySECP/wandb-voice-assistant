from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Optional

class Settings(BaseSettings):
    """Application settings"""
    # Required settings
    OPENAI_API_KEY: str
    WANDB_API_KEY: str
    
    # Optional settings with defaults
    PROJECT_NAME: str = "c-metrics/hallucination"
    MODEL_NAME: str = "gpt-4"
    MAX_RETRIES: int = 3
    INITIAL_RETRY_DELAY: float = 1.0
    DAILY_ROOM_URL: Optional[str] = None

    # Audio settings
    DEFAULT_SAMPLE_RATE: int = 24000
    ENABLE_STREAMING: bool = True

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "allow"  # Allow extra fields in the settings

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()