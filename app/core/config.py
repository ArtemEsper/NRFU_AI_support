from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional
import os

class Settings(BaseSettings):
    PROJECT_NAME: str = "NRFU AI Formal Criteria Checking MVP"
    API_V1_STR: str = "/api/v1"
    
    ENVIRONMENT: str = "development"  # development, test, production

    POSTGRES_SERVER: str = "db"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    POSTGRES_DB: str = "nrfu_ai"
    DATABASE_URL: Optional[str] = None

    @property
    def sqlalchemy_database_uri(self) -> str:
        if self.DATABASE_URL:
            return self.DATABASE_URL
        
        if self.ENVIRONMENT == "test":
            # Use a unique name to avoid multi-process/session conflicts
            return f"sqlite:///./test_{os.getpid()}.db"
            
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_SERVER}/{self.POSTGRES_DB}"

    REDIS_HOST: str = "redis"
    REDIS_PORT: int = 6379

    LOG_LEVEL: str = "INFO"

    UPLOAD_DIR: str = "uploads"
    SCREENSHOTS_DIR: str = "screenshots"

    # LiteParse configuration
    USE_LITEPARSE_ENRICHMENT: bool = False
    LITEPARSE_OCR_ENABLED: bool = True
    LITEPARSE_OCR_ENGINE: str = "tesseract"  # or "external"
    LITEPARSE_SCREENSHOT_ENABLED: bool = True
    LITEPARSE_MAX_PAGES: Optional[int] = None
    LITEPARSE_USE_CLI: bool = True
    LITEPARSE_CLI_PATH: str = "lit"
    LITEPARSE_DEFAULT_DPI: int = 150
    LITEPARSE_HIGH_QUALITY_DPI: int = 300
    LITEPARSE_OCR_SERVER_URL: Optional[str] = None

    # noinspection PyTypeChecker
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )

settings = Settings()
