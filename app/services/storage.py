import os
import shutil
from abc import ABC, abstractmethod
from typing import BinaryIO
from pathlib import Path
from app.core.config import settings
from app.core.logger import logger

class StorageService(ABC):
    @abstractmethod
    async def save_file(self, file: BinaryIO, filename: str) -> str:
        """Save file and return the path or identifier."""
        pass

    @abstractmethod
    async def get_file(self, path: str) -> BinaryIO:
        """Retrieve file content."""
        pass

    @abstractmethod
    async def delete_file(self, path: str) -> bool:
        """Delete file."""
        pass

class LocalStorageService(StorageService):
    def __init__(self, upload_dir: str = settings.UPLOAD_DIR):
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Local storage initialized at {self.upload_dir.absolute()}")

    async def save_file(self, file: BinaryIO, filename: str) -> str:
        # For simplicity, we keep original filename but in a real app 
        # we should probably use a UUID to avoid collisions
        file_path = self.upload_dir / filename
        
        # Ensure directory exists (in case it was deleted)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file, buffer)
        
        return str(file_path)

    async def get_file(self, path: str) -> BinaryIO:
        return open(path, "rb")

    async def delete_file(self, path: str) -> bool:
        try:
            os.remove(path)
            return True
        except FileNotFoundError:
            return False

storage_service = LocalStorageService()
