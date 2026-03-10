import logging
import sys
from app.core.config import settings


def setup_logging():
    logging.basicConfig(
        level=settings.LOG_LEVEL,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )
    nrfu_logger = logging.getLogger("nrfu_ai")
    return nrfu_logger


logger = setup_logging()
