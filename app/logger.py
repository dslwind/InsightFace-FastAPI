import logging
import os
import sys
from logging.handlers import RotatingFileHandler

from app.config import settings


def setup_logging():
    """
    Setup logging configuration:
    - Log to console (StreamHandler)
    - Log to rotating file (RotatingFileHandler) using size-based rotation
    """

    # 1. Create Log Directory
    if not os.path.exists(settings.LOG_DIR):
        try:
            os.makedirs(settings.LOG_DIR)
        except OSError as e:
            print(f"Failed to create log directory {settings.LOG_DIR}: {e}")
            return

    log_filepath = os.path.join(settings.LOG_DIR, settings.LOG_FILENAME)

    # 2. Get Root Logger
    logger = logging.getLogger()
    logger.setLevel(settings.LOG_LEVEL.upper())

    # Clean up existing handlers to avoid duplicates on re-init
    if logger.hasHandlers():
        logger.handlers.clear()

    # 3. Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 4. Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 5. Rotating File Handler
    try:
        file_handler = RotatingFileHandler(
            log_filepath,
            maxBytes=settings.LOG_MAX_BYTES,
            backupCount=settings.LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"Failed to set up file logging: {e}")

    logging.info(
        f"Logging setup complete. Level: {settings.LOG_LEVEL}, File: {log_filepath}"
    )
