import gzip
import logging
import os
import shutil
import sys
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler

from app.config import settings


def gz_namer(name):
    """Append .gz to the log filename."""
    return name + ".gz"


def gz_rotator(source, dest):
    """Compress the log file using gzip."""
    with open(source, "rb") as f_in:
        with gzip.open(dest, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    os.remove(source)


def setup_logging():
    """
    Setup logging configuration:
    - Log to console (StreamHandler)
    - Log to files with rotation (Size-based or Time-based)
    - Optional GZIP compression for old logs
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

    # 5. File Handler with Rotation
    try:
        if settings.LOG_ROTATION_TYPE == "time":
            file_handler = TimedRotatingFileHandler(
                log_filepath,
                when=settings.LOG_ROTATION_WHEN,
                interval=settings.LOG_ROTATION_INTERVAL,
                backupCount=settings.LOG_BACKUP_COUNT,
                encoding="utf-8",
            )
        else:  # Default to size-based
            file_handler = RotatingFileHandler(
                log_filepath,
                maxBytes=settings.LOG_MAX_BYTES,
                backupCount=settings.LOG_BACKUP_COUNT,
                encoding="utf-8",
            )

        # Enable compression if configured
        if settings.LOG_COMPRESS:
            file_handler.namer = gz_namer
            file_handler.rotator = gz_rotator

        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"Failed to set up file logging: {e}")


    logging.info(
        f"Logging setup complete. Level: {settings.LOG_LEVEL}, File: {log_filepath}"
    )
