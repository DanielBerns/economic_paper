import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from economic_graph.config import LoggingSettings


def setup_logger(
    name: str = "economic_graph",
    settings: Optional[LoggingSettings] = None,
) -> logging.Logger:
    """Setup and configure a logger with both Console and Rotating File Handlers."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # Already configured

    level_str = settings.level if settings else "INFO"
    log_level = getattr(logging, level_str.upper(), logging.INFO)
    logger.setLevel(log_level)

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s:%(filename)s:%(lineno)d] - %(message)s"
    )

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler with rotation
    if settings and settings.log_file:
        log_path = Path(settings.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=settings.max_bytes,
            backupCount=settings.backup_count,
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
