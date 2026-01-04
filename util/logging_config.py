"""Logging configuration with print capture support."""

import logging
import os
import sys
import warnings
from datetime import datetime
from typing import Optional

_original_stdout = sys.stdout
_original_stderr = sys.stderr


class PrintCaptureStream:
    """Stream wrapper that redirects writes to a logger."""

    def __init__(self, logger: logging.Logger, level: int = logging.INFO, original_stream=None):
        self.logger = logger
        self.level = level
        self.original_stream = original_stream

    def write(self, message: str):
        if message and "\r" in message:
            if self.original_stream:
                self.original_stream.write(message)
                self.original_stream.flush()
            return
        if message.strip():
            self.logger.log(self.level, message.strip())

    def flush(self):
        if self.original_stream:
            self.original_stream.flush()


def setup_logging(
    debug_mode: bool = False, log_dir: Optional[str] = None, log_prefix: str = "run", capture_print: bool = True
) -> logging.Logger:
    """
    Configure logging with console output and optional file logging.
    """
    global _original_stdout, _original_stderr
    _original_stdout = sys.stdout
    _original_stderr = sys.stderr

    warnings.filterwarnings("ignore", category=FutureWarning)

    level = logging.DEBUG if debug_mode else logging.INFO
    logger = logging.getLogger()
    logger.setLevel(level)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler
    console = logging.StreamHandler(_original_stdout)
    console.setLevel(level)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # File handler
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"{log_prefix}_{timestamp}.log")

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info(f"Logging to file: {log_file}")

    # Capture print statements
    if capture_print:
        sys.stdout = PrintCaptureStream(logger, logging.INFO, _original_stdout)
        sys.stderr = PrintCaptureStream(logger, logging.ERROR, _original_stderr)

    return logger