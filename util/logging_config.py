"""Logging configuration with print capture support."""

import logging
import os
import sys
import warnings
from datetime import datetime
from typing import Optional

_original_stdout = sys.stdout
_original_stderr = sys.stderr

import builtins

def install_print_hook(logger: logging.Logger, level: int = logging.INFO):
    original_print = builtins.print

    def hooked_print(*args, **kwargs):
        sep = kwargs.get("sep", " ")
        end = kwargs.get("end", "\n")
        file = kwargs.get("file", None)

        # Only intercept default prints (or prints explicitly going to stdout)
        if file is None or file is sys.stdout:
            msg = sep.join(str(a) for a in args)

            # If caller uses custom end (rare), include it in the message except final newline
            if end and end != "\n":
                msg += end

            logger.log(level, msg)
            return

        # Anything printed to a different file behaves normally
        original_print(*args, **kwargs)

    builtins.print = hooked_print
    return original_print


def install_excepthook(logger: logging.Logger) -> None:
    def _hook(exc_type, exc, tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc, tb)
            return

        logger.critical("Uncaught exception", exc_info=(exc_type, exc, tb))

        # Force flush to file/console before exit
        for h in logging.getLogger().handlers:
            try:
                h.flush()
            except Exception:
                pass
        logging.shutdown()

    sys.excepthook = _hook



class PrintCaptureStream:
    """Stream wrapper that redirects writes to a logger."""

    def __init__(self, logger: logging.Logger, level: int = logging.INFO, original_stream=None):
        self.logger = logger
        self.level = level
        self.original_stream = original_stream
        self.buffer = ""

    def write(self, message: str):
        if not message:
            return

        # Pass through carriage returns (e.g., tqdm)
        if "\r" in message:
            if self.original_stream:
                self.original_stream.write(message)
                self.original_stream.flush()
            return

        self.buffer += message

        if "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            line = line.rstrip()
            if line:
                self.logger.log(self.level, line)

    def flush(self):
        if self.buffer:
            line = self.buffer.rstrip()
            if line:
                self.logger.log(self.level, line)
            self.buffer = ""

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
        # sys.stderr = PrintCaptureStream(logger, logging.ERROR, _original_stderr)

    install_excepthook(logger)
    install_print_hook(logger, logging.INFO)
    
    return logger
