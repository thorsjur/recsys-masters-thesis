import logging
import os
import sys
import warnings
from datetime import datetime
from typing import Optional

_original_stdout = sys.stdout
_original_stderr = sys.stderr

class PrintLogger:
    """Redirect print statements to logging while preserving tqdm output."""
    def __init__(self, logger, level=logging.INFO, original_stream=None):
        self.logger = logger
        self.level = level
        self.original_stream = original_stream
        
    def write(self, message):
        if message and ('\r' in message or message.startswith('\r')):
            if self.original_stream:
                self.original_stream.write(message)
                self.original_stream.flush()
            return
        
        if message.strip():
            self.logger.log(self.level, message.strip())
    
    def flush(self):
        if self.original_stream:
            self.original_stream.flush()

def setup_logging(debug_mode: bool = False, log_dir: Optional[str] = None, log_prefix: str = "run", capture_print: bool = True):
    """Setup logging with both console and file handlers."""
    global _original_stdout, _original_stderr
    
    _original_stdout = sys.stdout
    _original_stderr = sys.stderr
    
    warnings.filterwarnings('ignore', category=FutureWarning)
    
    level = logging.DEBUG if debug_mode else logging.INFO
    
    logger = logging.getLogger()
    logger.setLevel(level)
    logger.handlers.clear()
    
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_handler = logging.StreamHandler(_original_stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f"{log_prefix}_{timestamp}.log")
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logging.info(f"Logging to file: {log_file}")
    
    if capture_print:
        sys.stdout = PrintLogger(logger, logging.INFO, _original_stdout)
        sys.stderr = PrintLogger(logger, logging.ERROR, _original_stderr)
    
    return logger
