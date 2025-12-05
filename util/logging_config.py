import logging
import os
import sys
from datetime import datetime

class PrintLogger:
    """Redirect print statements to logging."""
    def __init__(self, logger, level=logging.INFO):
        self.logger = logger
        self.level = level
        
    def write(self, message):
        if message.strip():
            self.logger.log(self.level, message.strip())
    
    def flush(self):
        pass

def setup_logging(debug_mode: bool = False, log_dir: str = None, log_prefix: str = "run", capture_print: bool = True):
    """Setup logging with both console and file handlers."""
    level = logging.DEBUG if debug_mode else logging.INFO
    
    logger = logging.getLogger()
    logger.setLevel(level)
    logger.handlers.clear()
    
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_handler = logging.StreamHandler()
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
        sys.stdout = PrintLogger(logger, logging.INFO)
        sys.stderr = PrintLogger(logger, logging.ERROR)
    
    return logger
