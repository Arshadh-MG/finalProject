"""
Logger Utility Module for Wildlife Injury Detection
Sets up logging for the application
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logger(name='WildlifeInjuryDetection', log_folder=None, level=logging.INFO):
    """
    Set up logger for the application
    
    Args:
        name: Logger name
        log_folder: Path to folder for log files
        level: Logging level
    
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if log folder provided
    if log_folder:
        log_folder = Path(log_folder)
        log_folder.mkdir(parents=True, exist_ok=True)
        
        log_file = log_folder / f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name='WildlifeInjuryDetection'):
    """
    Get existing logger
    
    Args:
        name: Logger name
    
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class LoggerMixin:
    """Mixin class to add logging capability to any class"""
    
    @property
    def logger(self):
        """Get logger for this class"""
        if not hasattr(self, '_logger'):
            self._logger = logging.getLogger(self.__class__.__name__)
        return self._logger


if __name__ == "__main__":
    # Test the logger
    logger = setup_logger('TestLogger')
    logger.info("Test info message")
    logger.warning("Test warning message")
    logger.error("Test error message")
    print("Logger setup complete")
