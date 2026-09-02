"""
Centralized logging configuration for the benchmark suite.

Provides a professional, colored logger with timestamps and module names
for use across all benchmark modules.
"""

import logging
import sys
from typing import Optional

try:
    import colorlog
    COLORLOG_AVAILABLE = True
except ImportError:
    COLORLOG_AVAILABLE = False


def setup_logger(name: str = "benchmark", level: int = logging.INFO) -> logging.Logger:
    """
    Configure and return a professional logger instance.

    Args:
        name: Logger name (typically "benchmark")
        level: Logging level (logging.INFO or logging.DEBUG)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Always update the level to allow runtime changes (e.g., --verbose flag)
    logger.setLevel(level)
    
    # Avoid duplicate handlers if setup is called multiple times
    if logger.handlers:
        # Update handler level for existing logger
        for handler in logger.handlers:
            handler.setLevel(level)
        return logger
    
    logger.propagate = False
    
    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    
    if COLORLOG_AVAILABLE:
        # Colored formatter using colorlog
        formatter = colorlog.ColoredFormatter(
            fmt="%(log_color)s%(asctime)s | %(levelname)-8s | %(name)s%(reset)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            },
            reset=True,
            style='%'
        )
    else:
        # Fallback to standard formatter if colorlog not available
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger


def get_logger(module_name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance for a specific module.
    
    Child loggers inherit the level from the root 'benchmark' logger,
    so they will respect the --verbose flag set in run_benchmark.py.

    Args:
        module_name: Name of the module (e.g., "runner", "scale_sweep")

    Returns:
        Logger instance that inherits parent's log level
    """
    if module_name:
        child_logger = logging.getLogger(f"benchmark.{module_name}")
        # Ensure child inherits parent level by not setting its own level
        # This allows the root "benchmark" logger's level to propagate
        return child_logger
    return logging.getLogger("benchmark")


# Default logger instance
logger = setup_logger()
