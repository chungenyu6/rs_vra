"""
Define the logger in a centralized module (e.g., logger.py) and import it wherever needed.

Note:
- FOLDER_PATH should be relative to the root directory of the project
"""

import logging
import sys
import os
from datetime import datetime

# Create a default logger instance
logger = logging.getLogger("global_logger")
logger.setLevel(logging.INFO)

def setup_logger(args):
    """Setup and return a configured logger instance"""
    global logger
    
    # Set up logging # TODO: change path if needed
    log_base_path = f"results/VRSBench_vqa-n1000/{args.model}/{args.version}/{args.qtype}/sam{args.sample}"
    os.makedirs(log_base_path, exist_ok=True)

    # Create log filenames with timestamp
    timestamp = datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
    log_filename = f"{log_base_path}/execution-{timestamp}.log"
    error_log_filename = f"{log_base_path}/errors-{timestamp}.log"

    # Prevent duplicate handlers (important when using imports)
    if not logger.hasHandlers():
        # Create handlers
        file_handler = logging.FileHandler(log_filename)
        error_handler = logging.FileHandler(error_log_filename)
        console_handler = logging.StreamHandler(sys.stdout)

        # Set levels
        file_handler.setLevel(logging.INFO)
        error_handler.setLevel(logging.ERROR)
        console_handler.setLevel(logging.INFO)

        # Define log format
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        error_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(error_handler)
        logger.addHandler(console_handler)

    return logger

# Example Usage
# logger.info("Script started...")
# logger.warning("This is a warning")
# logger.error("This is an error message")
# logger.info("Execution complete!")


# Export logger for use in all files
__all__ = ["logger", "setup_logger"]