"""
Define the logger in a centralized module (e.g., logger.py) and import it wherever needed.

Note:
- FOLDER_PATH should be relative to the root directory of the project
"""

import logging
import sys
import os
from datetime import datetime

FOLDER_PATH = "eval/VRSBench_vqa-n1000" # TODO: change if needed

# Set up logging
os.makedirs(f"{FOLDER_PATH}/logs", exist_ok=True)
log_filename = f"{FOLDER_PATH}/logs/execution-{datetime.now().strftime('%Y_%m_%d-%H_%M_%S')}.log"
error_log_filename = f"{FOLDER_PATH}/logs/errors-{datetime.now().strftime('%Y_%m_%d-%H_%M_%S')}.log"

# Create logger
logger = logging.getLogger("global_logger")  # Name it to avoid conflicts
logger.setLevel(logging.INFO)

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

# Example Usage
# logger.info("Script started...")
# logger.warning("This is a warning")
# logger.error("This is an error message")
# logger.info("Execution complete!")


# Export logger for use in all files
__all__ = ["logger"]