import logging
import os
import sys

# isort: off
from config import Config


class CustomFormatter(logging.Formatter):
    """Applies level-specific colors to log messages using a provided format string."""

    grey = "\x1b[38;20m"
    blue = "\x1b[34;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    def __init__(self, fmt: str):
        self.fmt = fmt
        self.formats = {
            logging.DEBUG: self.grey + self.fmt + self.reset,
            logging.INFO: self.blue + self.fmt + self.reset,
            logging.WARNING: self.yellow + self.fmt + self.reset,
            logging.ERROR: self.red + self.fmt + self.reset,
            logging.CRITICAL: self.bold_red + self.fmt + self.reset,
        }

    def format(self, record):
        log_fmt = self.formats.get(record.levelno, self.fmt)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def create_logger(
    config: Config,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
) -> logging.Logger:
    """Creates a logger with both console and file handler."""
    id = config.experiment_id
    format = f"%(asctime)s -  %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    os.makedirs(config.output.log_path, exist_ok=True)
    filename = os.path.join(config.output.log_path, f"{id}.log")

    logger = logging.getLogger(id)
    logger.setLevel(min(console_level, file_level))
    logger.propagate = False

    if logger.hasHandlers():
        logger.handlers.clear()

    file_handler = logging.FileHandler(filename, mode="w")
    file_handler.setLevel(file_level)
    file_handler.setFormatter(logging.Formatter(format))

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(CustomFormatter(format))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
