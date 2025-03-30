import logging
import os
import sys


class CustomFormatter(logging.Formatter):
    """Applies level-specific colors to log messages using a provided format string."""

    grey = "\x1b[38;20m"
    blue = "\x1b[34;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    def __init__(self, fmt: str, datefmt: str | None = None):
        self.base_fmt = fmt
        self.base_datefmt = datefmt
        self.formats = {
            logging.DEBUG: self.grey + self.base_fmt + self.reset,
            logging.INFO: self.blue + self.base_fmt + self.reset,
            logging.WARNING: self.yellow + self.base_fmt + self.reset,
            logging.ERROR: self.red + self.base_fmt + self.reset,
            logging.CRITICAL: self.bold_red + self.base_fmt + self.reset,
        }

    def format(self, record):
        log_fmt = self.formats.get(record.levelno, self.base_fmt)
        formatter = logging.Formatter(log_fmt, datefmt=self.base_datefmt)
        return formatter.format(record)


def setup_logger(name: str, log_path: str, include_location: bool = True, console_level: int = logging.INFO, file_level: int = logging.DEBUG, datefmt: str | None = None):
    """Configures and returns a logger with optional location info and colored console output."""
    log_format_base = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_format_location = " (%(filename)s:%(lineno)d)"
    chosen_format = log_format_base + (log_format_location if include_location else "")

    os.makedirs(log_path, exist_ok=True)
    log_filename = os.path.join(log_path, f"{name}.log")

    logger = logging.getLogger(name)
    logger.setLevel(min(console_level, file_level))
    logger.propagate = False

    if logger.hasHandlers():
        logger.handlers.clear()

    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(file_level)
    file_handler.setFormatter(logging.Formatter(chosen_format, datefmt=datefmt))

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(CustomFormatter(chosen_format, datefmt=datefmt))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


if __name__ == "__main__":
    logger = setup_logger("my_app", "logs", console_level=logging.INFO, file_level=logging.DEBUG)

    logger.debug("This is a debug message (should appear in file only).")
    logger.info("This is an info message.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")
    logger.critical("This is a critical message.")
