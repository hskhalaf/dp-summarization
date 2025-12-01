import logging
from enum import Enum


class Color(Enum):
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"


class TextColor:
    def color(self, string: str, color: str) -> str:
        return f"{color}{string}\033[0m"


class ColorFormatter(logging.Formatter):
    LEVEL_COLOR = {
        logging.DEBUG: Color.CYAN.value,
        logging.INFO: Color.GREEN.value,
        logging.WARNING: Color.YELLOW.value,
        logging.ERROR: Color.RED.value,
        logging.CRITICAL: Color.MAGENTA.value,
    }

    def format(self, record: logging.LogRecord) -> str:
        original_levelname = record.levelname
        color = self.LEVEL_COLOR.get(record.levelno, Color.WHITE.value)
        record.levelname = f"{color}{original_levelname}\033[0m"
        try:
            return super().format(record)
        finally:
            record.levelname = original_levelname


def _configure_root_logger(level: int = logging.INFO) -> None:
    handler = logging.StreamHandler()
    formatter = ColorFormatter("%(asctime)s | [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)


def set_level(level: int) -> None:
    """Change the logging level after initialization."""
    logging.getLogger().setLevel(level)


_configure_root_logger()
