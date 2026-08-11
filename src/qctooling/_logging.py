import logging, sys, os
from typing import Optional

class ColorFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: "\033[36m",      # cyan
        logging.INFO: "\033[32m",       # green
        logging.WARNING: "\033[33m",    # yellow
        logging.ERROR: "\033[31m",      # red
        logging.CRITICAL: "\033[1;31m", # bold red
    }
    RESET = "\033[0m"

    def __init__(self, *args, use_color: bool | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_color = sys.stdout.isatty() if use_color is None else use_color

    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        if not self.use_color:
            return message
        color = self.COLORS.get(record.levelno, self.RESET)
        return f"{color}{message}{self.RESET}"


def configure_logging(level: Optional[int] = None) -> None:
    if level is None:
        level = getattr(logging, os.environ.get("QCTOOLING_LOG_LEVEL", "INFO").upper(), logging.INFO)

    handler = logging.StreamHandler()
    handler.setFormatter(ColorFormatter(
        fmt="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    ))
    logging.basicConfig(level=level, handlers=[handler])