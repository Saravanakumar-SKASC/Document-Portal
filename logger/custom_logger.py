import logging
import os
from datetime import datetime

import structlog

_CONFIGURED = False
_LOG_FILE_PATH: str | None = None


def _configure(log_dir: str) -> str:
    """Configure stdlib logging + structlog exactly once per process."""
    global _CONFIGURED, _LOG_FILE_PATH
    if _CONFIGURED:
        return _LOG_FILE_PATH

    logs_dir = os.path.join(os.getcwd(), log_dir)
    os.makedirs(logs_dir, exist_ok=True)
    _LOG_FILE_PATH = os.path.join(logs_dir, f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log")

    file_handler = logging.FileHandler(_LOG_FILE_PATH, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(message)s"))

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(console_handler)
    root.addHandler(file_handler)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,  # adds request_id / user_role bound per request
            structlog.processors.TimeStamper(fmt="iso", utc=True, key="timestamp"),
            structlog.processors.add_log_level,
            structlog.processors.EventRenamer(to="event"),
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    _CONFIGURED = True
    return _LOG_FILE_PATH


class CustomLogger:
    """
    Structured JSON logging (console + timestamped file).

    Configuration happens once per process, so creating many CustomLogger
    instances no longer produces one log file per instance or duplicate handlers.
    """

    def __init__(self, log_dir: str | None = None):
        self.log_file_path = _configure(log_dir or os.getenv("LOG_DIR", "logs"))

    def get_logger(self, name: str = __file__):
        return structlog.get_logger(os.path.basename(name))


if __name__ == "__main__":
    logger = CustomLogger().get_logger(__file__)
    logger.info("User uploaded a file", user_id=123, filename="report.pdf")
    logger.error("Failed to process PDF", error="File not found", user_id=123)
