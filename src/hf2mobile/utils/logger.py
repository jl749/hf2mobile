"""Centralized logger for hf2mobile.

Writes INFO+ messages to stderr with a `[hf2mobile]` prefix by default.
Level names are colorized when stderr is a TTY
(DEBUG=gray, INFO=cyan, WARNING=yellow, ERROR=red, CRITICAL=bold red);
plain text otherwise so logs piped to files / CI stay clean.
Set `DEBUG=1` in the environment to start at DEBUG level (e.g. `DEBUG=1 python src/main.py`);
otherwise it defaults to INFO.
Verbosity can also be changed at runtime via the standard `logging` module:

    >> import logging
    >> logging.getLogger("hf2mobile").setLevel(logging.DEBUG)   # more chatty
    >> logging.getLogger("hf2mobile").setLevel(logging.WARNING) # quieter

Internal modules should import and use the shared `logger` instance::

    >> from hf2mobile.utils.logger import logger
    >> logger.info("Registering plugin ops...")
"""

import logging
import os
import sys

_LOGGER_NAME = "hf2mobile"


def _env_debug_enabled() -> bool:
    return os.environ.get("DEBUG", "").strip().lower() in ("1", "true", "yes", "on")


_RESET = "\033[0m"
_LEVEL_COLOR = {
    logging.DEBUG: "\033[90m",  # bright black / gray
    logging.INFO: "\033[36m",  # cyan
    logging.WARNING: "\033[33m",  # yellow
    logging.ERROR: "\033[31m",  # red
    logging.CRITICAL: "\033[1;31m",  # bold red
}


class _ColorFormatter(logging.Formatter):
    """Formatter that colorizes the levelname when `use_color` is True.

    The width-padded `%(levelname)-8s` keeps the message column aligned even
    when colors differ in printed length.
    """

    def __init__(self, use_color: bool):
        super().__init__("[hf2mobile] %(levelname)-8s | %(message)s")
        self._use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        if not self._use_color:
            return super().format(record)
        original_levelname = record.levelname
        color = _LEVEL_COLOR.get(record.levelno, "")
        # pad first, then wrap in color codes so width stays uniform.
        record.levelname = f"{color}{original_levelname:<8}{_RESET}"
        try:
            # replace the format string for this call to avoid double-padding.
            return logging.Formatter("[hf2mobile] %(levelname)s | %(message)s").format(record)
        finally:
            record.levelname = original_levelname


def _get_logger() -> logging.Logger:
    _NOISY_LOGGERS = ("torch.onnx._internal.exporter._registration",)
    log = logging.getLogger(_LOGGER_NAME)
    if not log.handlers:
        handler = logging.StreamHandler(sys.stderr)
        use_color = hasattr(sys.stderr, "isatty") and sys.stderr.isatty()
        handler.setFormatter(_ColorFormatter(use_color=use_color))
        log.addHandler(handler)
        if _env_debug_enabled():
            log.setLevel(logging.DEBUG)
            for _name in _NOISY_LOGGERS:
                logging.getLogger(_name).setLevel(logging.WARNING)
        else:
            log.setLevel(logging.INFO)
            for _name in _NOISY_LOGGERS:
                logging.getLogger(_name).setLevel(logging.ERROR)
        log.propagate = False  # do not double-emit via the root logger
    return log


logger = _get_logger()


__all__ = ["logger"]
