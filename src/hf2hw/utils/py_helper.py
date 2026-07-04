import functools
import io
import logging
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path


def check_parent_field(obj: object, field_name: str):
    if not hasattr(obj, field_name):
        raise AttributeError(
            f"{obj.__class__.__name__} is missing required field `{field_name}`. Please make sure you set `self.{field_name}` before calling ABC initialization."
        )


def with_temp_dir(method):
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        _dir = Path(os.getcwd()).joinpath("_subgraph_plugin_onnx_folder")
        with tempfile.TemporaryDirectory(dir=_dir) as temp_dir:
            return method(self, temp_dir, *args, **kwargs)

    return wrapper


@contextmanager
def suppress_onnx_export_logs():
    """Suppress torch.onnx.export info/debug console output; exceptions propagate normally."""
    _logger_names = ("torch.onnx", "torch.onnx.utils", "torch.onnx.diagnostics")
    _loggers = [logging.getLogger(n) for n in _logger_names]
    _orig_levels = [lg.level for lg in _loggers]
    for lg in _loggers:
        lg.setLevel(logging.WARNING)
    _buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(_buf):
            yield
    finally:
        for lg, lvl in zip(_loggers, _orig_levels):
            lg.setLevel(lvl)
