"""Shared 'module:function' extractor loading, used by the CLI and the MCP server."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Callable


def load_callable(spec: str) -> Callable:
    """
    Load a function from a 'module:function' specifier.

    Adds the current working directory to sys.path first, so a locally-written
    extractor module (not installed as a package) can be imported.

    Raises ValueError with a message describing what went wrong.
    """
    if ":" not in spec:
        raise ValueError(f"Extractor must be specified as 'module:function', got: {spec!r}")
    module_path, fn_name = spec.rsplit(":", 1)

    cwd = str(Path.cwd())
    if cwd not in sys.path:
        sys.path.insert(0, cwd)

    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as exc:
        raise ValueError(f"Cannot import module '{module_path}': {exc}") from exc

    if not hasattr(module, fn_name):
        raise ValueError(f"Function '{fn_name}' not found in module '{module_path}'")
    return getattr(module, fn_name)
