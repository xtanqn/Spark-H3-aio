"""Select existing package tools without installing pip into the target Python."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


def _available(command: list[str]) -> bool:
    try:
        subprocess.run(
            [*command, "--version"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return True


def _python_path(python: str) -> str:
    # Preserve venv symlinks: resolving them would select the base interpreter.
    return os.path.abspath(shutil.which(python) or python)


def _select(python: str) -> tuple[list[str], bool]:
    uv = shutil.which("uv")
    if uv and _available([os.path.abspath(uv)]):
        return [os.path.abspath(uv)], True
    if _available([python, "-m", "uv"]):
        return [python, "-m", "uv"], True
    if _available([python, "-m", "pip"]):
        return [python, "-m", "pip"], False
    raise RuntimeError(
        f"no usable uv or pip found for {python!r}. Make uv available on PATH "
        "or in this Python environment, or use an existing pip installation. "
        "Spark-H3 does not install package tools automatically."
    )


def install_command(python: str) -> list[str]:
    """Return an install command that targets exactly the requested Python."""
    python = _python_path(python)
    command, use_uv = _select(python)
    if use_uv:
        return [*command, "pip", "install", "--python", python]
    return [*command, "install"]


def wheel_command(python: str, output_dir: Path) -> list[str]:
    """Return a wheel build command for the requested Python and output path."""
    python = _python_path(python)
    command, use_uv = _select(python)
    output = str(output_dir.absolute())
    if use_uv:
        return [
            *command,
            "build",
            "--wheel",
            "--python",
            python,
            "--out-dir",
            output,
        ]
    return [*command, "wheel", "--no-deps", "--wheel-dir", output]
