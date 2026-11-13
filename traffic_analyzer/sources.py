"""Adapters supply a local file for the duration of parsing.

A future HTTP adapter can download to a temporary file in materialize(),
yield its path and remove it on exit. Detectors never need network access.
"""
from contextlib import contextmanager
from pathlib import Path
from typing import ContextManager, Iterator, Protocol


class FileSource(Protocol):
    @property
    def name(self) -> str: ...

    def materialize(self) -> ContextManager[Path]: ...


class LocalFileSource:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    @property
    def name(self) -> str:
        return str(self.path)

    @contextmanager
    def materialize(self) -> Iterator[Path]:
        yield self.path


def discover_sources(directory: str | Path) -> list[LocalFileSource]:
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Папка входных данных не найдена: {directory}")
    return [
        LocalFileSource(path)
        for path in sorted(directory.iterdir())
        if path.is_file()
        and path.suffix.lower() in {".xlsx", ".xls"}
        and not path.name.startswith("~$")
    ]
