"""File system utilities."""

from __future__ import annotations

from pathlib import Path

from ocr.utils.image import SUPPORTED_EXTENSIONS


def list_images(directory: Path) -> list[Path]:
    """Return a sorted list of image files in *directory*.

    Searches non-recursively. Only files with supported extensions
    (case-insensitive) are returned.

    Args:
        directory: Directory to search.

    Returns:
        Sorted list of Path objects.

    Raises:
        NotADirectoryError: If *directory* does not exist or is not a directory.
    """
    if not directory.exists():
        raise NotADirectoryError(f"Images directory does not exist: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {directory}")

    images = [
        p
        for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(images)


def ensure_dir(path: Path) -> Path:
    """Create *path* and all parents if they do not exist.

    Args:
        path: Directory path to create.

    Returns:
        The same *path* (for chaining).
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_stem(path: Path) -> str:
    """Return the stem of *path* (filename without final extension).

    Example: ``safe_stem(Path("docs/invoice.scan.tiff"))`` → ``"invoice.scan"``
    """
    return path.stem
