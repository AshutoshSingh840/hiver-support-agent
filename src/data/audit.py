"""Audit logging and SHA-256 integrity verification utilities."""

import hashlib
from pathlib import Path
from typing import Dict, Any


def compute_file_sha256(file_path: Path, chunk_size: int = 65536) -> str:
    """Computes SHA-256 checksum of a file in chunks."""
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            sha.update(chunk)
    return sha.hexdigest()
