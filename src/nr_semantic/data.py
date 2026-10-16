"""Source-data audit, deterministic split, ASCII serialization, and framing."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


SMART_ASCII = str.maketrans({
    "‘": "'", "’": "'", "‚": "'", "“": '"', "”": '"', "„": '"',
    "–": "-", "—": "-", "−": "-", "…": "...",
})


def normalize_ascii(text: str) -> str:
    """Apply the fixed, lossy normalization declared in the experiment guide."""
    text = unicodedata.normalize("NFKC", text).translate(SMART_ASCII)
    text = "".join(ch if 0x20 <= ord(ch) <= 0x7E else " " for ch in text)
    return re.sub(r" +", " ", text).strip()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class SourceRecord:
    source_id: int
    text: str
    normalized_hash: str


def read_line_delimited_text(path: str | Path) -> list[SourceRecord]:
    """Read the known headerless ``sentences.csv`` format; never CSV-parse it."""
    records: list[SourceRecord] = []
    with Path(path).open(encoding="utf-8", errors="replace") as handle:
        for source_id, line in enumerate(handle):
            text = normalize_ascii(line.rstrip("\r\n"))
            if not text:
                continue
            records.append(SourceRecord(source_id, text, hashlib.sha256(text.encode("ascii")).hexdigest()))
    return records


def audit_line_delimited_text(path: str | Path) -> dict:
    path = Path(path)
    raw_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    normalized = [normalize_ascii(value) for value in raw_lines]
    nonempty = [value for value in normalized if value]
    counts = Counter(nonempty)
    return {
        "path": str(path.resolve()), "sha256": file_sha256(path), "input_format": "line_delimited_text",
        "raw_records": len(raw_lines), "raw_empty": sum(not x for x in raw_lines),
        "normalized_empty": len(normalized) - len(nonempty), "normalized_unique": len(counts),
        "normalized_duplicate_records": sum(n - 1 for n in counts.values() if n > 1),
        "normalization_changed_records": sum(a != b for a, b in zip(raw_lines, normalized)),
        "max_normalized_characters": max(map(len, nonempty), default=0),
        "source_id_available": False,
    }


def assign_split(normalized_hash: str, split_seed: int, ratios: tuple[float, float, float]) -> str:
    """Hash-group split: duplicate normalized text can never cross a split."""
    if len(ratios) != 3 or not np.isclose(sum(ratios), 1.0):
        raise ValueError("split ratios must contain three values summing to one")
    value = int.from_bytes(hashlib.sha256(f"{split_seed}:{normalized_hash}".encode()).digest()[:8], "big") / 2**64
    if value < ratios[0]:
        return "train"
    if value < ratios[0] + ratios[1]:
        return "validation"
    return "test"


@dataclass(frozen=True)
class Frame:
    frame_id: int
    payload: bytes
    valid_bytes: int
    source_ids: tuple[int, ...]

    @property
    def info_bits(self) -> np.ndarray:
        return np.unpackbits(np.frombuffer(self.payload, dtype=np.uint8), bitorder="big")


def frame_records(records: Iterable[SourceRecord], payload_bytes: int) -> list[Frame]:
    """Stream records with a single ASCII-space delimiter into fixed-size frames.

    Records are never length-filtered: long records naturally span frames, while only
    the final stream frame is padded. ``valid_bytes`` is authoritative framing metadata.
    """
    if payload_bytes <= 0:
        raise ValueError("payload_bytes must be positive")
    frames: list[Frame] = []
    buffer = bytearray()
    owners: list[int] = []
    frame_id = 0
    have_previous = False

    def flush_full() -> None:
        nonlocal buffer, owners, frame_id
        while len(buffer) >= payload_bytes:
            frame_sources = tuple(sorted(set(owners[:payload_bytes])))
            frames.append(Frame(frame_id, bytes(buffer[:payload_bytes]), payload_bytes, frame_sources))
            frame_id += 1
            del buffer[:payload_bytes]
            del owners[:payload_bytes]

    for record in records:
        if have_previous:
            buffer.extend(b" ")
            owners.append(record.source_id)
        buffer.extend(record.text.encode("ascii", errors="strict"))
        owners.extend([record.source_id] * len(record.text))
        have_previous = True
        flush_full()
    if buffer:
        valid = len(buffer)
        frames.append(Frame(frame_id, bytes(buffer) + b"\0" * (payload_bytes - valid), valid, tuple(sorted(set(owners)))) )
    return frames


def sentence_frames(records: Iterable[SourceRecord], payload_bytes: int) -> list[Frame]:
    """One complete normalized sentence per zero-padded frame; drop over-length rows."""
    frames: list[Frame] = []
    for record in records:
        raw = record.text.encode("ascii", errors="strict")
        if len(raw) <= payload_bytes:
            frames.append(Frame(len(frames), raw + b"\0" * (payload_bytes - len(raw)), len(raw), (record.source_id,)))
    return frames


def render_received_bytes(payload: bytes, valid_bytes: int) -> tuple[str, np.ndarray]:
    """Render invalid bytes as '?' without changing byte positions."""
    visible = payload[:valid_bytes]
    invalid = np.fromiter((not (0x20 <= value <= 0x7E) for value in visible), dtype=bool, count=len(visible))
    return "".join(chr(value) if 0x20 <= value <= 0x7E else "?" for value in visible), invalid
