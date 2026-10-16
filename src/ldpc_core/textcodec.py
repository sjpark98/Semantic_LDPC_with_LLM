"""ASCII text <-> bit vector, framed to a fixed LDPC info-word length k."""
import numpy as np
from bitmap import chars_per_frame, pad_bits, CHAR_BITS


def text_to_bits(text: str) -> np.ndarray:
    byte_vals = np.frombuffer(text.encode("latin-1", errors="replace"), dtype=np.uint8)
    bits = np.unpackbits(byte_vals)
    return bits.astype(np.int8)


def bits_to_text(bits: np.ndarray) -> str:
    byte_vals = np.packbits(bits.astype(np.uint8))
    return byte_vals.tobytes().decode("latin-1", errors="replace")


def frame_text(text: str, k: int) -> np.ndarray:
    """Pad/truncate text to chars_per_frame(k) chars, return length-k bit vector (extra bits = 0)."""
    n_chars = chars_per_frame(k)
    text = text[:n_chars].ljust(n_chars)  # space-pad
    bits = text_to_bits(text)
    u = np.zeros(k, dtype=np.int8)
    u[: len(bits)] = bits
    return u


def unframe_text(u: np.ndarray, k: int) -> str:
    n_chars = chars_per_frame(k)
    bits = u[: n_chars * CHAR_BITS]
    return bits_to_text(bits)
