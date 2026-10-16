"""Character <-> bit-index mapping table. No interleaver: char i occupies bits [8i, 8i+8)."""

CHAR_BITS = 8


def chars_per_frame(k: int) -> int:
    """Number of full characters that fit in k info bits (remainder bits are fixed padding)."""
    return k // CHAR_BITS


def pad_bits(k: int) -> int:
    return k % CHAR_BITS


def char_bit_range(i: int) -> tuple[int, int]:
    """Bit index range [lo, hi) for character i."""
    return i * CHAR_BITS, i * CHAR_BITS + CHAR_BITS


def bit_to_char_index(bit_idx: int) -> int:
    return bit_idx // CHAR_BITS


def char_indices_for_bits(bit_indices) -> set:
    """Given a collection of bit indices, return the set of char indices they touch."""
    return {bit_to_char_index(b) for b in bit_indices}


def padding_bit_range(k: int) -> tuple[int, int]:
    """[lo, hi) of the k-length info word occupied by fixed (non-character) padding bits."""
    return chars_per_frame(k) * CHAR_BITS, k


def apply_known_padding(llr_ch, k: int, confidence: float = 20.0):
    """Phase-2 A-3 'free' improvement: padding bits are always 0 (see textcodec.frame_text),
    a constant known to the receiver, so they need no channel estimate at all -- pin their
    LLR to a strong prior instead of trusting the noisy channel measurement (shortened-code
    treatment)."""
    import numpy as np
    llr_ch = llr_ch.copy()
    lo, hi = padding_bit_range(k)
    llr_ch[lo:hi] = confidence  # bit=0 -> positive LLR under our 0->+1 convention
    return llr_ch
