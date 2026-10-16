"""BPSK + AWGN channel and LLR computation."""
import numpy as np


def bpsk_modulate(bits: np.ndarray) -> np.ndarray:
    """0 -> +1, 1 -> -1."""
    return 1.0 - 2.0 * bits.astype(np.float64)


def awgn(x: np.ndarray, ebn0_db: float, rate: float, rng: np.random.Generator) -> tuple[np.ndarray, float]:
    """Add AWGN calibrated by Eb/N0 (dB) and code rate. Returns (y, sigma2)."""
    ebn0 = 10 ** (ebn0_db / 10.0)
    # Es/N0 = rate * Eb/N0 for unit-energy BPSK symbols (Es=1)
    esn0 = rate * ebn0
    sigma2 = 1.0 / (2.0 * esn0)
    noise = rng.normal(0.0, np.sqrt(sigma2), size=x.shape)
    return x + noise, sigma2


def llr_from_y(y: np.ndarray, sigma2: float) -> np.ndarray:
    """LLR = log P(b=0|y)/P(b=1|y) = 2y/sigma^2 under bit->symbol map 0->+1,1->-1."""
    return 2.0 * y / sigma2
