from __future__ import annotations

import numpy as np
from scipy.fftpack import dct, idct

from utils import to_grayscale

HEADER_BITS = 32
BLOCK = 8
COEF_POS = (4, 3)
QUANT_STEP = 16


def _dct2(block: np.ndarray) -> np.ndarray:
    return dct(dct(block.T, norm="ortho").T, norm="ortho")


def _idct2(block: np.ndarray) -> np.ndarray:
    return idct(idct(block.T, norm="ortho").T, norm="ortho")


def _bytes_to_bits(data: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8))


def _bits_to_bytes(bits: np.ndarray) -> bytes:
    return np.packbits(bits.astype(np.uint8)).tobytes()


def _int_to_bits(value: int, n_bits: int) -> np.ndarray:
    return np.array([(value >> (n_bits - 1 - i)) & 1 for i in range(n_bits)], dtype=np.uint8)


def _bits_to_int(bits: np.ndarray) -> int:
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def _block_count(image: np.ndarray) -> int:
    h, w = np.asarray(image).shape[:2]
    return (h // BLOCK) * (w // BLOCK)


def capacity_bits(image: np.ndarray) -> int:
    return int(_block_count(image) - HEADER_BITS)


def embed_bytes(data: bytes, image: np.ndarray, quant_step: int = QUANT_STEP) -> np.ndarray:
    gray = to_grayscale(image).astype(np.float64)
    h, w = gray.shape
    n_blocks_y, n_blocks_x = h // BLOCK, w // BLOCK

    bits = np.concatenate([_int_to_bits(len(data), HEADER_BITS), _bytes_to_bits(data)])

    total_blocks = n_blocks_y * n_blocks_x
    if bits.size > total_blocks:
        raise ValueError(
            f"Payload too large: needs {bits.size} blocks but image holds {total_blocks}"
        )

    out = gray.copy()
    r, c = COEF_POS
    idx = 0
    for by in range(n_blocks_y):
        for bx in range(n_blocks_x):
            if idx >= bits.size:
                break
            ys, xs = by * BLOCK, bx * BLOCK
            block = out[ys : ys + BLOCK, xs : xs + BLOCK]
            coeffs = _dct2(block)
            q = int(np.round(coeffs[r, c] / quant_step))
            if (q & 1) != int(bits[idx]):
                q += 1
            coeffs[r, c] = q * quant_step
            out[ys : ys + BLOCK, xs : xs + BLOCK] = _idct2(coeffs)
            idx += 1
        if idx >= bits.size:
            break

    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def _read_bits(image: np.ndarray, n_bits: int, start: int, quant_step: int) -> np.ndarray:
    gray = to_grayscale(image).astype(np.float64)
    h, w = gray.shape
    n_blocks_x = w // BLOCK
    r, c = COEF_POS
    bits = np.empty(n_bits, dtype=np.uint8)
    for i in range(n_bits):
        block_index = start + i
        by, bx = divmod(block_index, n_blocks_x)
        ys, xs = by * BLOCK, bx * BLOCK
        coeffs = _dct2(gray[ys : ys + BLOCK, xs : xs + BLOCK])
        q = int(np.round(coeffs[r, c] / quant_step))
        bits[i] = q & 1
    return bits


def extract_bytes(image: np.ndarray, quant_step: int = QUANT_STEP) -> bytes:
    total_blocks = _block_count(image)
    if total_blocks < HEADER_BITS:
        raise ValueError("Image too small to contain a length header")

    length = _bits_to_int(_read_bits(image, HEADER_BITS, 0, quant_step))

    if HEADER_BITS + length * 8 > total_blocks:
        raise ValueError("Declared payload length exceeds available data")

    return _bits_to_bytes(_read_bits(image, length * 8, HEADER_BITS, quant_step))


def embed(message: str, image: np.ndarray, quant_step: int = QUANT_STEP) -> np.ndarray:
    return embed_bytes(message.encode("utf-8"), image, quant_step)


def extract(image: np.ndarray, quant_step: int = QUANT_STEP) -> str:
    return extract_bytes(image, quant_step).decode("utf-8")
