from __future__ import annotations

import numpy as np

HEADER_BITS = 32


def _bytes_to_bits(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    return np.unpackbits(arr)


def _bits_to_bytes(bits: np.ndarray) -> bytes:
    return np.packbits(bits.astype(np.uint8)).tobytes()


def _int_to_bits(value: int, n_bits: int) -> np.ndarray:
    return np.array([(value >> (n_bits - 1 - i)) & 1 for i in range(n_bits)], dtype=np.uint8)


def _bits_to_int(bits: np.ndarray) -> int:
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def capacity_bits(image: np.ndarray) -> int:
    return int(np.asarray(image).size - HEADER_BITS)


def embed(message: str, image: np.ndarray) -> np.ndarray:
    img = np.asarray(image, dtype=np.uint8)
    payload = message.encode("utf-8")
    message_bits = _bytes_to_bits(payload)
    header_bits = _int_to_bits(len(payload), HEADER_BITS)
    bits = np.concatenate([header_bits, message_bits])

    if bits.size > img.size:
        raise ValueError(
            f"Message too large: needs {bits.size} bits but image holds {img.size}"
        )

    flat = img.reshape(-1).copy()
    flat[: bits.size] = (flat[: bits.size] & 0xFE) | bits
    return flat.reshape(img.shape)


def extract(image: np.ndarray) -> str:
    img = np.asarray(image, dtype=np.uint8)
    flat = img.reshape(-1)
    lsb = flat & 1

    if lsb.size < HEADER_BITS:
        raise ValueError("Image too small to contain a length header")

    length = _bits_to_int(lsb[:HEADER_BITS])
    message_bits_needed = length * 8
    end = HEADER_BITS + message_bits_needed

    if end > lsb.size:
        raise ValueError("Declared message length exceeds available data")

    message_bits = lsb[HEADER_BITS:end]
    return _bits_to_bytes(message_bits).decode("utf-8")
