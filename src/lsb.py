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


def embed_bytes(data: bytes, image: np.ndarray) -> np.ndarray:
    img = np.asarray(image, dtype=np.uint8)
    header_bits = _int_to_bits(len(data), HEADER_BITS)
    bits = np.concatenate([header_bits, _bytes_to_bits(data)])

    if bits.size > img.size:
        raise ValueError(
            f"Payload too large: needs {bits.size} bits but image holds {img.size}"
        )

    flat = img.reshape(-1).copy()
    flat[: bits.size] = (flat[: bits.size] & 0xFE) | bits
    return flat.reshape(img.shape)


def extract_bytes(image: np.ndarray) -> bytes:
    img = np.asarray(image, dtype=np.uint8)
    lsb = img.reshape(-1) & 1

    if lsb.size < HEADER_BITS:
        raise ValueError("Image too small to contain a length header")

    length = _bits_to_int(lsb[:HEADER_BITS])
    end = HEADER_BITS + length * 8

    if end > lsb.size:
        raise ValueError("Declared payload length exceeds available data")

    return _bits_to_bytes(lsb[HEADER_BITS:end])


def embed(message: str, image: np.ndarray) -> np.ndarray:
    return embed_bytes(message.encode("utf-8"), image)


def extract(image: np.ndarray) -> str:
    return extract_bytes(image).decode("utf-8")
