from __future__ import annotations

import numpy as np

# SplitMix64 constants (Vigna, 2015) for stateless 64-bit mixing.
SPLITMIX64_INCREMENT = np.uint64(0x9E3779B97F4A7C15)
SPLITMIX64_MUL1 = np.uint64(0xBF58476D1CE4E5B9)
SPLITMIX64_MUL2 = np.uint64(0x94D049BB133111EB)
U64_MASK = np.uint64(0xFFFFFFFFFFFFFFFF)

# Independent stream/domain seed for sampling to avoid correlation with split stream.
SAMPLE_STREAM_DOMAIN_SEED = np.uint64(0xD6E8FEB86659FD93)


def splitmix64(values: np.ndarray) -> np.ndarray:
    """Vectorized SplitMix64 mix on uint64 values."""
    x = values.astype(np.uint64, copy=False)
    x = (x + SPLITMIX64_INCREMENT) & U64_MASK
    x = (x ^ (x >> np.uint64(30))) * SPLITMIX64_MUL1
    x = x & U64_MASK
    x = (x ^ (x >> np.uint64(27))) * SPLITMIX64_MUL2
    x = x & U64_MASK
    x = x ^ (x >> np.uint64(31))
    return x


def uniform01_from_u64(hash_values: np.ndarray) -> np.ndarray:
    """Map uint64 hashes to deterministic float64 values in [0, 1)."""
    return ((hash_values >> np.uint64(11)).astype(np.float64)) * (1.0 / float(1 << 53))


def keyed_uniform01(*, key_values: np.ndarray, seed: int, domain_seed: int = 0) -> np.ndarray:
    """Deterministic keyed uniform stream in [0, 1), vectorized and order-independent."""
    seed_u64 = np.uint64(int(seed) & 0xFFFFFFFFFFFFFFFF)
    domain_u64 = np.uint64(int(domain_seed) & 0xFFFFFFFFFFFFFFFF)
    base = key_values.astype(np.uint64, copy=False) ^ seed_u64 ^ domain_u64
    return uniform01_from_u64(splitmix64(base))
