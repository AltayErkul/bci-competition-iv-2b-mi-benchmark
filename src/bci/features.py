"""Feature helpers for classical motor-imagery EEG decoders."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, sosfiltfilt


DEFAULT_FILTER_BANK: tuple[tuple[float, float], ...] = (
    (8.0, 12.0),
    (12.0, 16.0),
    (16.0, 20.0),
    (20.0, 24.0),
    (24.0, 28.0),
    (28.0, 32.0),
)


@dataclass(frozen=True)
class FilterBand:
    low_hz: float
    high_hz: float


def normalize_bands(
    bands: list[list[float]] | list[tuple[float, float]] | tuple[tuple[float, float], ...],
) -> tuple[FilterBand, ...]:
    """Return validated filter-band objects."""

    normalized: list[FilterBand] = []
    for band in bands:
        low, high = float(band[0]), float(band[1])
        if low <= 0 or high <= low:
            raise ValueError(f"Invalid filter band: {band}")
        normalized.append(FilterBand(low_hz=low, high_hz=high))
    return tuple(normalized)


def bandpass_epochs(
    X: np.ndarray,
    *,
    sfreq: float,
    low_hz: float,
    high_hz: float,
    order: int = 4,
) -> np.ndarray:
    """Band-pass epochs shaped as ``(trials, channels, samples)``."""

    if X.ndim != 3:
        raise ValueError("X must have shape (trials, channels, samples)")
    sos = butter(order, [low_hz, high_hz], btype="bandpass", fs=sfreq, output="sos")
    return sosfiltfilt(sos, X, axis=-1)


def filter_bank_epochs(
    X: np.ndarray,
    *,
    sfreq: float,
    bands: tuple[FilterBand, ...],
    order: int = 4,
) -> list[np.ndarray]:
    """Apply a filter bank to epochs."""

    return [
        bandpass_epochs(
            X,
            sfreq=sfreq,
            low_hz=band.low_hz,
            high_hz=band.high_hz,
            order=order,
        )
        for band in bands
    ]
