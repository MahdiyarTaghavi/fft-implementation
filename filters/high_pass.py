import numpy as np
from filters.base import Filter


class HighPassFilter(Filter):
    """High-pass filter — keeps high frequencies, removes low frequencies."""

    def apply(self, spectrum: np.ndarray, cutoff: int) -> np.ndarray:
        """
        Zero out all frequency bins below the cutoff.

        Parameters
        ----------
        spectrum : np.ndarray
            Complex frequency spectrum.
        cutoff : int
            Bin index below which frequencies are zeroed out.

        Returns
        -------
        np.ndarray
            Filtered spectrum with low frequencies removed.
        """
        if cutoff < 0:
            raise ValueError(f"Cutoff must be non-negative, got {cutoff}.")
        filtered = spectrum.copy()
        filtered[:cutoff] = 0
        filtered[-cutoff:] = 0
        return filtered