import numpy as np
from filters.base import Filter


class LowPassFilter(Filter):
    """Low-pass filter — keeps low frequencies, removes high frequencies."""

    def apply(self, spectrum: np.ndarray, cutoff: int) -> np.ndarray:
        """
        Zero out all frequency bins above the cutoff.

        Parameters
        ----------
        spectrum : np.ndarray
            Complex frequency spectrum.
        cutoff : int
            Bin index above which frequencies are zeroed out.

        Returns
        -------
        np.ndarray
            Filtered spectrum with high frequencies removed.
        """
        if cutoff < 0:
            raise ValueError(f"Cutoff must be non-negative, got {cutoff}.")
        filtered = spectrum.copy()
        filtered[cutoff:-cutoff] = 0
        return filtered
