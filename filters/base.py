from abc import ABC, abstractmethod
import numpy as np


class Filter(ABC):
    """Abstract base class for frequency domain filters."""

    @abstractmethod
    def apply(self, spectrum: np.ndarray, cutoff: int) -> np.ndarray:
        """
        Apply filter to a frequency spectrum.

        Parameters
        ----------
        spectrum : np.ndarray
            Complex frequency spectrum produced by FFT.forward().
        cutoff : int
            Cutoff bin index.

        Returns
        -------
        np.ndarray
            Filtered frequency spectrum.
        """
        pass
