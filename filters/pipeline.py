import numpy as np
from fft.fft import FFT
from filters.base import Filter


class FilterPipeline:
    """
    Applies a frequency domain filter to a signal.

    Full pipeline: FFT → filter → inverse FFT.
    Accepts any Filter subclass — open for extension, closed for modification.
    """

    def __init__(self, fft: FFT):
        """
        Initialize pipeline with an FFT instance.

        Parameters
        ----------
        fft : FFT
            FFT instance used for forward and inverse transforms.
        """
        self._fft = fft

    def run(
        self,
        signal: np.ndarray,
        filter: Filter,
        cutoff: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run the full filter pipeline on a signal.

        Parameters
        ----------
        signal : np.ndarray
            Input time domain signal.
        filter : Filter
            Filter instance to apply.
        cutoff : int
            Cutoff bin index passed to the filter.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            - spectrum : frequency spectrum before filtering
            - filtered_spectrum : frequency spectrum after filtering
            - filtered_signal : reconstructed signal after filtering
        """
        spectrum = self._fft.forward(signal)
        filtered_spectrum = filter.apply(spectrum, cutoff)
        filtered_signal = np.real(self._fft.inverse(filtered_spectrum))
        return spectrum, filtered_spectrum, filtered_signal
