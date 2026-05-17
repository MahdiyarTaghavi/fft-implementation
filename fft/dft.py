"""
Naive Discrete Fourier Transform (DFT) implementation.

This module implements the DFT algorithm directly from its mathematical
definition. Time complexity is O(N²) — for each output frequency bin,
we sum over all N input samples.
"""

import numpy as np


class DFT:
    """
    Naive implementation of the Discrete Fourier Transform.

    Computes the DFT directly from its mathematical definition:
        X[k] = Σ x[n] * e^(-2πi * k * n / N)    for n = 0 to N-1

    This is intentionally unoptimized to serve as a correctness baseline
    for comparing against the faster FFT implementation.
    """

    def forward(self, signal: np.ndarray) -> np.ndarray:
        """
        Compute the DFT of a signal.

        Parameters
        ----------
        signal : np.ndarray
            Input signal as a 1D array of real or complex values.

        Returns
        -------
        np.ndarray
            Complex array of length N containing the frequency spectrum.
            X[k] represents the magnitude and phase of frequency bin k.
        """
        N = len(signal)
        X = np.zeros(N, dtype=complex)

        for k in range(N):
            for n in range(N):
                X[k] += signal[n] * np.exp(-2j * np.pi * k * n / N)

        self._verify(signal, X)
        return X

    def inverse(self, spectrum: np.ndarray) -> np.ndarray:
        """
        Compute the inverse DFT — reconstruct signal from frequency spectrum.

        Uses the conjugate formula:
            x[n] = (1/N) Σ X[k] * e^(2πi * k * n / N)

        Parameters
        ----------
        spectrum : np.ndarray
            Complex frequency spectrum produced by forward().

        Returns
        -------
        np.ndarray
            Reconstructed signal as a 1D complex array.
            Take np.real() if original signal was real-valued.
        """
        N = len(spectrum)
        x = np.zeros(N, dtype=complex)

        for n in range(N):
            for k in range(N):
                x[n] += spectrum[k] * np.exp(2j * np.pi * k * n / N)

        return x / N

    def magnitude_spectrum(self, signal: np.ndarray) -> np.ndarray:
        """
        Compute the magnitude of each frequency bin.

        Parameters
        ----------
        signal : np.ndarray
            Input signal as a 1D array.

        Returns
        -------
        np.ndarray
            Real-valued array of magnitudes, one per frequency bin.
        """
        return np.abs(self.forward(signal))

    def _verify(self, signal: np.ndarray, result: np.ndarray) -> None:
        """
        Verify DFT output against numpy's reference implementation.

        Parameters
        ----------
        signal : np.ndarray
            Original input signal.
        result : np.ndarray
            DFT output to verify.

        Raises
        ------
        AssertionError
            If the result deviates from numpy's FFT beyond numerical tolerance.
        """
        assert np.allclose(result, np.fft.fft(signal), atol=1e-10), \
            "DFT output does not match numpy.fft.fft — implementation error."