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
        if signal.ndim != 1:
            raise ValueError(
                f"Signal must be a 1D array, got shape {signal.shape}."
            )
        if len(signal) == 0:
            raise ValueError(
                "Signal must not be empty."
            )

        N = len(signal)
        X = np.zeros(N, dtype=complex)

        for k in range(N):
            for n in range(N):
                X[k] += signal[n] * np.exp(-2j * np.pi * k * n / N)

        self._verify(signal, X)
        return X

    def forward_vectorized(self, signal: np.ndarray) -> np.ndarray:
        """
        Compute DFT using vectorized numpy operations instead of Python loops.

        Achieves the same O(N²) complexity as forward() but runs significantly
        faster by replacing nested Python loops with numpy matrix multiplication.

        Constructs an N×N matrix W where W[k,n] = e^(-2πi × k × n / N),
        then computes X = W @ x in a single numpy operation.

        Parameters
        ----------
        signal : np.ndarray
            Input signal as a 1D array of real or complex values.

        Returns
        -------
        np.ndarray
            Complex frequency spectrum identical to forward().
        """
        if signal.ndim != 1:
            raise ValueError(
                f"Signal must be a 1D array, got shape {signal.shape}."
            )
        if len(signal) == 0:
            raise ValueError(
                "Signal must not be empty."
            )

        N = len(signal)
        n = np.arange(N)
        k = np.arange(N).reshape(N, 1)  # column vector — shape (N, 1)
        W = np.exp(-2j * np.pi * k * n / N)  # N×N twiddle factor matrix
        result = W @ signal.astype(complex)  # matrix multiplication — shape (N,)

        self._verify(signal, result)
        return result

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
