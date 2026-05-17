"""
Fast Fourier Transform (FFT) implementation using the Cooley-Tukey algorithm.

This module implements the same mathematical operation as DFT but uses a
recursive divide and conquer approach to achieve O(N log N) complexity
instead of the naive O(N²) approach.

The key insight: split the signal into even and odd indexed samples,
compute a half-size DFT on each, then combine using twiddle factors.
Each output bin X[k] and X[k + N/2] share the same intermediate
calculations — computed once, used twice.
"""

import numpy as np


class FFT:
    """
    Cooley-Tukey recursive FFT implementation.

    Computes the same result as DFT but in O(N log N) time by
    recursively splitting the signal into even and odd indexed samples.

    Notes
    -----
    Input length N must be a power of 2 (e.g. 2, 4, 8, 16, 32...).
    This is a requirement of the Cooley-Tukey algorithm — the signal
    must be divisible in half at every recursion level.
    """

    def forward(self, signal: np.ndarray) -> np.ndarray:
        """
        Compute the FFT of a signal using Cooley-Tukey divide and conquer.

        Parameters
        ----------
        signal : np.ndarray
            Input signal as a 1D array. Length must be a power of 2.

        Returns
        -------
        np.ndarray
            Complex array of length N containing the frequency spectrum.
            Identical result to DFT.forward() but computed in O(N log N).
        ValueError
                If signal is empty or not a 1D array.
        """
        if signal.ndim != 1:
            raise ValueError(f"Signal must be a 1D array, got shape {signal.shape}.")
        if len(signal) == 0:
            raise ValueError("Signal must not be empty.")

        signal = self._pad_to_power_of_2(signal)
        result = self._fft_recursive(signal.astype(complex))
        self._verify(signal, result)
        return result

    def inverse(self, spectrum: np.ndarray) -> np.ndarray:
        """
        Compute the inverse FFT to reconstruct a signal from its spectrum.

        Uses the conjugate trick: IFFT(X) = conj(FFT(conj(X))) / N
        This means we can reuse the forward FFT implementation entirely.

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
        spectrum = self._pad_to_power_of_2(spectrum)
        N = len(spectrum)

        conjugated = np.conj(spectrum)
        result = self._fft_recursive(conjugated)
        return np.conj(result) / N

    def _fft_recursive(self, signal: np.ndarray) -> np.ndarray:
        """
        Core recursive Cooley-Tukey FFT implementation.

        Parameters
        ----------
        signal : np.ndarray
            Complex input signal of length N (must be power of 2).

        Returns
        -------
        np.ndarray
            Complex frequency spectrum of length N.
        """
        N = len(signal)

        # Base case — DFT of a single sample is the sample itself
        if N == 1:
            return signal

        # Split into even and odd indexed samples
        even = signal[::2]
        odd = signal[1::2]

        # Recursively compute FFT of each half
        E = self._fft_recursive(even)
        O = self._fft_recursive(odd)

        # Compute twiddle factors for this level
        # W[k] = e^(-2πi × k / N) for k = 0 to N/2 - 1
        k = np.arange(N // 2)
        W = np.exp(-2j * np.pi * k / N)

        # Combine even and odd results
        # X[k]       = E[k] + W[k] × O[k]   (first half)
        # X[k + N/2] = E[k] - W[k] × O[k]   (second half)
        WO = W * O
        return np.concatenate([E + WO, E - WO])

    def _pad_to_power_of_2(self, signal: np.ndarray) -> np.ndarray:
        """
        Pad signal with zeros to the next power of 2 if necessary.

        Parameters
        ----------
        signal : np.ndarray
            Input signal of any length.

        Returns
        -------
        np.ndarray
            Zero-padded signal whose length is a power of 2.
        """
        N = len(signal)
        if N & (N - 1) == 0:
            return signal  # already a power of 2

        next_pow2 = 2 ** int(np.ceil(np.log2(N)))
        print(
            f"Warning: signal length {N} is not a power of 2. "
            f"Zero-padding to {next_pow2}."
        )
        padded = np.zeros(next_pow2, dtype=complex)
        padded[:N] = signal
        return padded

    def _verify(self, signal: np.ndarray, result: np.ndarray) -> None:
        """
        Verify FFT output against numpy's reference implementation.

        Parameters
        ----------
        signal : np.ndarray
            Original input signal.
        result : np.ndarray
            FFT output to verify.

        Raises
        ------
        AssertionError
            If result deviates from numpy.fft.fft beyond numerical tolerance.
        """
        assert np.allclose(
            result, np.fft.fft(signal), atol=1e-10
        ), "FFT output does not match numpy.fft.fft — implementation error."
