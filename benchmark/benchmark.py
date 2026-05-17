"""
Benchmarking module for comparing DFT, FFT, and numpy.fft performance.

Measures execution time of each implementation across increasing signal
sizes to empirically demonstrate the O(N²) vs O(N log N) complexity
difference.
"""

import time
from typing import Callable
import numpy as np
from fft.dft import DFT
from fft.fft import FFT


class Benchmark:
    """
    Benchmarks DFT, FFT, and numpy.fft.fft across increasing signal sizes.

    Generates random signals of increasing length (powers of 2),
    measures execution time for each implementation, and stores
    results for plotting and comparison.
    """

    def __init__(self, sizes: list[int] | None = None):
        """
        Initialize benchmark with signal sizes to test.

        Parameters
        ----------
        sizes : list[int] or None
            List of signal lengths to benchmark. Must be powers of 2.
            Defaults to [8, 16, 32, 64, 128, 256, 512, 1024].
        """
        self.sizes = sizes or [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
        self.dft = DFT()
        self.fft = FFT()
        self.results = {
            "sizes": np.array(self.sizes),
            "dft_times": np.zeros(len(self.sizes)),
            "dft_vec_times": np.zeros(len(self.sizes)),
            "fft_times": np.zeros(len(self.sizes)),
            "numpy_times": np.zeros(len(self.sizes)),
        }

    def _time_function(
        self, fn: Callable, signal: np.ndarray, repeats: int = 5
    ) -> float:
        """
        Measure average execution time of a function over multiple runs.

        Running multiple times and averaging reduces the effect of
        random system noise on timing measurements.

        Parameters
        ----------
        fn : callable
            Function to time. Must accept a single numpy array argument.
        signal : np.ndarray
            Input signal to pass to the function.
        repeats : int
            Number of times to run the function. Default is 5.

        Returns
        -------
        float
            Average execution time in seconds.
        """
        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            fn(signal)
            end = time.perf_counter()
            times.append(end - start)
        return float(np.mean(times))

    def run(self) -> dict:
        """
        Run the benchmark across all signal sizes.

        For each size, generates a random signal and times DFT, FFT,
        and numpy.fft.fft. Skips DFT for large sizes (>256) since
        O(N²) becomes prohibitively slow.

        Returns
        -------
        dict
            Dictionary containing sizes and timing arrays for each
            implementation.
        """
        print(
            f"{'Size':<10} {'DFT (s)':<15} {'DFT Vec (s)':<15} {'FFT (s)':<15} {'NumPy (s)':<15}"
        )
        print("─" * 70)

        for i, N in enumerate(self.sizes):
            # Generate random signal of length N
            rng = np.random.default_rng(seed=42)
            signal = rng.random(N)

            # DFT becomes too slow beyond N=256 — skip to avoid hanging
            if N <= 512:
                dft_time = self._time_function(self.dft.forward, signal)
                dft_vec_time = self._time_function(self.dft.forward_vectorized, signal)
            else:
                dft_time = float("nan")
                dft_vec_time = float("nan")

            fft_time = self._time_function(self.fft.forward, signal)
            numpy_time = self._time_function(np.fft.fft, signal)

            self.results["dft_times"][i] = dft_time
            self.results["fft_times"][i] = fft_time
            self.results["dft_vec_times"][i] = dft_vec_time
            self.results["numpy_times"][i] = numpy_time

            dft_display = f"{dft_time:.6f}" if not np.isnan(dft_time) else "skipped"
            dft_vec_display = (
                f"{dft_vec_time:.6f}" if not np.isnan(dft_vec_time) else "skipped"
            )
            print(
                f"{N:<10} {dft_display:<15} {dft_vec_display:<15} {fft_time:<15.6f} {numpy_time:<15.6f}"
            )

        return self.results

    def verify_correctness(self) -> None:
        """
        Verify that DFT and FFT produce identical results to numpy.fft.fft,
        and that inverse(forward(signal)) reconstructs the original signal.

        Tests on a fixed signal of length 64.

        Raises
        ------
        AssertionError
            If any implementation deviates from expected results.
        """
        rng = np.random.default_rng(seed=42)
        signal = rng.random(64)

        expected = np.fft.fft(signal)
        dft_result = self.dft.forward(signal)
        fft_result = self.fft.forward(signal)
        dft_vec_result = self.dft.forward_vectorized(signal)

        # Verify forward results match numpy
        assert np.allclose(
            dft_result, expected, atol=1e-10
        ), "DFT forward result does not match numpy.fft.fft"
        assert np.allclose(
            fft_result, expected, atol=1e-10
        ), "FFT forward result does not match numpy.fft.fft"
        assert np.allclose(
            dft_vec_result, expected, atol=1e-10
        ), "DFT vectorized result does not match numpy.fft.fft"
        print("✓ Forward — DFT, vectorized DFT, and FFT match numpy.fft.fft")

        # Verify inverse reconstructs the original signal
        dft_reconstructed = np.real(self.dft.inverse(dft_result))
        fft_reconstructed = np.real(self.fft.inverse(fft_result))
        dft_vec_reconstructed = np.real(self.dft.inverse(dft_vec_result))

        assert np.allclose(
            dft_reconstructed, signal, atol=1e-10
        ), "DFT inverse did not reconstruct the original signal"
        assert np.allclose(
            fft_reconstructed, signal, atol=1e-10
        ), "FFT inverse did not reconstruct the original signal"
        assert np.allclose(
            np.real(dft_vec_reconstructed), signal, atol=1e-10
        ), "DFT vectorized inverse did not reconstruct original signal"
        print("✓ Inverse — both implementations reconstruct original signal")

        # Verify DFT and FFT inverses match each other
        assert np.allclose(
            dft_reconstructed, fft_reconstructed, dft_vec_reconstructed, atol=1e-10
        ), "DFT and FFT inverse results do not match each other"
        print("✓ Consistency — DFT and FFT inverses produce identical results")
