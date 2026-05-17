"""
Visualization module for FFT analysis results.

Provides plotting functions for:
    - Time domain signals
    - Frequency spectrums
    - Benchmark timing comparisons
    - Filter effects (before/after)

All plots are saved to the outputs/ directory.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)


class SignalPlotter:
    """
    Plots time domain signals and their frequency spectrums.
    """

    def plot_signal(
        self,
        signal: np.ndarray,
        title: str,
        filename: str,
        sample_rate: int = 1,
    ) -> None:
        """
        Plot a time domain signal.

        Parameters
        ----------
        signal : np.ndarray
            1D array of signal samples.
        title : str
            Plot title.
        filename : str
            Output filename without extension.
        sample_rate : int
            Number of samples per second. Used to label x-axis in seconds.
            Default is 1 (x-axis shows sample index).
        """
        N = len(signal)
        t = np.arange(N) / sample_rate

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(t, signal, color="steelblue", linewidth=0.8)
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Time (s)" if sample_rate > 1 else "Sample index")
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUTPUTS_DIR / f"{filename}.png", dpi=150)
        plt.close()
        print(f"✓ Saved {filename}.png")

    def plot_spectrum(
        self,
        spectrum: np.ndarray,
        title: str,
        filename: str,
        sample_rate: int = 1,
    ) -> None:
        """
        Plot the magnitude spectrum of a frequency domain signal.

        Only plots the first N/2 bins (meaningful frequency range
        up to Nyquist limit).

        Parameters
        ----------
        spectrum : np.ndarray
            Complex frequency spectrum produced by FFT.forward().
        title : str
            Plot title.
        filename : str
            Output filename without extension.
        sample_rate : int
            Sampling rate used to convert bin indices to Hz.
            Default is 1 (x-axis shows bin index).
        """
        N = len(spectrum)
        # Only plot first N/2 bins — upper half is mirror image
        half = N // 2
        magnitudes = np.abs(spectrum[:half])
        freqs = np.arange(half) * sample_rate / N

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(freqs, magnitudes, color="crimson", linewidth=0.8)
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Frequency (Hz)" if sample_rate > 1 else "Bin index")
        ax.set_ylabel("Magnitude")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUTPUTS_DIR / f"{filename}.png", dpi=150)
        plt.close()
        print(f"✓ Saved {filename}.png")

    def plot_filter_comparison(
        self,
        original: np.ndarray,
        filtered: np.ndarray,
        spectrum: np.ndarray,
        filtered_spectrum: np.ndarray,
        title: str,
        filename: str,
        sample_rate: int = 1,
    ) -> None:
        """
        Plot original vs filtered signal and their spectrums side by side.

        Parameters
        ----------
        original : np.ndarray
            Original time domain signal.
        filtered : np.ndarray
            Filtered time domain signal.
        spectrum : np.ndarray
            Original frequency spectrum.
        filtered_spectrum : np.ndarray
            Filtered frequency spectrum.
        title : str
            Overall plot title.
        filename : str
            Output filename without extension.
        sample_rate : int
            Sampling rate for axis labeling.
        """
        N = len(original)
        half = N // 2
        t = np.arange(N) / sample_rate
        freqs = np.arange(half) * sample_rate / N

        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        fig.suptitle(title, fontsize=16)

        # Original signal
        axes[0, 0].plot(t, original, color="steelblue", linewidth=0.8)
        axes[0, 0].set_title("Original Signal")
        axes[0, 0].set_xlabel("Time (s)" if sample_rate > 1 else "Sample")
        axes[0, 0].set_ylabel("Amplitude")
        axes[0, 0].grid(True, alpha=0.3)

        # Original spectrum
        axes[0, 1].plot(freqs, np.abs(spectrum[:half]), color="crimson", linewidth=0.8)
        axes[0, 1].set_title("Original Spectrum")
        axes[0, 1].set_xlabel("Frequency (Hz)" if sample_rate > 1 else "Bin")
        axes[0, 1].set_ylabel("Magnitude")
        axes[0, 1].grid(True, alpha=0.3)

        # Filtered signal
        axes[1, 0].plot(t, filtered, color="steelblue", linewidth=0.8)
        axes[1, 0].set_title("Filtered Signal")
        axes[1, 0].set_xlabel("Time (s)" if sample_rate > 1 else "Sample")
        axes[1, 0].set_ylabel("Amplitude")
        axes[1, 0].grid(True, alpha=0.3)

        # Filtered spectrum
        axes[1, 1].plot(
            freqs, np.abs(filtered_spectrum[:half]), color="crimson", linewidth=0.8
        )
        axes[1, 1].set_title("Filtered Spectrum")
        axes[1, 1].set_xlabel("Frequency (Hz)" if sample_rate > 1 else "Bin")
        axes[1, 1].set_ylabel("Magnitude")
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(OUTPUTS_DIR / f"{filename}.png", dpi=150)
        plt.close()
        print(f"✓ Saved {filename}.png")


class BenchmarkPlotter:
    """
    Plots benchmark timing comparison between DFT, FFT, and numpy.
    """

    def plot_timing(self, results: dict, filename: str = "benchmark") -> None:
        """
        Plot timing comparison on a log-log scale.

        Log-log scale makes the difference between O(N²) and O(N log N)
        visually clear — they appear as straight lines with different slopes.

        Parameters
        ----------
        results : dict
            Dictionary returned by Benchmark.run() containing sizes
            and timing arrays.
        filename : str
            Output filename without extension.
        """
        sizes = results["sizes"]
        dft_times = results["dft_times"]
        dft_vec_times = results["dft_vec_times"]
        fft_times = results["fft_times"]
        numpy_times = results["numpy_times"]

        # Only plot sizes where DFT was not skipped
        valid = ~np.isnan(dft_times)

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.loglog(
            sizes[valid],
            dft_times[valid],
            "o-",
            label="DFT naive O(N²)",
            color="crimson",
        )
        ax.loglog(
            sizes[valid],
            dft_vec_times[valid],
            "s-",
            label="DFT vectorized O(N²)",
            color="orange",
        )
        ax.loglog(
            sizes,
            fft_times,
            "^-",
            label="FFT Cooley-Tukey O(N log N)",
            color="steelblue",
        )
        ax.loglog(
            sizes, numpy_times, "D-", label="numpy.fft.fft (reference)", color="green"
        )

        ax.set_title("DFT vs FFT — Timing Comparison (log-log scale)", fontsize=14)
        ax.set_xlabel("Signal length (N)")
        ax.set_ylabel("Execution time (seconds)")
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)

        plt.tight_layout()
        plt.savefig(OUTPUTS_DIR / f"{filename}.png", dpi=150)
        plt.close()
        print(f"✓ Saved {filename}.png")
