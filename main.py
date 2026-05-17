"""
FFT Project — Main Entry Point

Runs the complete FFT analysis pipeline:
    1. Verify correctness of DFT and FFT implementations
    2. Benchmark DFT vs FFT vs numpy across increasing signal sizes
    3. Generate and filter a composite signal
    4. Visualize signal analysis results
    5. Apply 2D FFT analysis to an image

All plots are saved to the outputs/ directory.
"""

from pathlib import Path
import numpy as np
from benchmark.benchmark import Benchmark
from fft.dft import DFT
from fft.fft import FFT
from filters import LowPassFilter, HighPassFilter, FilterPipeline
from visualize.plots import SignalPlotter, BenchmarkPlotter
from analysis.image_analysis import ImageAnalyzer

RANDOM_SEED = 42
SAMPLE_RATE = 512
N = 512
IMAGE_PATH = "test_image.png"


def generate_composite_signal(
    n: int,
    sample_rate: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a composite signal made of three sine waves.

    Parameters
    ----------
    n : int
        Number of samples.
    sample_rate : int
        Samples per second.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        - t : time axis array
        - signal : composite signal array
    """
    rng = np.random.default_rng(seed=seed)
    t = np.linspace(0, 1, n)
    signal = (
        np.sin(2 * np.pi * 5 * t)  # 5 Hz — low frequency
        + np.sin(2 * np.pi * 50 * t)  # 50 Hz — mid frequency
        + np.sin(2 * np.pi * 120 * t)  # 120 Hz — high frequency
        + rng.normal(0, 0.2, n)  # random noise
    )
    return t, signal


def run_verification() -> None:
    """Verify correctness of DFT and FFT implementations."""
    print("\n" + "═" * 55)
    print("  PHASE 1 — Correctness Verification")
    print("═" * 55)
    bench = Benchmark()
    bench.verify_correctness()


def run_benchmark() -> dict:
    """Benchmark DFT vs FFT vs numpy and plot results."""
    print("\n" + "═" * 55)
    print("  PHASE 2 — Benchmarking")
    print("═" * 55)
    bench = Benchmark()
    results = bench.run()
    BenchmarkPlotter().plot_timing(results)
    return results


def run_signal_analysis() -> None:
    """Generate composite signal, apply filters, and visualize."""
    print("\n" + "═" * 55)
    print("  PHASE 3 — Signal Analysis & Filtering")
    print("═" * 55)

    t, signal = generate_composite_signal(N, SAMPLE_RATE, RANDOM_SEED)
    fft = FFT()
    pipeline = FilterPipeline(fft)
    plotter = SignalPlotter()

    # Plot original signal
    plotter.plot_signal(
        signal,
        "Composite Signal (5Hz + 50Hz + 120Hz + noise)",
        "original_signal",
        SAMPLE_RATE,
    )

    # Forward FFT
    spectrum = fft.forward(signal)

    # Plot spectrum
    plotter.plot_spectrum(spectrum, "Frequency Spectrum", "spectrum", SAMPLE_RATE)

    # Low pass filter — keep only low frequencies
    lp_spectrum, lp_filtered_spectrum, lp_signal = pipeline.run(
        signal, LowPassFilter(), cutoff=20
    )
    plotter.plot_filter_comparison(
        signal,
        lp_signal,
        lp_spectrum,
        lp_filtered_spectrum,
        "Low-Pass Filter (cutoff=20) — removes high frequencies",
        "low_pass_comparison",
        SAMPLE_RATE,
    )
    print("✓ Low-pass filter applied")

    # High pass filter — keep only high frequencies
    hp_spectrum, hp_filtered_spectrum, hp_signal = pipeline.run(
        signal, HighPassFilter(), cutoff=80
    )
    plotter.plot_filter_comparison(
        signal,
        hp_signal,
        hp_spectrum,
        hp_filtered_spectrum,
        "High-Pass Filter (cutoff=80) — removes low frequencies",
        "high_pass_comparison",
        SAMPLE_RATE,
    )
    print("✓ High-pass filter applied")


def run_image_analysis() -> None:
    """Apply 2D FFT analysis to an image."""
    print("\n" + "═" * 55)
    print("  PHASE 4 — Image Frequency Analysis")
    print("═" * 55)

    image_path = Path(IMAGE_PATH)
    if not image_path.exists():
        print(f"⚠ Image not found at {IMAGE_PATH} — skipping image analysis.")
        print("  Place any image named 'test_image.png' in the project root.")
        return

    analyzer = ImageAnalyzer()
    analyzer.run(IMAGE_PATH)


def main() -> None:
    """Run the complete FFT analysis pipeline."""
    print("\n" + "═" * 55)
    print("  FFT Project — Full Analysis Pipeline")
    print("═" * 55)

    run_verification()
    run_benchmark()
    run_signal_analysis()
    run_image_analysis()

    print("\n" + "═" * 55)
    print("  ✓ All phases complete — check outputs/ folder")
    print("═" * 55)


if __name__ == "__main__":
    main()
