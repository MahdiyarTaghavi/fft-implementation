# FFT From Scratch — Signal & Image Frequency Analysis

A rigorous Python implementation of the **Discrete Fourier Transform (DFT)**
and **Fast Fourier Transform (FFT)** built entirely from first principles —
no signal processing libraries, no shortcuts. The project progresses from a
naive O(N²) brute-force algorithm to an optimized O(N log N) Cooley-Tukey
recursive implementation, then applies the result to real signal filtering
and image frequency analysis.

---

## Algorithmic Scope

| Implementation | Complexity | Technique |
|---|---|---|
| `DFT.forward()` | O(N²) | Direct mathematical formula — nested loops |
| `DFT.forward_vectorized()` | O(N²) | Numpy matrix multiplication — no loops |
| `FFT.forward()` | O(N log N) | Cooley-Tukey divide and conquer recursion |
| `numpy.fft.fft` | O(N log N) | C-optimized reference — correctness baseline |

Four implementations of the same mathematical operation — each one
demonstrating a distinct optimization technique, from brute force to
algorithmic improvement to hardware-level vectorization.

---

## What Is Built

### Core Algorithms — Zero External Dependencies
- **Naive DFT** — direct summation formula, two nested Python loops,
  intentionally unoptimized to make O(N²) complexity visible
- **Vectorized DFT** — same O(N²) formula expressed as an N×N matrix
  multiplication `X = W @ x`, replacing all loops with a single numpy
  operation
- **Cooley-Tukey FFT** — recursive divide and conquer, splits signal into
  even and odd indexed samples at every level, combines with twiddle
  factors to achieve O(N log N)
- **Inverse DFT and FFT** — signal reconstruction using the conjugate
  trick: `IFFT(X) = conj(FFT(conj(X))) / N`
- **Correctness verification** — every implementation asserted against
  numpy within floating point tolerance `atol=1e-10`

### Signal Filtering — OOP, Strategy Pattern, SOLID Principles
- Abstract `Filter` base class enforcing interface contracts via `ABC`
- `LowPassFilter` — zeroes high frequency bins symmetrically, preserves
  low frequencies
- `HighPassFilter` — zeroes low frequency bins symmetrically, preserves
  high frequencies
- `FilterPipeline` — encapsulates the full FFT → filter → inverse FFT
  pipeline, accepts any `Filter` subclass without modification
  (Open/Closed Principle)

### Benchmarking
- Times all four implementations across signal sizes N=8 to N=1024
- Averages over 5 runs per size to reduce system noise
- Skips naive DFT beyond N=256 where O(N²) becomes prohibitive
- Log-log scale plot — straight lines with different slopes visually
  prove the complexity difference

### Image Frequency Analysis
- Grayscale conversion using ITU-R BT.601 luminance weights
- 2D FFT via `numpy.fft.fft2` — same mathematical principle applied
  to both spatial dimensions simultaneously
- Circular low-pass and high-pass masks in frequency domain
- Image reconstruction via inverse 2D FFT and `ifftshift`
- Before/after visualization — blurred vs edge-detected results

### Visualization — Matplotlib Throughout
- Time domain signal plots
- Frequency magnitude spectrum (log-scaled)
- 2×2 filter comparison grids (signal and spectrum before/after)
- Log-log benchmark timing chart
- 2×2 image analysis grid (original, spectrum, low-pass, high-pass)

---

## Project Structure

```text
fft-project/
├── fft/
│   ├── __init__.py
│   ├── dft.py
│   │   └── Naive + vectorized DFT, forward and inverse
│   └── fft.py
│       └── Cooley-Tukey recursive FFT, forward and inverse
├── filters/
│   ├── __init__.py
│   ├── base.py
│   │   └── Abstract Filter base class (ABC)
│   ├── low_pass.py
│   │   └── LowPassFilter
│   ├── high_pass.py
│   │   └── HighPassFilter
│   └── pipeline.py
│       └── FilterPipeline
├── benchmark/
│   ├── __init__.py
│   └── benchmark.py
│       └── Timing comparison and correctness verification
├── visualize/
│   ├── __init__.py
│   └── plots.py
│       └── All matplotlib visualizations
├── analysis/
│   ├── __init__.py
│   └── image_analysis.py
│       └── 2D FFT image filtering and visualization
├── outputs/
│   └── All generated plots saved here (git ignored)
├── main.py
│   └── Runs the complete pipeline end to end
├── requirements.txt
│   
└── README.md
```
---

## Technical Highlights

### Cooley-Tukey Divide and Conquer

At every recursion level the signal is split into even and odd indexed
samples. Each half is a smaller DFT — solved recursively. Results are
combined using twiddle factors:

- X[k] = E[k] + W_N^k * O[k]  
- X[k + N/2] = E[k] - W_N^k * O[k]  

Where:

- W_N^k = e^(-2π i k / N)

`X[k]` and `X[k + N/2]` share intermediate values `E[k]` and `O[k]`
— computed once, used twice. This sharing is what reduces complexity
from O(N²) to O(N log N). The recursion bottoms out at N=1 where the
DFT of a single sample is the sample itself.

### Vectorized DFT — Matrix Form

The DFT summation is equivalent to a matrix multiplication:

X = W · x

Where:

W[k, n] = e^(-2πi · k · n / N)

Shapes:
- W : (N × N)
- x : (N,)
- X : (N,)N,)

`W` is constructed using `np.ogrid` for memory-efficient broadcasting,
then multiplied against the signal in a single C-level numpy operation.
Same O(N²) complexity as the loop version — but dramatically faster
due to vectorized CPU instructions.

### Strategy Pattern

The filter design follows the Open/Closed Principle. `FilterPipeline`
never needs to change regardless of how many filter types are added:

```python
# Any Filter subclass works without modifying FilterPipeline
pipeline.run(signal, LowPassFilter(), cutoff=20)
pipeline.run(signal, HighPassFilter(), cutoff=80)
pipeline.run(signal, BandPassFilter(), cutoff=30)  # future extension
```

### Correctness Verification — Round Trip

Every implementation verified in two directions:

```python
# Forward matches numpy reference
assert np.allclose(dft.forward(signal), np.fft.fft(signal), atol=1e-10)
assert np.allclose(fft.forward(signal), np.fft.fft(signal), atol=1e-10)

# Inverse reconstructs original signal
assert np.allclose(np.real(dft.inverse(dft.forward(signal))), signal, atol=1e-10)
assert np.allclose(np.real(fft.inverse(fft.forward(signal))), signal, atol=1e-10)
```

---

## Setup

### Prerequisites
- Anaconda or Miniconda installed
- Works on both **Windows** and **Linux**

### 1. Clone the repository

```bash
git clone https://github.com/your-username/fft-project.git
cd fft-project
```

### 2. Create and activate Anaconda environment

```bash
conda env create -f environment.yml
conda activate fft-project
```

Or using pip directly:

```bash
pip install -r requirements.txt
```

### 3. Run the full pipeline

```bash
python main.py
```

All plots are saved automatically to the `outputs/` folder.

---

## Output Plots

| File | Description |
|---|---|
| `original_signal.png` | Composite signal (3 sine waves + noise) |
| `spectrum.png` | Frequency magnitude spectrum |
| `low_pass_comparison.png` | Signal and spectrum before/after low-pass |
| `high_pass_comparison.png` | Signal and spectrum before/after high-pass |
| `benchmark.png` | Timing comparison — log-log scale |
| `image_analysis.png` | Original, spectrum, low-pass, high-pass image |

---

## Dependencies

| Package | Purpose |
|---|---|
| **numpy** | Array operations, matrix multiplication, reference FFT |
| **matplotlib** | All plots and visualizations |

No machine learning libraries. No web frameworks. No databases.
Pure scientific Python — numpy and matplotlib throughout.

---

## Code Quality

- **PEP8 compliant** — verified with `flake8`
- **NumPy-style docstrings** on every class and function
- **Modular structure** — each concern isolated in its own module
- **OOP throughout** — classes with single responsibilities
- **Abstract base classes** — `Filter` enforces interface contracts
- **`pathlib.Path`** for all file system operations
- **`RANDOM_SEED = 42`** — fully reproducible results
- **Exception handling** — empty signals, invalid shapes, missing files
- **Zero dead code** — every method is used and purposeful
- **Developed and tested on Windows and Linux**
