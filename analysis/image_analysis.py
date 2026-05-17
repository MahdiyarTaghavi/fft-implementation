"""
Image frequency analysis using numpy's 2D FFT.

Demonstrates how FFT concepts extend to 2D data (images):
    - Images have spatial frequencies instead of temporal frequencies
    - Low spatial frequencies = smooth gradients, general shapes
    - High spatial frequencies = sharp edges, fine details, noise

Pipeline:
    Load image → 2D FFT → visualize spectrum → apply filter → reconstruct
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)


class ImageAnalyzer:
    """
    Analyzes and filters images using 2D FFT.

    Uses numpy.fft.fft2 for the 2D transform — the same mathematical
    principle as 1D FFT both but applied to spatial dimensions
    simultaneously.
    """

    def load_grayscale(self, path: str) -> np.ndarray:
        """
        Load an image and convert to grayscale.

        Parameters
        ----------
        path : str
            Path to image file.

        Returns
        -------
        np.ndarray
            2D array of pixel values in range [0, 255].

        Raises
        ------
        FileNotFoundError
            If image file does not exist.
        ValueError
            If image cannot be converted to grayscale.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        image = plt.imread(str(path))

        if image.ndim == 3:
            # Convert RGB to grayscale using luminance weights
            # Human eye is most sensitive to green, least to blue
            grayscale = (
                0.2989 * image[:, :, 0] +
                0.5870 * image[:, :, 1] +
                0.1140 * image[:, :, 2]
            )
        elif image.ndim == 2:
            grayscale = image
        else:
            raise ValueError(
                f"Unexpected image shape: {image.shape}. "
                "Expected 2D or 3D array."
            )

        # Normalize to [0, 255] if float image
        if grayscale.max() <= 1.0:
            grayscale = grayscale * 255.0

        return grayscale.astype(float)

    def forward_2d(self, image: np.ndarray) -> np.ndarray:
        """
        Compute the 2D FFT of an image.

        Applies FFT along both spatial dimensions. Result is shifted
        so that the zero frequency (DC) component is at the center
        of the spectrum — easier to visualize and filter.

        Parameters
        ----------
        image : np.ndarray
            2D grayscale image array.

        Returns
        -------
        np.ndarray
            Complex 2D frequency spectrum, zero-centered.
        """
        spectrum = np.fft.fft2(image)
        # Shift zero frequency to center
        return np.fft.fftshift(spectrum)

    def inverse_2d(self, spectrum: np.ndarray) -> np.ndarray:
        """
        Reconstruct image from a 2D frequency spectrum.

        Reverses fftshift before applying inverse FFT.

        Parameters
        ----------
        spectrum : np.ndarray
            Complex 2D frequency spectrum (zero-centered).

        Returns
        -------
        np.ndarray
            Reconstructed image as a real-valued 2D array.
        """
        # Reverse the shift before inverse FFT
        unshifted = np.fft.ifftshift(spectrum)
        return np.real(np.fft.ifft2(unshifted))

    def magnitude_spectrum(self, spectrum: np.ndarray) -> np.ndarray:
        """
        Compute log-scaled magnitude spectrum for visualization.

        Log scale is used because the DC component (center) is
        typically orders of magnitude larger than other frequencies,
        making the rest invisible without scaling.

        Parameters
        ----------
        spectrum : np.ndarray
            Complex 2D frequency spectrum.

        Returns
        -------
        np.ndarray
            Log-scaled magnitude array for visualization.
        """
        return np.log1p(np.abs(spectrum))

    def low_pass_filter_2d(
        self, spectrum: np.ndarray, radius: int
    ) -> np.ndarray:
        """
        Apply a circular low-pass filter to a 2D spectrum.

        Creates a circular mask centered at the DC component.
        Frequencies outside the circle are zeroed out.
        Keeps smooth gradients, removes edges and noise.

        Parameters
        ----------
        spectrum : np.ndarray
            Complex 2D frequency spectrum (zero-centered).
        radius : int
            Radius of the circular mask in pixels.
            Larger radius = less smoothing.

        Returns
        -------
        np.ndarray
            Filtered 2D frequency spectrum.
        """
        rows, cols = spectrum.shape
        center_row, center_col = rows // 2, cols // 2

        # Build circular mask — True inside circle, False outside
        row_indices, col_indices = np.ogrid[:rows, :cols]
        distance = np.sqrt(
            (row_indices - center_row) ** 2 +
            (col_indices - center_col) ** 2
        )
        mask = distance <= radius

        filtered = spectrum.copy()
        filtered[~mask] = 0
        return filtered

    def high_pass_filter_2d(
        self, spectrum: np.ndarray, radius: int
    ) -> np.ndarray:
        """
        Apply a circular high-pass filter to a 2D spectrum.

        Inverse of low-pass — zeroes out frequencies inside
        the circle. Keeps edges and fine details, removes
        smooth gradients and DC offset.

        Parameters
        ----------
        spectrum : np.ndarray
            Complex 2D frequency spectrum (zero-centered).
        radius : int
            Radius of the circular mask in pixels.
            Larger radius = more aggressive edge detection.

        Returns
        -------
        np.ndarray
            Filtered 2D frequency spectrum.
        """
        rows, cols = spectrum.shape
        center_row, center_col = rows // 2, cols // 2

        row_indices, col_indices = np.ogrid[:rows, :cols]
        distance = np.sqrt(
            (row_indices - center_row) ** 2 +
            (col_indices - center_col) ** 2
        )
        mask = distance <= radius

        filtered = spectrum.copy()
        filtered[mask] = 0
        return filtered

    def plot_analysis(
        self,
        image: np.ndarray,
        spectrum: np.ndarray,
        low_passed: np.ndarray,
        high_passed: np.ndarray,
        filename: str = "image_analysis",
    ) -> None:
        """
        Plot full image analysis — original, spectrum, and filtered results.

        Parameters
        ----------
        image : np.ndarray
            Original grayscale image.
        spectrum : np.ndarray
            2D frequency spectrum (zero-centered).
        low_passed : np.ndarray
            Reconstructed image after low-pass filtering.
        high_passed : np.ndarray
            Reconstructed image after high-pass filtering.
        filename : str
            Output filename without extension.
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle("Image Frequency Analysis using 2D FFT", fontsize=16)

        # Original image
        axes[0, 0].imshow(image, cmap="gray")
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis("off")

        # Magnitude spectrum — log scaled
        axes[0, 1].imshow(self.magnitude_spectrum(spectrum), cmap="inferno")
        axes[0, 1].set_title("Frequency Spectrum (log scale)")
        axes[0, 1].axis("off")

        # Low pass result — blurred
        axes[1, 0].imshow(low_passed, cmap="gray")
        axes[1, 0].set_title("Low-Pass Filter (smoothed)")
        axes[1, 0].axis("off")

        # High pass result — edges
        axes[1, 1].imshow(np.abs(high_passed), cmap="gray")
        axes[1, 1].set_title("High-Pass Filter (edges)")
        axes[1, 1].axis("off")

        plt.tight_layout()
        plt.savefig(OUTPUTS_DIR / f"{filename}.png", dpi=150)
        plt.close()
        print(f"✓ Saved {filename}.png")

    def run(self, image_path: str) -> None:
        """
        Run full image analysis pipeline.

        Parameters
        ----------
        image_path : str
            Path to input image file.
        """
        print(f"Loading image: {image_path}")
        image = self.load_grayscale(image_path)
        print(f"Image shape: {image.shape}")

        # Forward 2D FFT
        spectrum = self.forward_2d(image)

        # Apply filters
        low_passed_spectrum = self.low_pass_filter_2d(spectrum, radius=30)
        high_passed_spectrum = self.high_pass_filter_2d(spectrum, radius=30)

        # Reconstruct filtered images
        low_passed = self.inverse_2d(low_passed_spectrum)
        high_passed = self.inverse_2d(high_passed_spectrum)

        # Plot everything
        self.plot_analysis(image, spectrum, low_passed, high_passed)
        print("✓ Image analysis complete")