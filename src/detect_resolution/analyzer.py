import logging

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from detect_resolution.nn import preprocess_image, detect_scaling_factor

logger = logging.getLogger(__name__)


def find_resolution(frames: list[npt.NDArray], debug_images: bool) -> tuple[int, int]:
    if not frames:
        raise ValueError("No frames provided to determine resolution.")

    results = []
    # remember the frame with the lowest / highest value
    best_frame = None
    best_sample = None
    best_value = float('inf')
    worst_frame = None
    worst_sample = None
    worst_value = float('-inf')
    for i, frame in tqdm(enumerate(frames), desc='Processing frames', unit='frame', total=len(frames)):
        for j, sample in enumerate(_get_samples_from_frame(frame, n_samples_per_side=3)):
            # preprocessing
            transformed_sample = preprocess_image(sample)
            detected_scaling_factor = detect_scaling_factor(transformed_sample)
            results.append(detected_scaling_factor)
            logger.debug(f"Frame {i + 1} / Sample {j + 1} Detected scaling factor: {detected_scaling_factor:.2f}")

            # find the best frame with the lowest scaling factor
            if detected_scaling_factor < best_value:
                best_value = detected_scaling_factor
                best_frame = frame
                best_sample = transformed_sample
            # find the worst frame with the highest scaling factor
            if detected_scaling_factor > worst_value:
                worst_value = detected_scaling_factor
                worst_frame = frame
                worst_sample = transformed_sample

    p10_scaling_factor = float(np.percentile(results, 5))
    logger.debug(f"Estimated scaling factor (p10): {p10_scaling_factor}")

    if debug_images:
        _show_debug_images(best_frame=best_frame, worst_frame=worst_frame,
                           best_sample=best_sample, worst_sample=worst_sample,
                           best_value=best_value, worst_value=worst_value,
                           scaling_factors=results, p10_scaling_factor=p10_scaling_factor)

    estimated_width = int(best_frame.shape[1] / p10_scaling_factor)
    estimated_height = int(best_frame.shape[0] / p10_scaling_factor)
    return estimated_width, estimated_height


def _get_samples_from_frame(frame: npt.NDArray, n_samples_per_side: int) -> list[npt.NDArray]:
    h, w, _ = frame.shape
    samples = []
    # subdivide frame into a grid of n_samples_per_side x n_samples_per_side
    for i in range(n_samples_per_side):
        for j in range(n_samples_per_side):
            y1 = i * h // n_samples_per_side
            y2 = (i + 1) * h // n_samples_per_side
            x1 = j * w // n_samples_per_side
            x2 = (j + 1) * w // n_samples_per_side
            sample = frame[y1:y2, x1:x2]
            samples.append(sample)
    return samples


def _show_debug_images(best_frame, worst_frame, best_sample, worst_sample,
                       best_value, worst_value, scaling_factors: list[float],
                       p10_scaling_factor: float):
    plt.figure('detect-resolution debug images', figsize=(10, 5))
    plt.subplot(2, 4, 1)
    plt.imshow(best_frame, cmap='gray')
    plt.title(f'Best Frame (Scaling Factor: {best_value:.2f})')
    plt.axis('off')
    plt.subplot(2, 4, 3)
    plt.imshow(worst_frame, cmap='gray')
    plt.title(f'Worst Frame (Scaling Factor: {worst_value:.2f})')
    plt.axis('off')
    plt.subplot(2, 4, 2)
    plt.imshow(best_sample, cmap='gray', interpolation='none')
    plt.axis('off')
    plt.subplot(2, 4, 4)
    plt.imshow(worst_sample, cmap='gray', interpolation='none')
    plt.axis('off')

    # histogram of scaling factors
    plt.subplot(2, 1, 2)
    plt.hist(scaling_factors, bins=20, color='blue', alpha=0.7)
    plt.title('Scaling Factors Histogram')
    plt.xlabel('Scaling Factor')
    plt.ylabel('Frequency')
    plt.axvline(p10_scaling_factor, color='red', linestyle='dashed', linewidth=1)

    plt.tight_layout(pad=0.5)

    plt.show()
