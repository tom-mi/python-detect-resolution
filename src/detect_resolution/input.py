import logging

import moviepy
import numpy.typing as npt
from tqdm import tqdm

logger = logging.getLogger(__name__)


def extract_sample_frames(path: str, n_samples: int = 10) -> list[npt.NDArray]:
    logger.debug(f'Reading input file {path}')
    reader = moviepy.VideoFileClip(path, audio=False)
    logger.debug(f'n_frames={reader.n_frames}, fps={reader.fps}, resolution={reader.size}')

    if n_samples >= reader.n_frames:
        return list(reader.iter_frames())

    step = reader.n_frames // (n_samples + 1)  # avoid frames at start/end

    frames = []
    for i in tqdm(range(1, n_samples + 1), desc='Extracting frames', unit='frame'):
        frame = reader.get_frame(i * step / reader.fps)
        frames.append(frame)
    return frames
