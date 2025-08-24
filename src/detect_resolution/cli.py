import argparse
import logging

from detect_resolution.analyzer import find_resolution
from detect_resolution.input import extract_sample_frames
from detect_resolution.resolutions import find_typical_resolution_close_to


DEFAULT_N_SAMPLES = 20


def main():
    args = parse_args()

    frames = extract_sample_frames(args.input, n_samples=args.n_samples)

    resolution = find_resolution(frames, args.debug_images)
    print(f'Original resolution: {frames[0].shape[1]}x{frames[0].shape[0]}')
    print(f'Estimated resolution: {resolution[0]}x{resolution[1]}')
    known_resolution = find_typical_resolution_close_to(resolution)
    if known_resolution:
        print(f'Maybe {known_resolution}?')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='Path to the input file')
    parser.add_argument('--n-samples', '-n', type=int, default=DEFAULT_N_SAMPLES, metavar='N',
                        help=f'Number of frames to sample from the input file (default={DEFAULT_N_SAMPLES})')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--debug-images', action='store_true', help='Show debug images')

    args = parser.parse_args()

    if args.debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(format='%(levelname)-8s %(message)s')
    logging.getLogger('detect_resolution').setLevel(level)

    return args
