import dataclasses
from typing import Optional


@dataclasses.dataclass
class KnownResolution:
    width: int
    height: int
    name: str

    def __str__(self):
        return f"{self.width}x{self.height} ({self.name})"


KNOWN_RESOLUTIONS = [
    KnownResolution(720, 480, "SD"),
    KnownResolution(720, 576, "SD"),
    KnownResolution(960, 540, "qHD"),
    KnownResolution(1280, 720, "HD"),
    KnownResolution(1920, 1080, "Full HD"),
    KnownResolution(2560, 1440, "QHD"),
    KnownResolution(3840, 2160, "4K UHD"),
    KnownResolution(7680, 4320, "8K UHD"),
]


def find_typical_resolution_close_to(resolution: tuple[int, int]) -> Optional[KnownResolution]:
    # if both values are within 10% of a known resolution, return that resolution
    for known in KNOWN_RESOLUTIONS:
        if (abs(known.width - resolution[0]) / known.width < 0.1 and
                abs(known.height - resolution[1]) / known.height < 0.1):
            return known
    # second pass with only width
    for known in KNOWN_RESOLUTIONS:
        if abs(known.width - resolution[0]) / known.width < 0.1:
            return known
    return None
