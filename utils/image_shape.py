import dataclasses


@dataclasses.dataclass
class ImageShape:
    width: int
    height: int
    channels: int
