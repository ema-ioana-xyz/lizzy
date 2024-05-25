import dataclasses


@dataclasses.dataclass
class AlunArguments:
    sampling_ratio: float
    importance_ratio: float
    architecture: str

    def __init__(self):
        self.sampling_ratio = 0.4
        self.importance_ratio = 0.7
        self.architecture = "BN"
