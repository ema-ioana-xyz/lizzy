import lightning as L
from torch import Tensor

from utils.camera_intrinsics import CameraIntrinsics
from utils.image_shape import ImageShape


class Manydepth_shim(L.LightningModule):
    def __init__(
        self, camera_intrinsics: CameraIntrinsics, input_shape: ImageShape, model
    ):
        super().__init__()
        self.model = model

    def forward(self, depth: Tensor):
        return normals
