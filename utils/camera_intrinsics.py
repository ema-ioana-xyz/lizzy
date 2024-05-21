import dataclasses
import torch
from torch import Tensor

from utils.image_shape import ImageShape


@dataclasses.dataclass
class CameraIntrinsics:
    # Pixel width and height
    focal_point_x: float
    focal_point_y: float
    # Image center
    principal_point_x: float
    principal_point_y: float

    def get_matrix(self) -> Tensor:
        return torch.tensor(
            [
                [self.focal_point_x, 0, self.principal_point_x, 0],
                [0, self.focal_point_y, self.principal_point_y, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )

    def make_grid(self, shape: ImageShape) -> Tensor:
        grid = torch.ones([shape.height, shape.width, 3])

        grid_y, grid_x = torch.meshgrid(
            torch.arange(shape.height), torch.arange(shape.width), indexing="ij"
        )
        grid[..., 0] = (
            (grid_x - self.principal_point_x * shape.width) / (self.focal_point_x * shape.width)
        )
        grid[..., 1] = (
            (grid_y - self.principal_point_y * shape.width) / (self.focal_point_y * shape.height)
        )

        return grid


class NYU_Intrinsics(CameraIntrinsics):
    def __init__(self):
        focal_point_x = 5.8262448167737955e02 / 640
        focal_point_y = 5.8269103270988637e02 / 480
        principal_point_x = 3.1304475870804731e02 / 640
        principal_point_y = 2.3844389626620386e02 / 480
        super().__init__(
            focal_point_x=focal_point_x,
            focal_point_y=focal_point_y,
            principal_point_x=principal_point_x,
            principal_point_y=principal_point_y,
        )


class Manydepth_Intrinsics(CameraIntrinsics):
    def __init__(self):
        focal_point_x = 0.58
        focal_point_y = 1.92
        principal_point_x = 0.5
        principal_point_y = 0.5
        super().__init__(
            focal_point_x=focal_point_x,
            focal_point_y=focal_point_y,
            principal_point_x=principal_point_x,
            principal_point_y=principal_point_y,
        )

class TFTN_dataset_intrinsics(CameraIntrinsics):
    def __init__(self):
        focal_point_x = 1400 / 640
        focal_point_y = 1380 / 480
        principal_point_x = 350 / 640
        principal_point_y = 200 / 480
        super().__init__(
            focal_point_x=focal_point_x,
            focal_point_y=focal_point_y,
            principal_point_x=principal_point_x,
            principal_point_y=principal_point_y,
        )
