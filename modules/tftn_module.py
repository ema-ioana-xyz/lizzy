import lightning as L
import torch
from torch import Tensor
from torch.nn.functional import unfold

from utils.camera_intrinsics import CameraIntrinsics
from utils.image_shape import ImageShape
import einops as e


class TFTN_module(L.LightningModule):
    def __init__(self, camera_intrinsics: CameraIntrinsics, input_shape: ImageShape):
        super().__init__()
        self.camera_intrinsics = camera_intrinsics
        self.input_shape = input_shape
        self.intrinsics_derived_grid = camera_intrinsics.make_grid(input_shape).cuda()

    def forward(self, depth: Tensor):
        depth = depth.cuda()

        # Sobel filters for the first derivatives
        kernel_x = torch.tensor(
            [
                [-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1],
            ],
            device="cuda",
            dtype=torch.float,
        )
        kernel_x = e.rearrange(kernel_x, "h w -> 1 1 h w")

        kernel_y = e.rearrange(kernel_x, "1 1 h w -> 1 1 w h")
        depth = e.rearrange(depth, "h w -> 1 1 h w")
        FOCAL_POINT_X = self.camera_intrinsics.focal_point_x
        FOCAL_POINT_Y = self.camera_intrinsics.focal_point_y

        inverse_depth = 1 / depth

        directional_derivative_x = torch.conv2d(inverse_depth, kernel_x, padding="same")
        directional_derivative_y = torch.conv2d(inverse_depth, kernel_y, padding="same")

        directional_derivative_x = e.rearrange(
            directional_derivative_x, "1 1 h w -> h w"
        )
        directional_derivative_y = e.rearrange(
            directional_derivative_y, "1 1 h w -> h w"
        )
        depth = e.rearrange(depth, "1 1 h w -> h w")

        normals_x = directional_derivative_x * FOCAL_POINT_X
        normals_y = directional_derivative_y * FOCAL_POINT_Y

        # Normals Z
        depth = e.rearrange(depth, "h w -> h w 1")
        points_3d = self.intrinsics_derived_grid * depth

        # I'm deviating from the paper by using x, y, z instdead of the delta values
        # (where you subtract the current point's values from the neighbors' values)
        # x_part = FOCAL_POINT_X * directional_derivative_x * points_3d[..., 0]
        # y_part = FOCAL_POINT_Y * directional_derivative_y * points_3d[..., 1]
        points_3d = e.rearrange(points_3d, "h w c -> 1 c h w")
        patches = unfold(points_3d, kernel_size=3, padding=1)
        patches = patches.unflatten(1, [3, 3**2])
        patches = patches.unflatten(
            3, [self.input_shape.height, self.input_shape.width]
        )
        # patches is 1 batch X 3 channels X kern_size^2 X height X width
        patches = e.rearrange(patches, "1 c k h w -> c k h w")

        # Subtract the middle part of each patch
        deltas = patches - e.rearrange(points_3d, "1 c h w -> c 1 h w")

        x_part = deltas[0, ...] * e.rearrange(normals_x, "h w -> 1 h w")
        y_part = deltas[1, ...] * e.rearrange(normals_y, "h w -> 1 h w")
        D_plane_shift_constant = -1
        normals_z = D_plane_shift_constant * (x_part + y_part) / deltas[2, ...]
        normals_z = torch.nan_to_num(normals_z)
        # normals_z = e.reduce(normals_z, "k h w -> h w", "median")
        normals_z = torch.mean(normals_z, dim=0)

        # Handle NaN values
        normals_x = torch.nan_to_num(normals_x, nan=0)
        normals_y = torch.nan_to_num(normals_y, nan=0)
        normals_z = torch.nan_to_num(normals_z, nan=-1)

        normals = torch.stack([normals_x, normals_y, normals_z])
        normals = e.rearrange(normals, "c h w -> h w c")
        normals = torch.nn.functional.normalize(normals, dim=2)

        # The D from Ax + By + Cz + D = 0
        # D_plane_shift_constant = -1
        # normals_z = D_plane_shift_constant * (x_part + y_part) / points_3d[..., 2]
        # normals = torch.stack([normals_x, normals_y, normals_z])
        # normals = e.rearrange(normals, "c h w -> h w c")
        # normals = torch.nn.functional.normalize(normals, dim=2)
        return normals

    def training_step(self, batch, batch_idx):
        inputs, target = batch
        output = self(inputs, target)
        return torch.zeros(1)
