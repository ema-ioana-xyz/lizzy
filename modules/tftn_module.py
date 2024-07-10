import lightning as L
import torch
from torch import Tensor, conv2d
from torch.nn.functional import unfold

from metrics.normal_metrics import L1_relative_error, RMS_error, RMS_log_error, delta_error
from utils.camera_intrinsics import CameraIntrinsics
from utils.image_shape import ImageShape
from jaxtyping import Float, jaxtyped
from typeguard import typechecked as typechecker
import einops as e


class TFTN_module(L.LightningModule):
    def __init__(self, camera_intrinsics: CameraIntrinsics):
        super().__init__()
        self.camera_intrinsics = camera_intrinsics

    @jaxtyped(typechecker=typechecker)
    def forward(self, depth: Float[Tensor, "h w"]) -> Float[Tensor, "h w 3"]:
        input_shape = ImageShape(height=depth.shape[0], width=depth.shape[1], channels=3)

        intrinsics_derived_grid = self.camera_intrinsics.make_grid(input_shape).cuda()

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
        kernel_x = e.rearrange(kernel_x.fliplr(), "h w -> 1 1 h w")

        kernel_y = e.rearrange(kernel_x.fliplr(), "1 1 h w -> 1 1 w h")
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

        normals_x = directional_derivative_x * FOCAL_POINT_X * input_shape.width
        normals_y = directional_derivative_y * FOCAL_POINT_Y * input_shape.height

        # Normals Z
        normals_z_volume = (
            torch.zeros_like(depth).unsqueeze(-1).expand([-1, -1, 8]).clone()
        )

        depth = e.rearrange(depth, "h w -> h w 1")
        points_3d = intrinsics_derived_grid * depth
        points_3d = points_3d.unsqueeze(0)

        for position in range(8):
            x_delta, y_delta, z_delta = self.compute_delta(
                points_3d[..., 0], points_3d[..., 1], points_3d[..., 2], position + 1
            )

            normals_z_volume[..., position] = (
                -(normals_x * x_delta + normals_y * y_delta) / z_delta
            )

        normals_z = torch.nanmedian(normals_z_volume, dim=-1).values

        # Handle NaN values
        normals_x = torch.nan_to_num(normals_x, nan=0)
        normals_y = torch.nan_to_num(normals_y, nan=0)
        normals_z = torch.nan_to_num(normals_z, nan=-1)

        normals = torch.stack([normals_x, normals_y, normals_z])
        normals = e.rearrange(normals, "c h w -> h w c")
        normals = torch.nn.functional.normalize(normals, dim=2)

        # Flip vector so that it is in the same coordinate space as the other
        # methods
        normals = normals * -1

        # The D from Ax + By + Cz + D = 0
        # D_plane_shift_constant = -1
        # normals_z = D_plane_shift_constant * (x_part + y_part) / points_3d[..., 2]
        # normals = torch.stack([normals_x, normals_y, normals_z])
        # normals = e.rearrange(normals, "c h w -> h w c")
        # normals = torch.nn.functional.normalize(normals, dim=2)
        return normals

    def compute_delta(self, X, Y, Z, position):
        if position == 1:
            kernel = [[0, -1, 0], [0, 1, 0], [0, 0, 0]]
        elif position == 2:
            kernel = [[0, 0, 0], [-1, 1, 0], [0, 0, 0]]
        elif position == 3:
            kernel = [[0, 0, 0], [0, 1, -1], [0, 0, 0]]
        elif position == 4:
            kernel = [[0, 0, 0], [0, 1, 0], [0, -1, 0]]
        elif position == 5:
            kernel = [[-1, 0, 0], [0, 1, 0], [0, 0, 0]]
        elif position == 6:
            kernel = [[0, 0, 0], [0, 1, 0], [-1, 0, 0]]
        elif position == 7:
            kernel = [[0, 0, -1], [0, 1, 0], [0, 0, 0]]
        else:
            kernel = [[0, 0, 0], [0, 1, 0], [0, 0, -1]]

        kernel = (
            torch.tensor(kernel, device="cuda", dtype=torch.float)
            .fliplr()
            .unsqueeze(0)
            .unsqueeze(0)
        )
        X_d = torch.conv2d(X, kernel, padding="same")
        Y_d = torch.conv2d(Y, kernel, padding="same")
        Z_d = torch.conv2d(Z, kernel, padding="same")

        mask = Z_d == 0
        X_d = torch.where(mask, torch.nan, X_d)
        Y_d = torch.where(mask, torch.nan, Y_d)
        Z_d = torch.where(mask, torch.nan, Z_d)

        return X_d, Y_d, Z_d

    def training_step(self, batch, batch_idx):
        inputs, target = batch
        output = self(inputs, target)
        return torch.zeros(1)

    def test_step(self, batch):
        depth = batch["depth"].squeeze(0)
        normals_mask = batch["normals_mask"].squeeze(0)
        normals_gt = batch["normals"].squeeze(0)

        normals = self.forward(depth)

        self.log("RMS error", RMS_error(normals, normals_gt, normals_mask))
        self.log("RMS log error", RMS_log_error(normals, normals_gt, normals_mask))
        self.log("L1 relative error", L1_relative_error(normals, normals_gt, normals_mask))
        self.log("Ang err < 11.25", delta_error(normals, normals_gt, 11.25, normals_mask))
        self.log("Ang err < 22.5", delta_error(normals, normals_gt, 22.5, normals_mask))
        self.log("Ang err < 30", delta_error(normals, normals_gt, 30, normals_mask))
        self.log("Ang err < 40", delta_error(normals, normals_gt, 45, normals_mask))
