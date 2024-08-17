import lightning as L
import torch
from torch import Tensor

from metrics.normal_metrics import RMS_error, mean_of_values_under_threshold, sn_angle_error
from utils.camera_intrinsics import CameraIntrinsics
from utils.image_shape import ImageShape
from jaxtyping import Float, jaxtyped
from typeguard import typechecked as typechecker
import einops as e


class TFTN_module(L.LightningModule):
    def __init__(self, camera_intrinsics: CameraIntrinsics, kernel_size: int = 3, kernel_type: str = "sobel"):
        super().__init__()
        self.camera_intrinsics = camera_intrinsics

        self.kernels = {}

        self.kernels[("sobel", 3)] = torch.tensor(
            [
                [-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1],
            ],
            device="cuda",
            dtype=torch.float,
        )

        self.kernels[("scharr", 3)] = torch.tensor(
            [
                [-3,  0,  3],
                [-10, 0, 10],
                [-3,  0,  3],
            ],
            device="cuda",
            dtype=torch.float,
        )

        self.kernels[("prewitt", 3)] = torch.tensor(
            [
                [-1, 0, 1],
                [-1, 0, 1],
                [-1, 0, 1],
            ],
            device="cuda",
            dtype=torch.float,
        )

        for size in [3, 5, 7]:
            fd_filter = torch.zeros((size, size), dtype=torch.float, device="cuda")
            middle = (size - 1) // 2
            for i in range(middle):
                fd_filter[middle, middle + i] = i
                fd_filter[middle, middle - i] = -i
            
            self.kernels[("fd", size)] = fd_filter

        sobel5 = """
            -1    -2     0     2     1
            -4    -8     0     8     4
            -6   -12     0    12     6
            -4    -8     0     8     4
            -1    -2     0     2     1
            """
        
        sobel7 = """
            -1    -4    -5     0     5     4     1
            -6   -24   -30     0    30    24     6
            -15   -60   -75     0    75    60    15
            -20   -80  -100     0   100    80    20
            -15   -60   -75     0    75    60    15
            -6   -24   -30     0    30    24     6
            -1    -4    -5     0     5     4     1
            """

        prewitt5 = """
            -1    -1     0     1     1
            -2    -2     0     2     2
            -3    -3     0     3     3
            -2    -2     0     2     2
            -1    -1     0     1     1"""
        
        prewitt7 = """
            -1    -2    -2     0     2     2     1
            -3    -6    -6     0     6     6     3
            -6   -12   -12     0    12    12     6
            -7   -14   -14     0    14    14     7
            -6   -12   -12     0    12    12     6
            -3    -6    -6     0     6     6     3
            -1    -2    -2     0     2     2     1"""
        
        
        scharr5 = """
        -27         -90           0          90          27
        -180        -600           0         600         180
        -354       -1180           0        1180         354
        -180        -600           0         600         180
         -27         -90           0          90          27"""
        
        scharr7 = """
        -243       -1620       -2943           0        2943        1620         243
       -2430      -16200      -29430           0       29430       16200        2430
       -8829      -58860     -106929           0      106929       58860        8829
      -13860      -92400     -167860           0      167860       92400       13860
       -8829      -58860     -106929           0      106929       58860        8829
       -2430      -16200      -29430           0       29430       16200        2430
        -243       -1620       -2943           0        2943        1620         243"""

        for s in [sobel5, sobel7]:
            lines = s.split("\n")
            lines = [line.split() for line in lines]
            lines = [line for line in lines if len(line) > 0]

            k = len(lines)

            for i in range(k):
                lines[i] = [int(x) for x in lines[i]]
            
            filter = torch.tensor(lines, dtype=torch.float, device="cuda")
            self.kernels[("sobel", k)] = filter

        for s in [prewitt5, prewitt7]:
            lines = s.split("\n")
            lines = [line.split() for line in lines]
            lines = [line for line in lines if len(line) > 0]

            k = len(lines)

            for i in range(k):
                lines[i] = [int(x) for x in lines[i]]
            
            filter = torch.tensor(lines, dtype=torch.float, device="cuda")
            self.kernels[("prewitt", k)] = filter

        for s in [scharr5, scharr7]:
            lines = s.split("\n")
            lines = [line.split() for line in lines]
            lines = [line for line in lines if len(line) > 0]

            k = len(lines)

            for i in range(k):
                lines[i] = [int(x) for x in lines[i]]
            
            filter = torch.tensor(lines, dtype=torch.float, device="cuda")
            self.kernels[("scharr", k)] = filter

        self.kernel_x = self.kernels[(kernel_type, kernel_size)]

       

    @jaxtyped(typechecker=typechecker)
    def forward(self, depth: Float[Tensor, "h w"]) -> Float[Tensor, "h w 3"]:
        input_shape = ImageShape(height=depth.shape[0], width=depth.shape[1], channels=3)

        intrinsics_derived_grid = self.camera_intrinsics.make_grid(input_shape).cuda()

        depth = depth.cuda()

        # Sobel filters for the first derivatives
        kernel_x = self.kernel_x
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

        angle_error_vector = sn_angle_error(normals, normals_gt, normals_mask)

        self.log("Mean angle error", angle_error_vector.mean())
        self.log("Median angle error", angle_error_vector.median())
        self.log("RMS error", RMS_error(normals, normals_gt, normals_mask))
        self.log("Ang err < 5", mean_of_values_under_threshold(angle_error_vector, 5))
        self.log("Ang err < 7.5", mean_of_values_under_threshold(angle_error_vector, 7.5))
        self.log("Ang err < 11.25", mean_of_values_under_threshold(angle_error_vector, 11.25))
        self.log("Ang err < 22.5", mean_of_values_under_threshold(angle_error_vector, 22.5))
        self.log("Ang err < 30", mean_of_values_under_threshold(angle_error_vector, 30))
        self.log("Ang err < 45", mean_of_values_under_threshold(angle_error_vector, 45))