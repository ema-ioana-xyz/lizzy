import lightning as L
import torch
from torch import Tensor
from jaxtyping import Float, jaxtyped
from typeguard import typechecked as typechecker
import einops as e

from metrics.normal_metrics import RMS_error, mean_of_values_under_threshold, sn_angle_error
from utils.camera_intrinsics import CameraIntrinsics
from utils.image_shape import ImageShape


class PlaneFitter_module(L.LightningModule):
    def __init__(self, camera_intrinsics: CameraIntrinsics, kernel_size: int):
        super().__init__()
        self.camera_intrinsics = camera_intrinsics
        self.kernel_size = kernel_size

    @jaxtyped(typechecker=typechecker)
    def forward(self, depth: Float[Tensor, "h w"]) -> Float[Tensor, "h w c=3"]:
        # Must be an odd integer
        KERNEL_SIZE = self.kernel_size

        input_shape = ImageShape(height=depth.shape[0], width=depth.shape[1], channels=3)
        intrinsics_derived_grid = self.camera_intrinsics.make_grid(input_shape).cuda()

        depth = depth.cuda()
        device = depth.device
        depth = e.rearrange(depth, "h w -> h w 1")
        points = intrinsics_derived_grid * depth
        points = e.rearrange(points, "h w c -> 1 c h w")

        k = KERNEL_SIZE
        kernel_padding = (k - 1) // 2
        image_shape = points.shape[2:]
        points.to(torch.float)
        patches = torch.nn.functional.unfold(points, kernel_size=k, padding=kernel_padding)
        # At this point, patches is [1 batch X k^2 * 3 channels X Many kernels]

        # Lay the coordinates in k^2 by 3 matrices
        patches = patches.unflatten(dim=1, sizes=[3, k**2])
        # After unflattening, patches is [1batch X 3channels X k^2 kernel_elements X Many kernels]

        patches = patches.permute([0, 3, 2, 1])
        # After permuting, patches is [1batch X Manykernels X k^2kernel_elements X 3channels]

        # Filter out invalid patches (those where the depth channel is zero for at least N-2 rows),
        # and replace them with identity matrices.
        # This works because each coordinate of a point is a linear function of the depth, so if the
        # depth is zero, then the entire row will be made of zeroes.
        nonzero_rows = torch.count_nonzero(patches[..., 2], dim=-1).unsqueeze_(-1).unsqueeze_(-1)
        identity_matrix = torch.eye(n=k**2, m=3, device=device)
        patches = torch.where(nonzero_rows >= min(3 + k, k**2), patches, identity_matrix).to(device)


        A = patches
        B = torch.ones([1, 1, k**2, 1], dtype=torch.float, device=device)

        normals = torch.linalg.lstsq(A, B).solution
        del A, B
        # normals is [1 batch X Many kernels X 3 solution_components X 1 problem_solved]

        normals.squeeze_(dim=-1)
        # normals is [1 batch X Many kernels X 3 solution_components]

        vector_norms = torch.linalg.vector_norm(normals, dim=2)
        vector_norms.unsqueeze_(-1)
        normals /= vector_norms

        # Mark out the invalid normals obtained from the identity matrices
        identity_norm = torch.sqrt(torch.tensor(3, dtype=torch.float))
        normals = torch.where(torch.isclose(vector_norms, identity_norm), 0.0, normals)

        normals = normals.unflatten(dim=1, sizes=image_shape).squeeze()
        return normals

    def training_step(self, batch, batch_idx):
        inputs, target = batch
        output = self(inputs, target)
        return torch.zeros(1)
    
    def test_step(self, batch):
        depth = batch["depth"].squeeze()
        normals_mask = batch["normals_mask"].squeeze()
        normals_gt = batch["normals"].squeeze()

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
