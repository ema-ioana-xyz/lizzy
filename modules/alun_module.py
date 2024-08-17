import lightning as L
import torch
from torchvision import transforms
from torch import Tensor
import einops as e
from jaxtyping import Float, jaxtyped
from typeguard import typechecked as typechecker

from alun.models.NNET import NNET
from alun.utils.arguments import AlunArguments
import alun.utils.utils as alun_utils
from metrics.normal_metrics import RMS_error, sn_angle_error, mean_of_values_under_threshold


class ALUN_module(L.LightningModule):
    def __init__(
        self,
    ):
        super().__init__()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        checkpoint_path = "./alun_weights/scannet.pt"
        model = NNET(AlunArguments()).to(device)
        model = alun_utils.load_checkpoint(checkpoint_path, model)
        model.eval()
        self.model = model


    @jaxtyped(typechecker=typechecker)
    def forward(
        self, img: Float[Tensor, "hin win c=3"]
    ) -> Float[Tensor, "hout wout c=3"]:
        img = e.rearrange(img, "h w c -> c h w")
        img = self.normalize(img)
        img = img.unsqueeze(0).cuda()

        norm_out_list, _, _ = self.model(img)
        norm_out = norm_out_list[-1]

        pred_norm = norm_out[:, :3, :, :]
        normals = e.rearrange(pred_norm, "1 c h w -> h w c")
        return normals

    def test_step(self, batch):
        img = batch["image"].squeeze()
        normals_mask = batch["normals_mask"].squeeze()
        normals_gt = batch["normals"].squeeze()

        normals = self.forward(img)

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
