import lightning as L
import torch
from torchvision import transforms
from torchvision.transforms.functional import resize
from torchvision.transforms import InterpolationMode
from torch import Tensor
from pathlib import Path
import einops as e
from jaxtyping import Float, jaxtyped
from typeguard import typechecked as typechecker

from alun.models.NNET import NNET
from alun.utils.arguments import AlunArguments
import alun.utils.utils as alun_utils


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
        img = resize(img, [480, 640], interpolation=InterpolationMode.BICUBIC)
        img = self.normalize(img)
        img = img.unsqueeze(0).cuda()

        norm_out_list, _, _ = self.model(img)
        norm_out = norm_out_list[-1]

        pred_norm = norm_out[:, :3, :, :]
        normals = e.rearrange(pred_norm, "1 c h w -> h w c")
        return normals
