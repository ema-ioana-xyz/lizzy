from typing import Optional
import lightning as L
import torch
from torch import Tensor
from torchvision.transforms.functional import resize
from torchvision.transforms import InterpolationMode
from pathlib import Path
import einops as e
from jaxtyping import Float, jaxtyped
from typeguard import typechecked as typechecker


from utils.camera_intrinsics import CameraIntrinsics
from utils.image_shape import ImageShape
from manydepth import networks
from manydepth.layers import disp_to_depth, transformation_from_parameters


class Manydepth_module(L.LightningModule):
    def __init__(
        self,
        unscaled_intrinsics: CameraIntrinsics,
        model_path: Path,
    ):
        super().__init__()

        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        self.encoder_dict = torch.load(model_path / "encoder.pth", map_location=device)
        self.camera_intrinsics = unscaled_intrinsics
        self.input_shape = ImageShape(
            width=self.encoder_dict["width"],
            height=self.encoder_dict["height"],
            channels=3,
        )

        # Rescale intrinsics for Manydepth
        scale_factor_x = self.input_shape.width // 4
        scale_factor_y = self.input_shape.height // 4
        self.camera_intrinsics.focal_point_x *= scale_factor_x
        self.camera_intrinsics.focal_point_y *= scale_factor_y
        self.camera_intrinsics.principal_point_x *= scale_factor_x
        self.camera_intrinsics.principal_point_y *= scale_factor_y

        self.encoder = networks.ResnetEncoderMatching(
            18,
            False,
            input_width=self.encoder_dict["width"],
            input_height=self.encoder_dict["height"],
            adaptive_bins=True,
            min_depth_bin=self.encoder_dict["min_depth_bin"],
            max_depth_bin=self.encoder_dict["max_depth_bin"],
            depth_binning="linear",
            num_depth_bins=96,
        )

        filtered_dict_enc = {
            k: v for k, v in self.encoder_dict.items() if k in self.encoder.state_dict()
        }
        self.encoder.load_state_dict(filtered_dict_enc)

        self.depth_decoder = networks.DepthDecoder(
            num_ch_enc=self.encoder.num_ch_enc, scales=range(4)
        )

        loaded_dict = torch.load(model_path / "depth.pth", map_location=device)
        self.depth_decoder.load_state_dict(loaded_dict)

        pose_enc_dict = torch.load(model_path / "pose_encoder.pth", map_location=device)
        pose_dec_dict = torch.load(model_path / "pose.pth", map_location=device)

        self.pose_enc = networks.ResnetEncoder(18, False, num_input_images=2)
        self.pose_dec = networks.PoseDecoder(
            self.pose_enc.num_ch_enc, num_input_features=1, num_frames_to_predict_for=2
        )

        self.pose_enc.load_state_dict(pose_enc_dict, strict=True)
        self.pose_dec.load_state_dict(pose_dec_dict, strict=True)

        # Setting states of networks
        self.encoder.eval()
        self.depth_decoder.eval()
        self.pose_enc.eval()
        self.pose_dec.eval()

        if torch.cuda.is_available():
            self.encoder.cuda()
            self.depth_decoder.cuda()
            self.pose_enc.cuda()
            self.pose_dec.cuda()

    def __str__(self):
        return "Manydepth model string representation (default one is too long)"

    # @jaxtyped(typechecker=typechecker)
    def forward(
        self,
        input_frame: Float[Tensor, "h1 w1 c"],
        prev_frame: Optional[Float[Tensor, "h2 w2 c"]] = None,
        multiframe: bool = False,
    ) -> Float[Tensor, "h1 w1"]:
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        if (multiframe is True and prev_frame is None) or (
            multiframe is False and prev_frame is not None
        ):
            raise (ValueError("Frame input and multiframe variable are inconsistent"))

        if prev_frame is None:
            prev_frame = input_frame

        input_frame = input_frame.to(device)
        prev_frame = prev_frame.to(device)

        original_size = input_frame.shape[0:2]

        # Resize inputs
        input_frame = e.rearrange(input_frame, "h w c -> 1 c h w")
        prev_frame = e.rearrange(prev_frame, "h w c -> 1 c h w")
        input_frame = resize(
            input_frame,
            size=[self.input_shape.height, self.input_shape.width],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
        prev_frame = resize(
            prev_frame,
            size=[self.input_shape.height, self.input_shape.width],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )

        # Run inference
        pose_inputs = [prev_frame, input_frame]
        pose_inputs = [self.pose_enc(torch.cat(pose_inputs, 1))]
        axisangle, translation = self.pose_dec(pose_inputs)
        pose = transformation_from_parameters(
            axisangle[:, 0], translation[:, 0], invert=True
        )

        if not multiframe:
            pose *= 0  # zero poses are a signal to the encoder not to construct a cost volume
            prev_frame *= 0

        # Estimate depth
        K = self.camera_intrinsics.get_matrix()
        inv_K = torch.linalg.inv(K)
        K = e.rearrange(K, "w h -> 1 w h").cuda()
        inv_K = e.rearrange(inv_K, "w h -> 1 w h").cuda()

        output, lowest_cost, _ = self.encoder(
            current_image=input_frame,
            lookup_images=prev_frame.unsqueeze(1),
            poses=pose.unsqueeze(1),
            K=K,
            invK=inv_K,
            min_depth_bin=self.encoder_dict["min_depth_bin"],
            max_depth_bin=self.encoder_dict["max_depth_bin"],
        )

        output = self.depth_decoder(output)

        sigmoid_output = output[("disp", 0)]
        sigmoid_output_resized = torch.nn.functional.interpolate(
            sigmoid_output, original_size, mode="bilinear", align_corners=False
        )

        result = e.rearrange(sigmoid_output_resized, "1 1 h w -> h w")
        _, result = disp_to_depth(result, min_depth=0.1, max_depth=100.0)
        result = 5.4 * result
        gt_median = 12
        ratio = gt_median / result.median()
        result = result * ratio
        return result
