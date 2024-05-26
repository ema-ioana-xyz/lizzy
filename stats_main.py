from data.NYU_data_module import NyuDataModule
from modules.plane_fitter import PlaneFitter_module
from modules.tftn_module import TFTN_module
from modules.plane_fitter import PlaneFitter_module
from modules.alun_module import ALUN_module
from modules.manydepth import Manydepth_module
from utils.camera_intrinsics import (
    Manydepth_Intrinsics,
    NYU_Intrinsics,
    TFTN_dataset_intrinsics,
)
from utils.image_shape import ImageShape

import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import torch
from torch import Tensor
from torchvision.io import read_image
from torchvision.transforms.functional import to_tensor
import lightning as L
from scipy.io import loadmat
import einops as e
from pathlib import Path
import PIL.Image as pil
import numpy as np
import gradio as gr
from jaxtyping import Float, jaxtyped
from typeguard import typechecked as typechecker


@jaxtyped(typechecker=typechecker)
def visualize_depth(depth: Float[np.ndarray, "h w"]) -> None:
    fig = plt.figure()
    # normalizer = matplotlib.colors.Normalize(
    # vmin=depth.min(), vmax=np.percentile(depth, 95)
    # )
    # mapper = cm.ScalarMappable(norm=normalizer, cmap="magma")
    # colormapped_im = (mapper.to_rgba(depth) * 255).astype(np.uint8)
    img = plt.imshow(depth, cmap="inferno", vmax=np.percentile(depth, 95))
    plt.colorbar(img)


@jaxtyped(typechecker=typechecker)
def load_rgb_image(file_path: Path) -> Float[Tensor, "h w c=3"]:
    image = read_image(file_path.as_posix())
    image = e.rearrange(image, "c h w -> h w c")

    # Bring image into [0, 1] float range
    image = to_tensor(image.numpy())
    image = e.rearrange(image, "c h w -> h w c")

    return image


@jaxtyped(typechecker=typechecker)
def load_nyu_image(file_path: Path) -> Float[Tensor, "h=481 w=641 c=3"]:
    # Load
    image = loadmat(file_path)["img"]

    # Undo NYU (normalization?) preprocessing
    image += np.array([122.175, 116.169, 103.508]) * 2
    image = image.astype("uint8")

    # Bring image into [0, 1] float range
    image = to_tensor(image)
    image = e.rearrange(image, "c h w -> h w c")

    return image


@jaxtyped(typechecker=typechecker)
def load_nyu_depth(file_path: Path) -> Float[Tensor, "h=481 w=641"]:
    depth = loadmat(file_path)["depth"]
    depth = torch.from_numpy(depth)
    # depth = e.rearrange(depth, "h w -> h w")

    return depth


@jaxtyped(typechecker=typechecker)
def load_tftn_depth(file_path: Path) -> Float[Tensor, "h=480 w=640"]:
    with open(file_path, mode="rb") as file:
        data = np.fromfile(file, dtype=np.float32)
    data = data.reshape(480, 640)
    return torch.from_numpy(data)


@jaxtyped(typechecker=typechecker)
def visualize_normals(normals: Float[np.ndarray, "h w c=3"]) -> None:
    cmap = matplotlib.colormaps["bwr"]
    cmap.set_bad(color="black")
    # plt.figure()
    # normals = (1 - normals) / 2
    fig, axs = plt.subplots(nrows=3, ncols=1)
    for axis in range(3):
        plt.sca(axs[axis])
        axs[axis].set_aspect("equal")
        img = plt.imshow(normals[..., axis], cmap=cmap, vmin=-1, vmax=1)
        plt.colorbar(img)
    # plt.imshow(normals)


# file_input = gr.FileExplorer(label="Input File", file_count="single", root_dir="C:/")
# model_selector = gr.Dropdown(
#     label="Model",
#     choices=[
#         "Three-Filters-To-Normal",
#         "Three-Filters-To-Normal+",
#         "Aleatoric Uncertainty",
#         "Simple Plane Fitting",
#     ],
# )

# x_plot = gr.Plot(label="X Axis")
# y_plot = gr.Plot(label="Y Axis")
# z_plot = gr.Plot(label="Z Axis")
# depth_plot = gr.Plot(label="Depth")
# image_plot = gr.Plot(label="Image")

nyu_shape = ImageShape(height=481, width=641, channels=3)
img_shape = ImageShape(height=375, width=1242, channels=3)
tftn_shape = ImageShape(height=480, width=640, channels=3)


intrinsics = NYU_Intrinsics()
shape = tftn_shape
# manydepth = Manydepth_module(
# Manydepth_Intrinsics(), Path("./manydepth_weights_KITTI_MR")
# )
TFTN = TFTN_module(camera_intrinsics=intrinsics, input_shape=shape)
PlaneFitter = PlaneFitter_module(camera_intrinsics=intrinsics, input_shape=shape)
ALUN = ALUN_module()

trainer = L.Trainer()
test_dataset = NyuDataModule().test_dataloader(num_workers=0)

for model in [TFTN, ALUN, PlaneFitter]:
    trainer.test(model, dataloaders=test_dataset)
