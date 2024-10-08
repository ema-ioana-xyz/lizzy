import typing

from matplotlib.figure import Figure
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
from matplotlib.axes import Axes
import torch
from torch import Tensor
from torchvision.io import read_image, read_video, write_video
from torchvision.transforms.v2.functional import resize, to_tensor
from torchvision.transforms.v2 import InterpolationMode
from scipy.io import loadmat
import einops as e
from pathlib import Path
import PIL.Image as pil
import numpy as np
import gradio as gr
from jaxtyping import Float, Int, UInt8, jaxtyped
from typeguard import typechecked as typechecker


@jaxtyped(typechecker=typechecker)
def visualize_depth(depth: Float[np.ndarray, "h w"]) -> Figure:
    figure = plt.figure(figsize=(12, 4))
    # normalizer = matplotlib.colors.Normalize(
    # vmin=depth.min(), vmax=np.percentile(depth, 95)
    # )
    # mapper = cm.ScalarMappable(norm=normalizer, cmap="magma")
    # colormapped_im = (mapper.to_rgba(depth) * 255).astype(np.uint8)
    max_depth = np.percentile(depth, 95).astype(float)
    img = plt.imshow(depth, cmap="turbo", vmax=max_depth)
    plt.colorbar(img)
    figure.tight_layout()
    return figure


@jaxtyped(typechecker=typechecker)
def visualize_normals_channelwise(normals: Float[np.ndarray, "h w c=3"]) -> Figure:
    cmap = matplotlib.colormaps["bwr"]
    cmap.set_bad(color="black")
    figure, axs = plt.subplots(nrows=3, ncols=1, figsize=(10, 10))
    axs = typing.cast(list[Axes], axs)
    for axis in range(3):
        plt.sca(axs[axis])
        axs[axis].set_aspect("equal")
        img = plt.imshow(normals[..., axis], cmap=cmap, vmin=-1, vmax=1)
        plt.colorbar(img)
    figure.tight_layout()
    return figure


@jaxtyped(typechecker=typechecker)
def visualize_normals_combined(normals: Float[np.ndarray, "h w c=3"]) -> Figure:
    figure = plt.figure()
    normals = (1 - normals) / 2
    plt.imshow(normals)
    return figure


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

    return depth


@jaxtyped(typechecker=typechecker)
def load_tftn_depth(file_path: Path) -> Float[Tensor, "h=480 w=640"]:
    with open(file_path, mode="rb") as file:
        data = np.fromfile(file, dtype=np.float32)
    data = data.reshape(480, 640)
    return torch.from_numpy(data)


file_input = gr.FileExplorer(label="Input File", file_count="single", root_dir="C:/")
model_selector = gr.Dropdown(
    label="Model",
    choices=[
        "Three-Filters-To-Normal",
        "Three-Filters-To-Normal+",
        "Aleatoric Uncertainty",
        "Simple Plane Fitting",
    ],
)

x_plot = gr.Plot(label="X Axis")
y_plot = gr.Plot(label="Y Axis")
z_plot = gr.Plot(label="Z Axis")
depth_plot = gr.Plot(label="Depth")
image_plot = gr.Plot(label="Image")

nyu_shape = ImageShape(height=481, width=641, channels=3)
img_shape = ImageShape(height=375, width=1242, channels=3)
tftn_shape = ImageShape(height=480, width=640, channels=3)


intrinsics = Manydepth_Intrinsics()
manydepth = Manydepth_module(
Manydepth_Intrinsics(), Path("/home/eit/lizzy/manydepth_weights_MR/KITTI_MR")
)
TFTN = TFTN_module(camera_intrinsics=intrinsics, kernel_size=7, kernel_type="prewitt")
PlaneFitter = PlaneFitter_module(camera_intrinsics=intrinsics, kernel_size=11)
ALUN = ALUN_module()


def run_image_prediction(image_np: Int[np.ndarray, "h w c=3"], progress=gr.Progress()):
    """Run prediction pipeline on a single image"""
    # Bring image into [0, 1] float range
    image = to_tensor(image_np)
    image = e.rearrange(image, "c h w -> h w c")

    with torch.no_grad():
        depth_image = manydepth.forward(input_frame=image)
        progress(1/4)

        tftn_normals = TFTN(depth_image).cpu().numpy()
        progress(2/4)

        planefitter_normals = PlaneFitter(depth_image).cpu().numpy()
        progress(3/4)

        alun_normals = ALUN(image).cpu().numpy()
        progress(1 - 1/8)

    tftn_figure = visualize_normals_channelwise(tftn_normals)
    planefitter_figure = visualize_normals_channelwise(planefitter_normals)
    alun_figure = visualize_normals_channelwise(alun_normals)

    depth_figure = visualize_depth(depth_image.cpu().numpy())
    progress(1.0)

    return depth_figure, tftn_figure, planefitter_figure, alun_figure


def run_video_prediction(video_path: str, frame_skip, output_fps, progress=gr.Progress()) -> tuple[str, str, str, str]:
    rgb_frames, _, video_info = read_video(video_path, output_format="THWC")

    depth_path = "/home/eit/video/depth.mp4"
    alun_path = "/home/eit/video/alun.mp4"
    tftn_path = "/home/eit/video/tftn.mp4"
    planefitter_path = "/home/eit/video/pf.mp4"

    with torch.no_grad():
        depth_frames = single_video_prediction(
            model=manydepth,
            frames=rgb_frames,
            plot_fn=plot_depth_as_tensor,
            frame_skip=frame_skip,
            output_fps=output_fps,
            output_path=depth_path,
            progress_logger=progress,
            progress_start=0.0,
            progress_stop=1/4,
            progress_description="ManyDepth depth prediction is running..."
        )
        single_video_prediction(
            model=ALUN,
            frames=rgb_frames,
            plot_fn=plot_normals_as_tensor,
            frame_skip=frame_skip,
            output_fps=output_fps,
            output_path=alun_path,
            progress_logger=progress,
            progress_start=1/4,
            progress_stop=2/4,
            progress_description="Aleatoric Uncertainty is running..."
        )

        depth_sequence_prediction(
            model=TFTN,
            frames=depth_frames,
            output_fps=output_fps,
            output_path=tftn_path,
            progress_logger=progress,
            progress_start=2/4,
            progress_stop=3/4,
            progress_description="Three Filters to Normal is running..."
        )
        depth_sequence_prediction(
            model=PlaneFitter,
            frames=depth_frames,
            output_fps=output_fps,
            output_path=planefitter_path,
            progress_logger=progress,
            progress_start=3/4,
            progress_stop=4/4,
            progress_description="PlaneFitter is running..."
        )
        # depth_image = manydepth.forward(input_frame=frame)
        # tftn_normals = TFTN(depth_image).cpu().numpy()
        # planefitter_normals = PlaneFitter(depth_image).cpu().numpy()
        # alun_normals = ALUN(frame).cpu().numpy()

    return depth_path, tftn_path, planefitter_path, alun_path


@jaxtyped(typechecker=typechecker)
def plot_depth_as_tensor(depth: Float[np.ndarray, "h w"]) -> UInt8[Tensor, "h w c=3"]:
    depth_cm = plt.get_cmap("turbo")
    depth = depth_cm(depth / np.quantile(depth, 0.85))
    depth = depth[..., 0:3]
    return torch.from_numpy(depth * 255).to(dtype=torch.uint8)


@jaxtyped(typechecker=typechecker)
def plot_normals_as_tensor(normals: Float[np.ndarray, "h w c=3"]) -> UInt8[Tensor, "h_cat w c=3"]:
    normal_cm = plt.get_cmap("bwr")
    normals_x = torch.from_numpy(normal_cm(normals[..., 0]))
    normals_y = torch.from_numpy(normal_cm(normals[..., 1]))
    normals_z = torch.from_numpy(normal_cm(normals[..., 2]))
    blank_line = torch.zeros([10, normals_x.shape[1], 4])
    stacked = torch.vstack([normals_x, blank_line, normals_y, blank_line, normals_z])

    # Remove alpha channel
    return (stacked[..., :3] * 255).to(dtype=torch.uint8)


def single_video_prediction(
    model,
    frames: Float[Tensor, "t c h w"],
    plot_fn,
    frame_skip: int,
    output_fps: float,
    output_path: str | Path,
    progress_logger: gr.Progress,
    progress_start: float,
    progress_stop: float,
    progress_description: str,
) -> list[Tensor]:
    plots = []
    predictions = []
    for i, frame in enumerate(frames):
        if i % frame_skip != 0:
            continue

        # Rescale values into [0, 1.0] interval
        frame = to_tensor(frame.numpy())
        frame = e.rearrange(frame, "c h w -> h w c")
        pred = model(frame)
        pred = pred.cpu()
        predictions.append(pred)

        pred = pred.numpy()
        plots.append(plot_fn(pred))

        progress_val = progress_start + (progress_stop - progress_start) / (len(frames) - 1) * i
        progress_logger(progress_val, desc=progress_description)
    plots = torch.stack(plots, dim=0)
    write_video(output_path, plots, fps=output_fps, video_codec="h264")
    return(predictions)


def depth_sequence_prediction(
    model,
    frames: list[Float[Tensor, "h w"]],
    output_fps: float,
    output_path: str | Path,
    progress_logger: gr.Progress,
    progress_start: float,
    progress_stop: float,
    progress_description: str,
) -> None:
    plots = []
    for i, frame in enumerate(frames):
        pred = model(frame)
        pred = pred.cpu()

        pred = pred.numpy()
        plots.append(plot_normals_as_tensor(pred))

        progress_val = progress_start + (progress_stop - progress_start) / (len(frames) - 1) * i
        progress_logger(progress_val, desc=progress_description)
    plots = torch.stack(plots, dim=0)
    write_video(output_path, plots, fps=output_fps, video_codec="h264")


with gr.Blocks(analytics_enabled=False) as demo:
    with gr.Tab("Single image input"):
        gr.Markdown("First run initializes some stuff. Subsequent runs should be faster")
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(label="Image Input", type="numpy")
                submit_image = gr.Button(value="Run prediction")
            with gr.Column():
                depth_output = gr.Plot(label="Predicted depth")
                tftn_sn_output = gr.Plot(label="Three Filters to Normal SN")
                planefitter_sn_output = gr.Plot(label="PlaneFitter SN")
                alun_sn_output = gr.Plot(label="Aleatoric Uncertainty SN")
        submit_image.click(
            fn=run_image_prediction,
            inputs=image_input,
            outputs=[
                depth_output,
                tftn_sn_output,
                planefitter_sn_output,
                alun_sn_output,
            ],
        )
    with gr.Tab("Video input"):
        gr.Markdown("First run initializes some stuff. Subsequent runs should be faster")
        with gr.Row():
            with gr.Column():
                video_input = gr.Video(label="Video Input")
                frame_skip_selector = gr.Number(
                    label="Frame skipping",
                    info="Only process every `N`th frame to save execution time. This sets `N`.",
                    value=10,
                    minimum=1,
                    maximum=500,
                    step=1,
                    precision=0,
                    )
                fps_selector = gr.Number(
                    label="Output framerate",
                    value=5,
                    minimum=0.5,
                    maximum=60,
                    step=0.1,
                    precision=1,
                    )
                submit_video = gr.Button(value="Run prediction")
            with gr.Column():
                depth_output = gr.Video(label="Predicted depth")
                tftn_sn_output = gr.Video(label="Three Filters to Normal SN")
                planefitter_sn_output = gr.Video(label="PlaneFitter SN")
                alun_sn_output = gr.Video(label="Aleatoric Uncertainty SN")
            submit_video.click(
                fn=run_video_prediction,
                inputs=[video_input, frame_skip_selector, fps_selector],
                outputs=[
                    depth_output,
                    tftn_sn_output,
                    planefitter_sn_output,
                    alun_sn_output,
                ],
            )


if __name__ == "__main__":
    demo.launch()
