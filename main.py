from modules.tftn_module import TFTN_module
from utils.camera_intrinsics import NYU_Intrinsics
from utils.image_shape import ImageShape
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib
import torch
import einops as e
from pathlib import Path
from torchvision.io import read_image

import gradio as gr

file_input = gr.FileExplorer(label="Input File", file_count="single")
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


def predict_from_matfile(model: str, file_path: str):
    nyu_shape = ImageShape(height=481, width=641, channels=3)
    module = TFTN_module(camera_intrinsics=NYU_Intrinsics(), input_shape=nyu_shape)

    if Path(file_path).suffix == ".mat":
        data = loadmat(
            # R"C:\work\helvetica_neue\lizzy\office_kitchen_0003_r-1315419135.670169-693301627.mat"
            file_path
        )
        depth = torch.from_numpy(data["depth"]).cuda()
        depth = e.rearrange(depth, "h w -> 1 h w")
        # normals_gt = data["norm"]
        # mask = data["mask"]
    else: # Image file
        image = read_image(file_path)
        depth_pred = 

    normals_pred = module(depth)

    normals_pred = normals_pred.cpu().numpy()
    cmap = matplotlib.colormaps["bwr"]
    cmap.set_bad(color="black")
    fig_x = plt.figure()
    plt.imshow(normals_pred[..., 0], cmap=cmap)
    fig_y = plt.figure()
    plt.imshow(normals_pred[..., 1], cmap=cmap)
    fig_z = plt.figure()
    plt.imshow(normals_pred[..., 2], cmap=cmap)

    return fig_x, fig_y, fig_z


demo = gr.Interface(
    fn=predict_from_matfile,
    inputs=[model_selector, file_input],
    outputs=[x_plot, y_plot, z_plot],
)

demo.launch()
