import scipy.io
import torch
from torchvision.transforms.functional import to_tensor
from pathlib import Path
import numpy as np
import einops as e

from data.dataset import CustomDataset


class NyuDataSet(CustomDataset):
    def __init__(self) -> None:
        super().__init__()


    def get_dataset_item(self, index: int) -> dict:
        """Get the data at the specified index.

        Args:
            index: The index of the data to retrieve.

        Returns:
            A dict containing an image and its associated depth, normals, 
            depth and normals masks, and intrinsics matrix.
        """
        filename = f"{self.image_paths[index]}.mat"
        file_path = Path("/home/eit/mnt/ema-ioana-tit/geonet_nyu/GeoNet/test_data") / filename

        # Load the matlab image as a dictionary
        matlab_data = scipy.io.loadmat(file_path)

        # NYU image & normals data shape is [Height: 481 X Width: 641 X Channels: 3]
        # NYU depth & mask data shape is [Height: 481 X Width: 641]

        # Extract the data from the MATLAB file
        # ~~and crop extra border pixel~~
        image = matlab_data['img']
        depth = e.rearrange(matlab_data['depth'], "w h -> h w")
        normals = e.rearrange(matlab_data['norm'], "c w h -> h w c")
        depth_mask = e.rearrange(matlab_data['mask'], "w h -> h w")

        # image += np.array([122.175, 116.169, 103.508]) * 2
        # image = image.astype('uint8')

        # Return data as Tensors following Pytorch dimension order conventions.
        # Images are [Channel X Height X Width] in Torchvision
        image = to_tensor(image)
        image = e.rearrange(image, "c h w -> h w c")

        depth_mask = torch.from_numpy(depth_mask)
        normals_mask = depth_mask.unsqueeze(-1).expand(-1, -1, 3)  # [Height, Width, 3]

        # Channels dimensions with size 1 should be kept
        # depth_mask = depth_mask.unsqueeze(0)  # [1, Height, Width]
        # depth = torch.from_numpy(depth).unsqueeze(0)
        normals = torch.from_numpy(normals)

        data_dict = {
            "image": image,
            "normals": normals,
            "depth": depth,
            "depth_mask": depth_mask,
            "normals_mask": normals_mask,
        }

        return data_dict
