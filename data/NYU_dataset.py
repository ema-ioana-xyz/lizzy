import scipy.io
import torch
from torchvision.transforms.functional import to_tensor
from pathlib import Path
import numpy as np

from data.dataset import CustomDataset, DataKey


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
        filename = self.image_paths[index]
        file_path = Path("../lizzy_data/test_data") / filename

        # Load the matlab image as a dictionary
        matlab_data = scipy.io.loadmat(file_path)

        # NYU image & normals data shape is [Height: 481 X Width: 641 X Channels: 3]
        # NYU depth & mask data shape is [Height: 481 X Width: 641]

        # Extract the data from the MATLAB file
        # and crop extra border pixel
        image = matlab_data['img'][0:-1, 0:-1]
        depth = matlab_data['depth'][0:-1, 0:-1]
        normals = matlab_data['norm'][0:-1, 0:-1]
        depth_mask = matlab_data['mask'][0:-1, 0:-1]

        image += np.array([122.175, 116.169, 103.508]) * 2
        image = image.astype('uint8')

        # Return data as Tensors following Pytorch dimension order conventions.
        # Images are [Channel X Height X Width] in Torchvision
        image = to_tensor(image)

        depth_mask = torch.from_numpy(depth_mask)
        normals_mask = depth_mask.unsqueeze(-1).expand(-1, -1, 3)  # [Height, Width, 3]

        # Channels dimensions with size 1 should be kept
        depth_mask = depth_mask.unsqueeze(0)  # [1, Height, Width]
        depth = torch.from_numpy(depth).unsqueeze(0)
        normals = torch.from_numpy(normals)

        data_dict = {
            "image": image,
            "normals": normals,
            "depth": depth,
            "depth_mask": depth_mask,
            "normals_mask": normals_mask,
        }

        return data_dict
