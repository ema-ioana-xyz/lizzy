from torch.utils.data import Dataset


class CustomDataset(Dataset):

    def __init__(self) -> None:
        data_path = "/home/eit/mnt/ema-ioana-tit/geonet_nyu/GeoNet/gt/test_split.txt"

        with open(data_path) as file:
            self.image_paths = [line.rstrip() for line in file.readlines()]

    def __getitem__(self, index: int):
        """Get the data for the item at a given index in the data list.

        Derived classes must implement `self.get_dataset_item(index: int)`.
        """
        return self.get_dataset_item(index)

    def __len__(self) -> int:
        """Get the number of images in the dataset."""
        return len(self.image_paths)
