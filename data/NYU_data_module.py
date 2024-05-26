from torch.utils.data import DataLoader
import pytorch_lightning as pl
from data.NYU_dataset import NyuDataSet


class NyuDataModule(pl.LightningDataModule):
    """Data module which manages the data with PyTorchLightning framework."""

    def __init__(self):
        super().__init__()
        self.test = NyuDataSet()

    def test_dataloader(self, num_workers: int) -> DataLoader:
        """Creates and returns the test data loader.

        Returns:
            The test data loader.
        """
        return DataLoader(self.test, batch_size=1, num_workers=num_workers)
