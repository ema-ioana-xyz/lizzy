import lightning as L
import torch


class BaseModule(L.LightningModule):
    def test_step(self, batch):
        with torch.no_grad():
            self.forward(batch)
