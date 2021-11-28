import torch

from lib.models.base_model import BaseModel


class Regression(BaseModel):

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, batch):
        return self.net(batch[0].to(self.device))

    def loss(self, pred, batch, reduce=True):
        y = batch[1].to(self.device).float()
        N = y.shape[0]
        y = y.reshape(N, -1)
        y_pred = pred.y_pred.reshape(N, -1)
        loss = torch.nn.functional.mse_loss(y_pred, y, reduction="none").sum(
            dim=1)

        mae = torch.abs(y_pred - y).sum(dim=1)

        if reduce:
            mae = mae.mean()
            loss = loss.mean()

        return loss, {"mse": loss, "mae": mae}
