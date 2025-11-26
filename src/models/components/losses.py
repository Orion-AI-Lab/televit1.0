import torch


class WeightedMSELoss(torch.nn.Module):
    def __init__(self):
        super(WeightedMSELoss, self).__init__()

    def forward(self, predictions, targets, weights = None, mask=None):
        if mask is not None:
            predictions = predictions * mask
            targets = targets * mask
        loss = (predictions - targets) ** 2
        if weights is not None:
            if len(loss.shape) - len(weights.shape) == 1:
                weights = weights.unsqueeze(-1)
            elif abs(len(loss.shape) - len(weights.shape)) > 1:
                raise ValueError("weights must have the same number of dimensions as loss or one less to be broadcastable")
            loss *= weights
        return loss.mean()