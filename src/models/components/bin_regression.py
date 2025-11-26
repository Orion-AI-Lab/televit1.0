import torch
import torch.nn.functional as F

class cls_head(torch.nn.Module):
    def __init__(self, in_channels, bins):
        super(cls_head, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, bins, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    
class regression_head(torch.nn.Module):
    def __init__(self, in_channels):
        super(regression_head, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, 1, kernel_size=1)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x))


def get_log_bins(num_bins, min_val, max_val, device='cuda'):
    return torch.logspace(torch.log10(torch.tensor(min_val)), 
                          torch.log10(torch.tensor(max_val)), 
                          steps=num_bins).to(device)

def get_bin_centers(bins, agg='geom'):
    if agg == 'geom':
        return torch.sqrt(bins[:-1] * bins[1:])
    elif agg == 'mean':
        return (bins[:-1] + bins[1:]) / 2.0

def soft_bin_to_continuous(logits, bins):
    """ Converts logits (batch, bins, H, W) to a continuous regression map (batch, H, W). """
    probs = F.softmax(logits, dim=1)  # Convert logits to probabilities over bins

    bin_centers = get_bin_centers(bins)
    # Reshape bin_centers to match spatial dimensions
    bin_centers = bin_centers.view(1, -1, 1, 1)  # Shape: (1, bins, 1, 1)
    # Perform weighted sum across bin dimension (dim=1)
    return torch.sum(probs * bin_centers.to(probs.device), dim=1)  # Output: (batch, H, W)
