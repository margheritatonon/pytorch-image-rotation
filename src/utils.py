import torch
import torch.nn as nn

#creating an angular loss class
class AngularLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, predictions, targets):
        # predicted and target are [batch_size, 2] → sincos format
        pred_angles = torch.atan2(predictions[:, 0], predictions[:, 1])
        true_angles = torch.atan2(targets[:, 0], targets[:, 1])
        return (1 - torch.cos(pred_angles - true_angles)).mean()