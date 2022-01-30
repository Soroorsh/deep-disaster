"""
Losses
"""
# pylint: disable=C0301,C0103,R0902,R0915,W0221,W0622


##
import torch
import torch.nn as nn

##
def l1_loss(input, target):
    """ L1 Loss without reduce flag.

    Args:
        input (FloatTensor): Input tensor
        target (FloatTensor): Output tensor

    Returns:
        [FloatTensor]: L1 distance between input and output
    """

    return torch.mean(torch.abs(input - target))

##
def l2_loss(input, target, size_average=True):
    """ L2 Loss without reduce flag.

    Args:
        input (FloatTensor): Input tensor
        target (FloatTensor): Output tensor

    Returns:
        [FloatTensor]: L2 distance between input and output
    """
    if size_average:
        return torch.mean(torch.pow((input-target), 2))
    else:
        return torch.pow((input-target), 2)


class MseDirectionLoss(nn.Module):
    """ MseDirection Loss Class.
    """
    def __init__(self, lamda):
        super(MseDirectionLoss, self).__init__()
        self.lamda = lamda
        self.criterion = nn.MSELoss()
        self.similarity_loss = nn.CosineSimilarity()

    def forward(self, output_pred, output_real):
        """ MseDirection Loss Class.

        Args:
            output_pred (FloatTensor): predicted output
            output_real (FloatTensor): real output

        Returns:
            [FloatTensor]: the total loss which is a combination of Euclidean loss and Cosine Similarity
        """
        abs_loss = self.criterion(output_pred, output_real)
        loss = torch.mean(1 - self.similarity_loss(output_pred.view(output_pred.shape[0], -1), output_real.view(output_real.shape[0], -1)))
        total_loss = (self.lamda * abs_loss) + loss 
        return total_loss