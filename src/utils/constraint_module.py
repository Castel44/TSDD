import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss


class CentroidLoss(nn.Module):
    def __init__(self, feat_dim, num_classes, reduction='mean'):
        super(CentroidLoss, self).__init__()
        self.classes = num_classes
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim), requires_grad=True)
        self.reduction = reduction
        self.rho = 1.0

    def forward(self, h, y):
        C = self.centers
        norm_squared = torch.sum((h.unsqueeze(1) - C) ** 2, 2)
        # Attractive
        distance = norm_squared.gather(1, y.unsqueeze(1)).squeeze()
        # Repulsive
        logsum = torch.logsumexp(-torch.sqrt(norm_squared), dim=1)
        loss = reduce_loss(distance + logsum, reduction=self.reduction)
        # Regularization
        if self.classes != 1:
            reg = self.regularization(reduction='sum')
            return loss + self.rho * reg
        else:
            return loss

    def regularization(self, reduction='sum'):
        C = self.centers
        pairwise_dist = torch.cdist(C, C, p=2) ** 2
        pairwise_dist = pairwise_dist.masked_fill(
            torch.zeros((C.size(0), C.size(0))).fill_diagonal_(1).bool().to(device), float('inf'))
        distance_reg = reduce_loss(-(torch.min(torch.log(pairwise_dist), dim=-1)[0]), reduction=reduction)
        return distance_reg
