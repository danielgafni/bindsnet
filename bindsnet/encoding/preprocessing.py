import torch


def to_positive(inpts: torch.Tensor, dim=2):
    """
    Converts a mixture of negative and positive inputs to positive only.
    Separates positive and negative values and packs them along a new dimension.
    
    """
    positive_part = torch.clamp(inpts, min=0)
    negative_part = torch.clamp(inpts, max=0)

    return torch.stack((positive_part, torch.abs(negative_part)), dim=dim)
