import torch
import torch.nn.functional as F

def calculate_weights(labels):
    weight_positive = torch.zeros_like(labels)
    weight_positive[:, :] = 2
    weight_positive[:, 0] = 14
    weights = labels * weight_positive + (1 - labels) * weight_positive * 0.5
    return weights


def weighted_loss_with_logits(logits, labels):
    # logits N x 8
    # labels N x 8
    weights = calculate_weights(labels)
    loss = F.binary_cross_entropy_with_logits(logits, labels.to(torch.float), reduction='none')
    loss = weights * loss

    weights_sum = weights.sum(dim=1)
    overall_loss = loss[:, 0] / weights_sum
    c_loss = loss[:, 1:].sum(dim=1) / weights_sum

    return overall_loss.mean(), c_loss.mean()

def weighted_loss_with_probs(probs, labels):
    weights = calculate_weights(labels)
    loss = F.binary_cross_entropy(probs, labels.to(torch.float), reduction='none')

    loss = weights * loss

    weights_sum = weights.sum(dim=1)
    overall_loss = loss[:, 0] / weights_sum
    c_loss = loss[:, 1:].sum(dim=1) / weights_sum

    return overall_loss.mean(), c_loss.mean()