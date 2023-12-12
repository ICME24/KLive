import torch
import torch.nn.functional as F
import torch.nn as nn

PADDED_Y_VALUE = -1
DEFAULT_EPS = 1e-10


def pairwise_loss(y_pred, y_true, weighing_scheme=None, k=None, sigma=1., mu=10.,
               reduction="mean", reduction_log="natural"):
    eps=DEFAULT_EPS
    padded_value_indicator=PADDED_Y_VALUE
    norm_loss = nn.L1Loss()
    device = y_pred.device
    y_pred = y_pred.clone()
    y_true = y_true.clone()

    padded_mask = y_true == padded_value_indicator
    y_pred[padded_mask] = float("-inf")
    y_true[padded_mask] = float("-inf")

    y_pred_sorted, indices_pred = y_pred.sort(descending=True, dim=-1)
    y_true_sorted, _ = y_true.sort(descending=True, dim=-1)

    true_sorted_by_preds = torch.gather(y_true, dim=1, index=indices_pred)
    true_diffs = true_sorted_by_preds[:, :, None] - true_sorted_by_preds[:, None, :]
    padded_pairs_mask = torch.isfinite(true_diffs)

    padded_pairs_mask = padded_pairs_mask & (true_diffs > 0)

    at_k_mask = torch.zeros((y_pred.shape[1], y_pred.shape[1]), dtype=torch.bool, device=device)
    at_k_mask[:k, :k] = 1

    true_sorted_by_preds.clamp_(min=0.)
    y_true_sorted.clamp_(min=0.)

    weights = 1.

    scores_diffs = (y_pred_sorted[:, :, None] - y_pred_sorted[:, None, :]).clamp(min=-1e3, max=1e3)
    first_order_differences = true_diffs - scores_diffs
    pos_bool = first_order_differences >= 0
    scores_diffs.masked_fill(torch.isnan(scores_diffs), 0.)
    weighted_probas1 = (torch.sigmoid(sigma * scores_diffs).clamp(min=eps) ** weights).clamp(min=eps)
    if reduction_log == "natural":
        losses1 = torch.log(weighted_probas1)
    elif reduction_log == "binary":
        losses1 = torch.log2(weighted_probas1)
    else:
        raise ValueError("Reduction logarithm base can be either natural or binary")

    if reduction == "sum":
        loss = -torch.sum(losses1[padded_pairs_mask & at_k_mask & pos_bool])
        loss[torch.isnan(loss)] = 0 
    elif reduction == "mean":
        loss = -torch.mean(losses1[padded_pairs_mask & at_k_mask & pos_bool ])
        loss[torch.isnan(loss)] = 0 
    else:
        raise ValueError("Reduction method can be either sum or mean")

    return loss