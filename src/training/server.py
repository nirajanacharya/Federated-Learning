import copy
import torch


def fedavg(client_updates):
    total_samples = sum(n for _, n in client_updates)
    avg_weights   = None

    for weights, n in client_updates:
        weight_factor = n / total_samples
        if avg_weights is None:
            avg_weights = {k: v.clone().float() * weight_factor for k, v in weights.items()}
        else:
            for k in avg_weights:
                avg_weights[k] += weights[k].clone().float() * weight_factor

    return avg_weights
