import numpy as np
import torch


def get_predictions(model, dataloader, device):
    model.eval()
    all_probs  = []
    all_labels = []

    with torch.no_grad():
        for x, y in dataloader:
            x     = x.to(device)
            probs = model(x).squeeze(1).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(y.numpy().tolist())

    return np.array(all_probs), np.array(all_labels)
