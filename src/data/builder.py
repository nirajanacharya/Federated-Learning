import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


def make_tensor_ds(X, y):
    return TensorDataset(torch.from_numpy(X), torch.from_numpy(y))


def make_loader(ds, batch_size=32, shuffle=False):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def build_client_loaders(X_tr, y_tr, client_indices, batch_size=16):
    loaders = {}
    for cid, idx in client_indices.items():
        idx = np.array(idx)
        cx = torch.from_numpy(X_tr[idx])
        cy = torch.from_numpy(y_tr[idx])
        ds = TensorDataset(cx, cy)
        loaders[cid] = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
    return loaders
