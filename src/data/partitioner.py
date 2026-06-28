import numpy as np


def partition_noniid(X_tr, y_tr, num_clients, alpha=0.5, seed=42):
    rng = np.random.default_rng(seed)
    classes = np.unique(y_tr).astype(int)

    client_indices = {i: [] for i in range(num_clients)}

    for c in classes:
        idx_c = np.where(y_tr == c)[0]
        rng.shuffle(idx_c)
        proportions = rng.dirichlet(alpha=np.ones(num_clients) * alpha)
        splits = (proportions * len(idx_c)).astype(int)
        splits[-1] = len(idx_c) - splits[:-1].sum()

        start = 0
        for client_id, n in enumerate(splits):
            client_indices[client_id].extend(idx_c[start: start + n].tolist())
            start += n

    return client_indices
