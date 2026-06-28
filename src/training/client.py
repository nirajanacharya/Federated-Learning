import copy
import torch
import torch.nn as nn
import torch.optim as optim


class FLClient:
    def __init__(self, client_id, dataloader, model_fn, input_dim, lr=0.001, local_epochs=5, device="cpu"):
        self.client_id   = client_id
        self.dataloader  = dataloader
        self.model_fn    = model_fn
        self.input_dim   = input_dim
        self.lr          = lr
        self.local_epochs = local_epochs
        self.device      = device
        self.criterion   = nn.BCELoss()
        self.model       = model_fn(input_dim).to(device)

    def set_weights(self, global_weights):
        self.model.load_state_dict(copy.deepcopy(global_weights))

    def train_one_round(self):
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)

        total_loss  = 0.0
        num_batches = 0

        for _ in range(self.local_epochs):
            for bx, by in self.dataloader:
                bx = bx.to(self.device)
                by = by.to(self.device)
                optimizer.zero_grad()
                out  = self.model(bx).squeeze(1)
                loss = self.criterion(out, by)
                loss.backward()
                optimizer.step()
                total_loss  += loss.item()
                num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        return self.model.state_dict(), avg_loss, len(self.dataloader.dataset)
