from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class TabularFFNN(nn.Module):
    """Simple feed-forward network for tabular data with numeric and categorical features."""

    def __init__(self, num_features_dim: int, cat_cardinalities: Dict[str, int], embedding_dim: int = 5):
        super().__init__()
        self.cat_cardinalities = cat_cardinalities
        self.num_features_dim = num_features_dim

        self.embeddings = nn.ModuleDict(
            {
                col: nn.Embedding(num_embeddings=card, embedding_dim=embedding_dim)
                for col, card in cat_cardinalities.items()
            }
        )

        total_embed_dim = sum(emb.embedding_dim for emb in self.embeddings.values())
        input_dim = num_features_dim + total_embed_dim

        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.out = nn.Linear(32, 1)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor):
        embed_list = []
        for i, col in enumerate(self.embeddings):
            col_indices = x_cat[:, i]
            emb_out = self.embeddings[col](col_indices)
            embed_list.append(emb_out)

        x_cat_embedded = torch.cat(embed_list, dim=1)
        x = torch.cat([x_num, x_cat_embedded], dim=1)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        pred = self.out(x)
        return pred


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0

    for batch_num, batch_cat, batch_y in loader:
        optimizer.zero_grad()
        preds = model(batch_num, batch_cat)
        loss = criterion(preds, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch_num.size(0)

    return running_loss / len(loader.dataset)


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch_num, batch_cat, batch_y in loader:
            preds = model(batch_num, batch_cat)
            loss = criterion(preds, batch_y)
            total_loss += loss.item() * batch_num.size(0)

    return total_loss / len(loader.dataset)
