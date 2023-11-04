import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader


class MoveAutoEncoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_embeddings: int = 1024,
        embedding_dim: int = 8,
        beta: float = 1e-3,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )
        self.embedding_dim = embedding_dim
        r = embedding_dim**-0.5
        self.beta = beta
        self.codebook = nn.Parameter(
            torch.randn(num_embeddings, embedding_dim).uniform_(-r, r)
        )
        self.decoder = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
        )

    def encode(self, x):
        return self.encoder(x)

    def quantize(self, z):
        B = z.shape[0]
        z_flat = z.view(B, -1, self.embedding_dim).flatten(0, 1)
        d = (
            torch.sum(z_flat**2, dim=1, keepdim=True)
            + torch.sum(self.codebook**2, dim=1)
            - 2 * torch.matmul(z_flat, self.codebook.t())
        )
        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = torch.embedding(self.codebook, min_encoding_indices)
        z_q = z_q.view_as(z)

        loss = F.mse_loss(z_q.detach(), z) + self.beta * F.mse_loss(z_q, z.detach())

        z_q = z + (z_q - z).detach()

        return z_q, loss

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        z = self.encode(x)
        z_q, embedding_loss = self.quantize(z)
        x_hat = self.decode(z_q)
        return x_hat, embedding_loss


def encode(
    vectors: np.ndarray,
    input_size: int,
    hidden_size: int,
    batch_size: int = 2048,
    lr: float = 3e-4,
    num_epochs: int = 1000,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MoveAutoEncoder(input_size, hidden_size).to(device)
    torch_vectors = torch.from_numpy(vectors.astype(float)).float()
    loader = DataLoader(torch_vectors, batch_size)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    while True:
        try:
            for batch in loader:
                optimizer.zero_grad()

                batch = batch.to(device)

                pred, loss = model(batch)
                loss += criterion(batch, pred)

                loss.backward()
                optimizer.step()

                print(f"{loss.item():.3f}", end="\r")
        except KeyboardInterrupt:
            break

        if loss.item() < 5e-3:
            break

    model = model.cpu()
    with torch.no_grad():
        z_q, _ = model.quantize(model.encode(torch_vectors))
        return z_q.cpu().detach().numpy()
