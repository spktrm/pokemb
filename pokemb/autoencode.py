import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader


def make_mlp(
    input_size: int,
    output_size: int,
    n_layers: int = 3,
    layer_norm: bool = True,
    activate_final: bool = False,
) -> nn.Module:
    layers_sizes = np.linspace(input_size, output_size, n_layers + 1).astype(int)
    layers = []
    for n, (in_feature, out_feature) in enumerate(
        zip(layers_sizes[:-1], layers_sizes[1:])
    ):
        layer = [nn.Linear(in_feature, out_feature)]
        if n < (n_layers - 1) or activate_final:
            layer.append(nn.ReLU())
            if layer_norm:
                layer.insert(1, nn.LayerNorm(out_feature))
        layers.append(nn.Sequential(*layer))
    return nn.Sequential(*layers)


class AE(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 4):
        super().__init__()

        self.encoder = make_mlp(input_size, hidden_size, num_layers)
        self.decoder = make_mlp(hidden_size, input_size, num_layers)

    def encode(self, x):
        return F.normalize(self.encoder(x), dim=-1)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, None


class SimSiam(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 4):
        super().__init__()

        self.torso = make_mlp(input_size, hidden_size, num_layers, activate_final=True)
        self.proj = make_mlp(hidden_size, hidden_size, num_layers, activate_final=True)

    def encode1(self, x):
        return self.proj(F.relu(self.torso(x)))

    def encode2(self, x):
        return self.torso(x)


class CodeBook(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_embeddings: int = 1024,
        embedding_dim: int = 32,
        beta: float = 1e-3,
    ):
        super().__init__()

        self.beta = beta
        self.nsplits = input_dim // embedding_dim
        self.codebooks = nn.ModuleList(
            [nn.Embedding(num_embeddings, embedding_dim) for _ in range(self.nsplits)]
        )

    def quantize(self, z: torch.Tensor, index: int):
        codebook = self.codebooks[index]
        codebook_weight = codebook.weight
        d = (
            torch.sum(z**2, dim=1, keepdim=True)
            + torch.sum(codebook_weight**2, dim=1)
            - 2 * torch.matmul(z, codebook_weight.t())
        )
        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = codebook(min_encoding_indices)
        z_q = z_q.view_as(z)

        loss = F.mse_loss(z_q.detach(), z) + self.beta * F.mse_loss(z_q, z.detach())
        z_q = z + (z_q - z).detach()
        return z_q, loss

    def forward(self, z: torch.Tensor):
        loss = 0
        zqs = []
        for i, zc in enumerate(torch.chunk(z, self.nsplits, -1)):
            zqn, zqloss = self.quantize(zc, i)
            loss += zqloss
            zqs.append(zqn)

        return torch.cat(zqs, dim=-1), loss


class VQVAE(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_embeddings: int = 1024,
        embedding_dim: int = 16,
        beta: float = 1e-4,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_size, input_size // 2),
            nn.LayerNorm(input_size // 2),
            nn.ReLU(),
            nn.Linear(input_size // 2, input_size // 4),
            nn.LayerNorm(input_size // 4),
            nn.ReLU(),
            nn.Linear(input_size // 4, hidden_size),
            nn.LayerNorm(hidden_size),
        )
        self.codebook = CodeBook(hidden_size, num_embeddings, embedding_dim, beta)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, input_size // 4),
            nn.LayerNorm(input_size // 4),
            nn.ReLU(),
            nn.Linear(input_size // 4, input_size // 2),
            nn.LayerNorm(input_size // 2),
            nn.ReLU(),
            nn.Linear(input_size // 2, input_size),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        z = self.encode(x)
        z_q, embedding_loss = self.codebook(z)
        x_hat = self.decode(z_q)
        return x_hat, embedding_loss


def encode(
    vectors: np.ndarray,
    input_size: int,
    hidden_size: int,
    batch_size: int = 2048,
    lr: float = 3e-4,
    thresh: float = 1e-2,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AE(input_size, hidden_size).to(device)
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
                if loss is not None:
                    loss += criterion(batch, pred)
                else:
                    loss = criterion(batch, pred)

                loss.backward()
                optimizer.step()

                print(f"{loss.item():.3f}", end="\r")
        except KeyboardInterrupt:
            break

        if loss.item() < thresh:
            break

    torch_vectors = torch_vectors.to(device)

    with torch.no_grad():
        if isinstance(model, VQVAE):
            z_q, _ = model.quantize(model.encode(torch_vectors))
            return z_q.cpu().detach().numpy()
        else:
            z_q = model.encode(torch_vectors)
            return z_q.cpu().detach().numpy()


def encode2(
    vectors: np.ndarray,
    input_size: int,
    hidden_size: int,
    batch_size: int = 2048,
    lr: float = 3e-4,
    thresh: float = 1e-2,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SimSiam(input_size, hidden_size).to(device)
    torch_vectors = torch.from_numpy(vectors.astype(float)).float()
    loader = DataLoader(torch_vectors, batch_size)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    while True:
        try:
            for batch in loader:
                optimizer.zero_grad()

                batch = batch.to(device)

                f1 = model.encode1(batch)
                f2 = model.encode2(batch).detach()

                f1 = F.normalize(f1, p=2.0, dim=-1, eps=1e-5)
                f2 = F.normalize(f2, p=2.0, dim=-1, eps=1e-5)
                loss = -(f1 * f2).sum(dim=1)
                loss = loss.mean()

                loss.backward()
                optimizer.step()

                print(f"{loss.item():.3f}", end="\r")
        except KeyboardInterrupt:
            break

        if loss.item() < thresh:
            break

    torch_vectors = torch_vectors.to(device)

    with torch.no_grad():
        if isinstance(model, VQVAE):
            z_q, _ = model.quantize(model.encode(torch_vectors))
            return z_q.cpu().detach().numpy()
        else:
            z_q = model.encode1(torch_vectors)
            return z_q.cpu().detach().numpy()
