import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

import numpy as np

from pokemb.mod import make_mlp


def pred(
    X: np.ndarray,
    y: np.ndarray,
    hidden_dim: int = 128,
    model: nn.Module = None,
    num_epochs: int = 10,
    lr: float = 1e-4,
    criterion: nn.Module = nn.CosineSimilarity(dim=-1),
):
    if model is None:
        model = make_mlp(X.shape[-1], hidden_dim)
    pred = make_mlp(hidden_dim, hidden_dim)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    X = torch.from_numpy(X).to(device)
    y = torch.from_numpy(y).to(device)
    model = model.to(device)
    pred = pred.to(device)

    optimizer = optim.Adam(
        params=list(model.parameters()) + list(pred.parameters()), lr=lr
    )

    for i in range(num_epochs):
        optimizer.zero_grad()
        hid = model(X)
        out = pred(hid)
        loss = 1 - criterion(out, hid.detach())
        loss = loss.mean()
        print(loss.item())
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        return model(X).detach().cpu().numpy()


if __name__ == "__main__":
    import torch
    import numpy as np

    from pokemb.mod import PokEmb

    emb = PokEmb(9, 128, True, True)
    datum = "species"
    mod = getattr(emb, datum)
    embeddings = mod.data.weight.numpy()
    embeddings = embeddings[mod.mask]
    names = mod.names
    embeddings = pred(embeddings, np.arange(embeddings.shape[0]))
