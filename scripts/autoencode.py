from copy import deepcopy
import json
import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from pokemb.mod import ROOT_DIR, PokEmb, load_pkl_gen, make_mlp
from scripts.process import get_moves


class SimSiam(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 4):
        super().__init__()

        self.torso = make_mlp(input_size, hidden_size, num_layers)
        self.proj = make_mlp(hidden_size, hidden_size, num_layers)

    def project(self, x):
        return self.proj(F.relu(self.torso(x)))

    def forward(self, x):
        return self.torso(x)


class PokeEmbAE(nn.Module):
    def __init__(
        self,
        gen: int,
        hidden_size: int,
        num_layers: int = 4,
        use_layer_norm: bool = False,
    ):
        super().__init__()

        self.pokemb = PokEmb(
            gen=gen, output_size=hidden_size, use_layer_norm=use_layer_norm
        )
        self.simsiam = SimSiam(hidden_size, hidden_size)

    def encode(
        self,
        species_indices: torch.Tensor,
        moves_indices: torch.Tensor,
        abilities_indices: torch.Tensor,
        items_indices: torch.Tensor,
    ):
        summed = self.pokemb(
            species_indices, moves_indices, abilities_indices, items_indices
        )
        return summed

    def decode(self, x):
        return self.simsiam.project(x), self.simsiam(x)

    def forward(self, *args, **kwargs):
        z = self.encode(*args, **kwargs)
        return self.decode(z)


def load_json_data(gen: int):
    with open(os.path.join(ROOT_DIR, "data/data.json"), "r") as f:
        return json.load(f)[f"gen{gen}"]


def toid(string: str) -> str:
    return "".join((c for c in string if c.isalnum())).lower()


class PokeEmbLoader:
    def __init__(self, gen: int, batch_size: int):
        self.batch_size = batch_size

        global_data = load_json_data(gen)
        gen_data = load_json_data(9)
        pkl_data = load_pkl_gen(gen)

        self.species_index = {
            species["id"]: index for index, species in enumerate(global_data["species"])
        }

        self.species_categories = np.arange(len(self.species_index) - 1)
        self.species_dist = np.eye(len(self.species_index) - 1)[
            pkl_data["species"]["mask"]
        ].sum(0)
        self.species_dist /= self.species_dist.sum()

        ability_index = {
            ability["id"]: index for index, ability in enumerate(gen_data["abilities"])
        }
        self.ability_index = {
            self.species_index[pokemon["id"]]: list(
                set(
                    [
                        ability_index.get(toid(ability), -1)
                        for ability in pokemon["abilities"].values()
                    ]
                )
            )
            for pokemon in gen_data["species"]
        }

        self.item_index = {
            item["id"]: index for index, item in enumerate(gen_data["items"])
        }

        self.item_categories = np.arange(len(self.item_index))
        self.item_dist = np.eye(len(self.item_index))[pkl_data["items"]["mask"]].sum(0)
        self.item_dist /= self.item_dist.sum()

        move_index = {move["id"]: index for index, move in enumerate(gen_data["moves"])}

        self.categories = np.arange(len(self.species_index) - 1)
        self.dist = np.eye(len(self.species_index) - 1)[
            pkl_data["species"]["mask"]
        ].sum(0)
        self.dist /= self.dist.sum()

        self.learnset_indices = {
            index: list(
                set([move_index[move] for move in learnset.get("learnset", [])])
            )
            for index, learnset in enumerate(gen_data["learnsets"])
        }

        return

    def get_species_indices(self):
        indices = np.random.choice(
            self.species_categories, size=(self.batch_size,), p=self.species_dist
        )
        mask = np.random.sample((self.batch_size,)) < 0.15
        return np.where(mask, -1, indices), mask

    def get_item_indices(self, mask: np.ndarray):
        indices = np.random.choice(
            self.item_categories, size=(self.batch_size,), p=self.item_dist
        )
        mask = mask * (np.random.sample((self.batch_size,)) < 0.15)
        return np.where(mask, -1, indices)

    def get_ability_indices(self, species_indices: np.ndarray, mask: np.ndarray):
        indices = np.array(
            [random.choice(self.ability_index.get(i, [-1])) for i in species_indices]
        )
        mask = mask * (np.random.sample((self.batch_size,)) < 0.15)
        return np.where(mask, -1, indices)

    def get_move_indices(self, species_indices: np.ndarray, mask: np.ndarray, learnset):
        indices = []
        for i in species_indices:
            learnset_indices = learnset.get(i) or [-1]
            index = random.choice(learnset_indices)
            learnset_indices.remove(index)
            indices.append(index)

        indices = np.array(indices)
        mask = mask * (np.random.sample((self.batch_size,)) < 0.15)
        return np.where(mask, -1, indices)

    def get_batch(self):
        species_indices, mask = self.get_species_indices()
        item_indices = self.get_item_indices(mask)
        ability_indices = self.get_ability_indices(species_indices, mask)
        move_indices = np.zeros((self.batch_size, 4), dtype=np.int64)
        learnset = deepcopy(self.learnset_indices)
        for index in range(4):
            move_indices[:, index] = self.get_move_indices(
                species_indices, mask, learnset
            )
        return (
            species_indices + 1,
            item_indices + 1,
            ability_indices + 1,
            move_indices + 2,
        )

    def __iter__(self):
        while True:
            yield self.get_batch()


def train(
    gen: int,
    hidden_size: int,
    num_layers: int = 4,
    use_layer_norm: bool = False,
    batch_size: int = 2048,
    lr: float = 1e-5,
    thresh: float = -0.999,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"

    model = PokeEmbAE(gen, hidden_size, num_layers, use_layer_norm).to(device)
    loader = PokeEmbLoader(gen, batch_size)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    n = 0

    try:
        for species_indices, item_indices, ability_indices, move_indices in loader:
            species_indices = torch.from_numpy(species_indices).to(device)
            item_indices = torch.from_numpy(item_indices).to(device)
            ability_indices = torch.from_numpy(ability_indices).to(device)
            move_indices = torch.from_numpy(move_indices).to(device)

            optimizer.zero_grad()

            (proj, pred) = model(
                species_indices,
                move_indices,
                ability_indices,
                item_indices,
            )

            loss = -(F.normalize(proj) * F.normalize(pred.detach())).sum(-1).mean()

            loss.backward()
            optimizer.step()

            print(f"{n} {loss.item():.3f}", end="\r")
            n += 1

            if loss.item() < thresh:
                break

    except KeyboardInterrupt:
        pass

    torch.save(model.pokemb.state_dict(), "pokemb/data/weights.pt")


if __name__ == "__main__":
    train(9, 128, use_layer_norm=True)
