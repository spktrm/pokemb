import json
import os
import pickle
import torch
import torch.nn as nn


def load_gen_data(gen: int):
    with open(os.path.join(os.getcwd(), "data/data.json"), "r") as f:
        return json.load(f)[gen]


def load_pkl_gen(gen: int):
    with open(os.path.join(os.getcwd(), "data/encodings.pkl"), "rb") as f:
        return pickle.load(f)[gen]


class PokEmb(nn.Module):
    def __init__(self, gen: int = 9):
        super().__init__()

        raw = load_gen_data(f"gen{gen}")
        gendata = load_pkl_gen(f"gen{gen}")

        self.species = nn.Embedding.from_pretrained(
            torch.from_numpy(gendata["species"])
        )
        self.species_names = [v["id"] for v in raw["species"] if v["tier"] != "Illegal"]

        self.moves = nn.Embedding.from_pretrained(torch.from_numpy(gendata["moves"]))
        self.moves_names = [v["id"] for v in raw["moves"]]

        self.abilities = nn.Embedding.from_pretrained(
            torch.from_numpy(gendata["abilities"])
        )
        self.abilities_names = [v["id"] for v in raw["abilities"]]

        self.items = nn.Embedding.from_pretrained(torch.from_numpy(gendata["items"]))
        self.items_names = [v["id"] for v in raw["items"]]

        self.conditions = nn.Embedding.from_pretrained(
            torch.from_numpy(gendata["conditions"])
        )
        self.conditions_names = raw["conditions"]

        self.typechart = nn.Embedding.from_pretrained(
            torch.from_numpy(gendata["typechart"])
        )
        self.typechart_names = raw["typechart"]

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        ...

    def forward_species(self, indices: torch.Tensor) -> torch.Tensor:
        return self.species(indices)

    def forward_moves(self, indices: torch.Tensor) -> torch.Tensor:
        return self.moves(indices)

    def forward_abilities(self, indices: torch.Tensor) -> torch.Tensor:
        return self.abilities(indices)

    def forward_items(self, indices: torch.Tensor) -> torch.Tensor:
        return self.items(indices)

    def forward_conditions(self, indices: torch.Tensor) -> torch.Tensor:
        return self.conditions(indices)

    def forward_typechart(self, indices: torch.Tensor) -> torch.Tensor:
        return self.typechart(indices)
