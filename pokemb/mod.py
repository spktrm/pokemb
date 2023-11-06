import os
import json
import pickle

import torch
import torch.nn as nn

from typing import Any, Dict, Literal


EncodingType = Literal["raw", "pretrained"]
EncodingSize = Literal[32, 64, 128, 256]


def load_gen_data(gen: int):
    with open(
        os.path.join(os.getcwd(), "pokemb/data/data.json"), "r", encoding="utf-8"
    ) as f:
        return json.load(f)[gen]


def load_pkl_gen(gen: int, type: EncodingType, size: EncodingSize):
    path = "pokemb/data/encodings"
    if type != "raw":
        path += f"-{size}"
    path += ".pkl"

    with open(os.path.join(os.getcwd(), path), "rb") as f:
        return pickle.load(f)[gen]


def is_valid(obj: Dict[str, Any]) -> bool:
    is_illegal = obj.get("tier", "") == "Illegal"
    is_future = obj.get("isNonstandard", "") == "Future"
    return not (is_illegal or is_future)


class PokEmb(nn.Module):
    def __init__(
        self,
        gen: int = 9,
        encoding_type: EncodingType = "raw",
        size: EncodingSize = None,
    ):
        super().__init__()

        raw = load_gen_data(f"gen{gen}")
        gendata = load_pkl_gen(f"gen{gen}", encoding_type, size)

        self.species = nn.Embedding.from_pretrained(
            torch.from_numpy(gendata["species"])
        )
        self.species_names = [v["id"] for v in raw["species"] if is_valid(v)]

        self.moves = nn.Embedding.from_pretrained(torch.from_numpy(gendata["moves"]))
        self.moves_names = [v["id"] for v in raw["moves"] if is_valid(v)]

        self.abilities = nn.Embedding.from_pretrained(
            torch.from_numpy(gendata["abilities"])
        )
        self.abilities_names = [v["id"] for v in raw["abilities"] if is_valid(v)]

        self.items = nn.Embedding.from_pretrained(torch.from_numpy(gendata["items"]))
        self.items_names = [v["id"] for v in raw["items"] if is_valid(v)]

        if encoding_type == "raw":
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
