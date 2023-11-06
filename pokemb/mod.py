import os
import pickle
import numpy as np

import torch
import torch.nn as nn

from typing import Any, Dict, Literal


EncodingType = Literal["raw", "pretrained"]
EncodingSize = Literal[32, 64, 128, 256]

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))


def load_pkl_gen(gen: int, type: EncodingType, size: EncodingSize):
    path = "data/encodings"
    if type != "raw":
        path += f"-{size}"
    path += ".pkl"

    with open(os.path.join(ROOT_DIR, path), "rb") as f:
        return pickle.load(f)[gen]


class PokEmb(nn.Module):
    def __init__(
        self,
        gen: int = 9,
        encoding_type: EncodingType = "raw",
        size: EncodingSize = None,
    ):
        super().__init__()

        gendata = load_pkl_gen(f"gen{gen}", "raw", size)
        if encoding_type != "raw":
            override = load_pkl_gen(f"gen{gen}", encoding_type, size)

            for key, vectors in override.items():
                gendata[key]["vectors"] = vectors

        self.load_matrix(gendata, "species")
        self.load_matrix(gendata, "moves")
        self.load_matrix(gendata, "abilities")
        self.load_matrix(gendata, "items")

    def load_matrix(self, gendata: Dict[str, Any], name: str):
        data = gendata[name]
        mask = data["mask"]
        vectors = data["vectors"]
        placeholder = np.zeros((mask.max() + 1, vectors.shape[-1]))
        placeholder[mask] = data["vectors"]
        setattr(self, name, nn.Embedding.from_pretrained(torch.from_numpy(placeholder)))
        setattr(self, f"{name}_names", data["names"].tolist())

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
