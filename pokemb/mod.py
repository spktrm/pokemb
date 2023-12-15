import os
import pickle
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Any, Dict, Literal


EncodingType = Literal["raw", "pretrained"]
EncodingSize = Literal[32, 64, 128, 256]

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))


def load_pkl_gen(gen: int):
    path = "data/encodings.pkl"
    with open(os.path.join(ROOT_DIR, path), "rb") as f:
        return pickle.load(f)[f"gen{gen}"]


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


class ComponentEmbedding(nn.Module):
    def __init__(
        self,
        gendata: Dict[str, Any],
        name: str,
        output_size: int,
        num_layers: int = 1,
        num_unknown: int = 1,
        use_layer_norm: bool = False,
        include_mlp: bool = True,
    ):
        super().__init__()
        self.name = name
        data = gendata[name]
        mask = data["mask"]
        self.mask = mask

        vectors = data["vectors"]
        vector_size = vectors.shape[-1]

        placeholder = np.zeros((mask.max() + 1, vector_size))
        placeholder[mask] = data["vectors"]

        self.num_unknown = num_unknown
        self.unknown = nn.Embedding(num_unknown, output_size)
        self.data = nn.Embedding.from_pretrained(torch.from_numpy(placeholder).float())

        self.include_mlp = include_mlp
        if self.include_mlp:
            self.mlp = make_mlp(
                vector_size, output_size, n_layers=num_layers, layer_norm=use_layer_norm
            )
        self.names = data["names"].tolist()

    def forward_indices(self, indices: torch.Tensor):
        return self.data((indices - self.num_unknown).clamp(min=0))

    def forward(self, indices: torch.Tensor):
        if self.include_mlp:
            return torch.where(
                (indices < self.num_unknown).unsqueeze(-1),
                self.unknown(indices.clamp(min=0, max=self.num_unknown - 1)),
                self.mlp(self.data((indices - self.num_unknown).clamp(min=0))),
            )


class PokEmb(nn.Module):
    def __init__(
        self,
        gen: int = 9,
        output_size: EncodingSize = None,
        use_layer_norm: bool = False,
        include_mlp: bool = False,
    ):
        super().__init__()

        gendata = load_pkl_gen(gen)

        self.species = ComponentEmbedding(
            gendata=gendata,
            name="species",
            num_unknown=2,
            use_layer_norm=use_layer_norm,
            include_mlp=include_mlp,
            output_size=output_size,
        )
        self.moves = ComponentEmbedding(
            gendata=gendata,
            name="moves",
            num_unknown=3,
            use_layer_norm=use_layer_norm,
            include_mlp=include_mlp,
            output_size=output_size,
        )
        self.abilities = ComponentEmbedding(
            gendata=gendata,
            name="abilities",
            num_unknown=2,
            use_layer_norm=use_layer_norm,
            include_mlp=include_mlp,
            output_size=output_size,
        )
        self.items = ComponentEmbedding(
            gendata=gendata,
            name="items",
            num_unknown=2,
            use_layer_norm=use_layer_norm,
            include_mlp=include_mlp,
            output_size=output_size,
        )

    def forward_species(self, indices: torch.Tensor) -> torch.Tensor:
        return self.species(indices)

    def forward_moves(self, indices: torch.Tensor) -> torch.Tensor:
        return self.moves(indices)

    def forward_abilities(self, indices: torch.Tensor) -> torch.Tensor:
        return self.abilities(indices)

    def forward_items(self, indices: torch.Tensor) -> torch.Tensor:
        return self.items(indices)
