import json
import numpy as np
import pandas as pd

from typing import Dict, List, Any
from autoencode import encode
from utils import (
    z_score_scale,
    onehot_encode,
    sqrt_onehot_encode,
    json_norm_column,
    concat_dfs,
    get_column_types,
)

IGNORE_COLUMNS = {"id", "num", "desc", "shortDesc", "contestType"}

UNIQUE_ENCODINGS = {
    "drain",
    "multihit",
    "secondaries",
    "secondary",
    "heal",
    "accuracy",
    "recoil",
}

DROP_COLUMNS = {
    "isNonstandard",
    "isZ",
    "name",
    "fullname",
    "effectType",
    "kind",
    "gen",
    "isMax",
    "maxMove",
    "zMove",
    "realMove",
    "noMetronome",
    "exists",
}

SQRT_ONEHOT_COLUMNS = {"basePower", "pp"}

JSON_NORM_COLUMNS = {
    "flags",
    "boosts",
    "condition",
    "ignoreImmunity",
    "self",
    "selfBoost",
}


def load_dataframe(data: Dict[str, Any]):
    df = pd.DataFrame.from_records(data["moves"])
    return df[df["isNonstandard"] != "Future"]


def encode_multihit(df: pd.DataFrame) -> pd.DataFrame:
    max_value = max([value[-1] for value in df["multihit"] if isinstance(value, list)])
    padding = [0] * max_value

    def map_mulithit(value: list | int):
        if isinstance(value, list):
            if value[0] == 2 and value[1] == 5:
                output = [0, 0.35, 0.35, 0.15, 0.15]
                if len(output) < max_value:
                    output += padding
            else:
                output = [
                    1 / (value[1] - value[0] + 1)
                    if ((i <= value[1]) and (value[0] <= i))
                    else 0
                    for i in range(1, max_value + 1)
                ]
        else:
            output = [float(i == value) for i in range(1, max_value + 1)]
        return np.array(output)

    multihit_df = df["multihit"].map(map_mulithit)
    multihit_df = pd.DataFrame(
        data=np.stack(multihit_df.values),
        columns=[f"multihit: {col}" for col in range(1, max_value + 1)],
    )
    df = concat_dfs(df, multihit_df)
    return df


def encode_secondaries(df: pd.DataFrame) -> pd.DataFrame:
    secondary_effects = [
        i
        for o in df["secondaries"]
        .map(lambda x: x if isinstance(x, list) else [])
        .tolist()
        for i in o
    ]

    secondary_effects_deduped = {}
    for secondary_effect in secondary_effects:
        effect_hash = hash(json.dumps(secondary_effect))
        secondary_effects_deduped[effect_hash] = {
            "hash": effect_hash,
            **secondary_effect,
        }

    secondary_effects = secondary_effects_deduped
    secondary_effects = list(secondary_effects.values())
    secondary_effect_df = pd.DataFrame.from_records(
        pd.json_normalize(secondary_effects)
    )
    secondary_effect_df = secondary_effect_df[sorted(secondary_effect_df.columns)]

    hashes = secondary_effect_df["hash"]
    secondary_effect_df = secondary_effect_df.drop(["hash"], axis=1)

    chances = secondary_effect_df["chance"]
    secondary_effect_df = secondary_effect_df.drop(["chance"], axis=1)

    secondary_effect_df.columns = [
        f"secondary: {effect}" for effect in secondary_effect_df.columns
    ]
    for onehot_col in secondary_effect_df.columns:
        if onehot_col in {"chance", "hash"}:
            continue
        secondary_effect_df = onehot_encode(secondary_effect_df, onehot_col)

    secondary_effect_df = secondary_effect_df[
        sorted(secondary_effect_df.dtypes[secondary_effect_df.dtypes != object].index)
    ]

    secondary_effect_df = secondary_effect_df.fillna(0).astype(float)
    secondary_effect_df = secondary_effect_df.mul(chances / 100, axis=0)
    secondary_effect_vectors = secondary_effect_df.values

    def map_secondaries(secondary_effects):
        if isinstance(secondary_effects, list):
            secondary_effect_hashes = [
                hash(json.dumps(secondary_effect))
                for secondary_effect in secondary_effects
            ]
            output = np.concatenate(
                [
                    secondary_effect_vectors[hashes == secondary_effect_hash]
                    for secondary_effect_hash in secondary_effect_hashes
                ]
            ).sum(0)
        else:
            output = np.zeros(len(secondary_effect_df.columns))
        return output

    secondary_effect_df = pd.DataFrame(
        data=np.stack(df["secondaries"].map(map_secondaries).values),
        columns=secondary_effect_df.columns,
    )

    df = concat_dfs(df, secondary_effect_df)
    return df


def main():
    with open("data.json", "r") as f:
        data = json.load(f)

    df = load_dataframe(data)
    starting_columns = set(df.columns)

    df = df.drop(DROP_COLUMNS, axis=1)
    df = df.fillna(0)

    col_types, cardinalities = get_column_types(df)
    df = df.drop(col_types["ignore"], axis=1)

    for col in JSON_NORM_COLUMNS:
        df = json_norm_column(df, col)

    TWO_VALUES = (
        set(
            [
                column
                for column, cardinality in cardinalities.items()
                if cardinality == 2
            ]
        )
        - SQRT_ONEHOT_COLUMNS
        - JSON_NORM_COLUMNS
        - IGNORE_COLUMNS
        - UNIQUE_ENCODINGS
    )
    ONEHOT_COLUMNS = (
        set(
            [column for column, cardinality in cardinalities.items() if cardinality > 2]
        )
        - SQRT_ONEHOT_COLUMNS
        - JSON_NORM_COLUMNS
        - IGNORE_COLUMNS
        - UNIQUE_ENCODINGS
    )

    df = z_score_scale(df, "basePower")
    df = z_score_scale(df, "pp")

    for onehot_col in ONEHOT_COLUMNS:
        if onehot_col in df.columns:
            df = onehot_encode(df, onehot_col)

    for sqrt_onehot_col in SQRT_ONEHOT_COLUMNS:
        if sqrt_onehot_col in df.columns:
            try:
                df = sqrt_onehot_encode(df, sqrt_onehot_col)
            except:
                print(sqrt_onehot_col)

    df["accuracy: accuracyCheck"] = df["accuracy"].map(lambda x: int(x) == 1)
    df["accuracy"] = df["accuracy"].map(lambda value: 100 if value == 1 else value)
    df = z_score_scale(df, "accuracy")

    df["drainRatio"] = df["drain"].map(
        lambda x: x[0] / x[1] if isinstance(x, list) else 0
    )
    df["recoilRatio"] = df["recoil"].map(
        lambda x: x[0] / x[1] if isinstance(x, list) else 0
    )
    df = encode_multihit(df)
    df = encode_secondaries(df)

    persistent_onehot_columns = (
        set(df.dtypes[df.dtypes == object].index)
        - starting_columns
        - set(cardinalities)
    )
    for onehot_col in persistent_onehot_columns:
        if onehot_col in df.columns:
            df = onehot_encode(df, onehot_col)
        df = df.drop([onehot_col], axis=1)

    for column in TWO_VALUES:
        df[f"_{column}"] = df[column]

    heal_rename = {"heal": ["healRatio", "doesHeal"]}
    df = df.rename(
        columns=lambda c: heal_rename[c].pop(0) if c in heal_rename.keys() else c
    )
    df["healRatio"] = df["healRatio"].map(
        lambda x: x[0] / x[1] if isinstance(x, list) else 0
    )

    finish_columns = set(df.columns)

    vector_columns = finish_columns - starting_columns
    vector_columns = sorted(list(vector_columns))

    for column in df.columns:
        if column not in col_types["continuous"] and column.endswith(": 0"):
            vector_columns.remove(column)

    vectors = df[vector_columns].values.astype(float)
    input_size = vectors.shape[-1]
    hidden_size = 2 ** (len(bin(vectors.shape[-1])[2:]) - 1)
    vectors = encode(vectors, input_size=input_size, hidden_size=hidden_size)

    with open("gen3moves.json", "w") as f:
        json.dump(
            [
                {"name": name, "vector": vector}
                for name, vector in zip(df["id"], vectors.tolist())
            ],
            f,
        )


if __name__ == "__main__":
    main()
