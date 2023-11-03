import json
import math
import pickle
import pandas as pd
import numpy as np

from typing import Any, Callable, Mapping, Sequence, Union
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer, OneHotEncoder


def get_num_unique(dataframe: pd.DataFrame, column: str) -> int:
    values = dataframe[column].map(lambda x: json.dumps(x))
    return values.nunique()


def remove_redundant_columns(dataframe: pd.DataFrame):
    nunique = {}
    for column in dataframe.columns:
        nunique[column] = get_num_unique(dataframe, column)
    return dataframe.drop([column for column, n in nunique.items() if n <= 1], axis=1)


def get_dataframe(records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    dataframe = pd.json_normalize(records)
    try:
        dataframe = dataframe[dataframe["tier"] != "Illegal"]
    except:
        pass
    try:
        dataframe = dataframe[dataframe["isNonstandard"] != "Future"]
    except:
        pass
    try:
        dataframe = dataframe.drop(["exists"], axis=1)
    except:
        pass
    dataframe = remove_redundant_columns(dataframe)
    return dataframe


def multihot_encode(series: pd.Series) -> pd.DataFrame:
    binarizer = MultiLabelBinarizer()
    encodings = binarizer.fit_transform(series)
    encodings_dataframe = pd.DataFrame(data=encodings, columns=binarizer.classes_)
    return encodings_dataframe.add_prefix(f"{series.name}.")


def scale_encode(
    series: pd.Series,
    scaler_fn: Union[OneToOneFeatureMixin, TransformerMixin, BaseEstimator],
    name: str = None,
) -> pd.DataFrame:
    scaler = scaler_fn()
    encodings = scaler.fit_transform(series.values[..., None])
    encodings_dataframe = pd.DataFrame(
        data=encodings, columns=[name or scaler.__class__.__name__]
    )
    return encodings_dataframe.add_prefix(f"{series.name}.")


def zscore_encode(series: pd.Series) -> pd.DataFrame:
    return scale_encode(series, scaler_fn=StandardScaler, name="zscore")


def minmax_encode(series: pd.Series) -> pd.DataFrame:
    return scale_encode(series, scaler_fn=MinMaxScaler, name="minmax")


def onehot_encode(series: pd.Series, fn: Callable = lambda x: x) -> pd.DataFrame:
    encoder = OneHotEncoder()
    encodings = encoder.fit_transform(
        series.map(lambda x: fn(x)).values[..., None]
    ).toarray()
    encodings_dataframe = pd.DataFrame(data=encodings, columns=encoder.categories_[0])
    return encodings_dataframe.add_prefix(f"{series.name}.")


def sqrt_onehot_encode(series: pd.Series):
    return onehot_encode(series, lambda x: int(x**0.5))


def encode_stat(series: pd.Series):
    return [
        zscore_encode(series),
        sqrt_onehot_encode(series),
    ]


def get_species(dataframe: pd.DataFrame) -> np.ndarray:
    encodings = [
        multihot_encode(dataframe["types"]),
        *encode_stat(dataframe["weightkg"]),
        *encode_stat(dataframe["baseStats.hp"]),
        *encode_stat(dataframe["baseStats.atk"]),
        *encode_stat(dataframe["baseStats.def"]),
        *encode_stat(dataframe["baseStats.spa"]),
        *encode_stat(dataframe["baseStats.spd"]),
        *encode_stat(dataframe["baseStats.spe"]),
        *encode_stat(dataframe["bst"]),
        onehot_encode(dataframe["nfe"]),
    ]
    encoding = pd.concat(encodings, axis=1)
    return encoding


def get_moves(dataframe: pd.DataFrame) -> np.ndarray:
    flags = [column for column in dataframe if column.startswith("flags.")]
    flags_dataframe = dataframe[flags].fillna(0)

    boosts = [column for column in dataframe if column.startswith("boosts.")]
    boosts_dataframe = dataframe[boosts].fillna(0)
    boosts_dataframes = [onehot_encode(boosts_dataframe[field]) for field in boosts]

    secondary = [column for column in dataframe if column.startswith("secondary.")]
    secondary_dataframe = dataframe[secondary].fillna(0)
    secondary_dataframes = [
        onehot_encode(secondary_dataframe[field], fn=lambda x: str(x))
        for field in secondary
    ]

    encodings = [
        onehot_encode(dataframe["target"]),
        dataframe["accuracy"].map(lambda x: x if x == 1 else x / 100),
        onehot_encode(dataframe["accuracy"] == 1),
        zscore_encode(dataframe["basePower"]),
        sqrt_onehot_encode(dataframe["basePower"]),
        onehot_encode(dataframe["category"]),
        minmax_encode(dataframe["pp"]),
        onehot_encode(dataframe["pp"]),
        onehot_encode(dataframe["priority"]),
        onehot_encode(dataframe["type"]),
        onehot_encode(dataframe["critRatio"]),
        flags_dataframe,
        *boosts_dataframes,
        dataframe["drain"].map(lambda x: x[0] / x[1] if isinstance(x, list) else 0),
        dataframe["recoil"].map(lambda x: x[0] / x[1] if isinstance(x, list) else 0),
        dataframe["heal"].map(lambda x: x[0] / x[1] if isinstance(x, list) else 0),
        multihot_encode(
            dataframe["multihit"]
            .fillna(1)
            .map(lambda x: x if isinstance(x, list) else [x, x])
        ),
        dataframe[["forceSwitch", "hasCrashDamage"]].fillna(0),
        onehot_encode(dataframe["volatileStatus"]),
        *secondary_dataframes,
    ]
    encoding = pd.concat(encodings, axis=1)
    return encoding.values


def get_abilities(dataframe: pd.DataFrame) -> np.ndarray:
    conditions = [column for column in dataframe if column.startswith("condition.")]
    conditions_dataframe = dataframe[conditions].fillna(0)

    encodings = [
        onehot_encode(dataframe["isBreakable"]),
        onehot_encode(dataframe["onTryHitPriority"]),
        conditions_dataframe,
    ]
    encoding = pd.concat(encodings, axis=1)
    return encoding.values


def get_items(dataframe: pd.DataFrame) -> np.ndarray:
    return np.zeros(1)


def get_conditions(dataframe: pd.DataFrame) -> np.ndarray:
    return np.zeros(1)


def get_typechart(dataframe: pd.DataFrame) -> np.ndarray:
    return np.zeros(1)


def main():
    with open("data/data.json", "r") as f:
        data = json.load(f)

    output_obj = {}

    for gen, gendata in data.items():
        output_obj[gen] = {}
        for key, records in gendata.items():
            dataframe = get_dataframe(records)
            if key == "species":
                encodings = get_species(dataframe)
            elif key == "moves":
                encodings = get_moves(dataframe)
            elif key == "abilities":
                encodings = get_abilities(dataframe)
            elif key == "items":
                encodings = get_items(dataframe)
            elif key == "conditions":
                encodings = get_conditions(dataframe)
            elif key == "typechart":
                encodings = get_typechart(dataframe)

            print(gen, key, encodings.shape)
            output_obj[gen][key] = encodings

        print()

    with open("data/encodings.pkl", "wb") as f:
        pickle.dump(output_obj, f)


if __name__ == "__main__":
    main()
