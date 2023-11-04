import re
import json
import pickle
import pandas as pd
import numpy as np

from typing import Any, Callable, Mapping, Sequence, Union

from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer, OneHotEncoder


_EXCLUDE = {"isNonstandard"}


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
    dataframe = dataframe[sorted(dataframe.columns)]
    return dataframe


def multihot_encode(series: pd.Series) -> pd.DataFrame:
    binarizer = MultiLabelBinarizer()
    encodings = binarizer.fit_transform(series)
    encodings_dataframe = pd.DataFrame(data=encodings, columns=binarizer.classes_)
    return encodings_dataframe.add_prefix(f"{series.name}.")


def scale_encode(
    series: pd.Series,
    scaler_fn: Union[StandardScaler, MinMaxScaler],
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


def concat_encodings(dataframes: Sequence[pd.DataFrame]) -> pd.DataFrame:
    encoding = pd.concat([e.reset_index(drop=True) for e in dataframes], axis=1)
    return encoding


def get_species(dataframe: pd.DataFrame, typechart: pd.DataFrame) -> np.ndarray:
    type_str = dataframe["types"].apply(lambda x: ",".join(x))
    type_dummies = type_str.str.get_dummies(sep=",")
    typechart_transposed = pd.DataFrame(data=typechart.values, index=typechart.columns)
    weakness_matrix = type_dummies.apply(
        lambda types: typechart_transposed[types == 1].prod(), axis=1
    )
    weakness_matrix.columns = [f"weakness.{col}" for col in typechart.columns]

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
        weakness_matrix,
    ]

    return concat_encodings(encodings)


def get_moves(dataframe: pd.DataFrame) -> pd.DataFrame:
    flags = [column for column in dataframe.columns if column.startswith("flags.")]
    flags_dataframe = dataframe[flags].fillna(0)

    boosts = [column for column in dataframe.columns if column.startswith("boosts.")]
    boosts_dataframe = dataframe[boosts].fillna(0)
    boosts_dataframes = [onehot_encode(boosts_dataframe[field]) for field in boosts]

    secondary = [
        column for column in dataframe.columns if column.startswith("secondary.")
    ]
    secondary_dataframe = dataframe[secondary].fillna(0)
    secondary_dataframes = [
        onehot_encode(secondary_dataframe[field], fn=lambda x: str(x))
        for field in secondary
    ]

    encodings = [
        onehot_encode(dataframe["target"]),
        dataframe["accuracy"].map(lambda x: x if x == 1 else x / 100).to_frame(),
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
        dataframe["drain"]
        .map(lambda x: x[0] / x[1] if isinstance(x, list) else 0)
        .to_frame(),
        dataframe["recoil"]
        .map(lambda x: x[0] / x[1] if isinstance(x, list) else 0)
        .to_frame(),
        dataframe["heal"]
        .map(lambda x: x[0] / x[1] if isinstance(x, list) else 0)
        .to_frame(),
        multihot_encode(
            dataframe["multihit"]
            .fillna(1)
            .map(lambda x: x if isinstance(x, list) else [x, x])
        ),
        dataframe[["forceSwitch", "hasCrashDamage"]].fillna(0),
        onehot_encode(dataframe["volatileStatus"]),
        *secondary_dataframes,
    ]
    return concat_encodings(encodings)


def get_abilities(dataframe: pd.DataFrame) -> pd.DataFrame:
    conditions = [
        column for column in dataframe.columns if column.startswith("condition.")
    ]
    conditions_dataframe = dataframe[conditions].fillna(0)

    ons = [s for s in dataframe.columns if re.match(r"on[A-Z]", s)]
    ons_dataframe = dataframe[ons].fillna(0)

    iss = [
        s for s in dataframe.columns if re.match(r"is[A-Z]", s) and s not in _EXCLUDE
    ]
    iss_dataframe = dataframe[iss].fillna(0)

    encodings = [
        onehot_encode(dataframe["id"]),
        iss_dataframe,
        ons_dataframe,
        conditions_dataframe,
    ]
    if "suppressWeather" in dataframe.columns:
        encodings += [dataframe["suppressWeather"].fillna(0)]

    concat = concat_encodings(encodings)

    object_columns = concat.dtypes[concat.dtypes == object].index
    encodings = [
        concat.drop(object_columns.tolist(), axis=1),
        *[onehot_encode(concat[column], lambda x: str(x)) for column in object_columns],
    ]

    return concat_encodings(encodings)


def get_items(dataframe: pd.DataFrame) -> pd.DataFrame:
    conditions = [column for column in dataframe if column.startswith("condition.")]
    conditions_dataframe = dataframe[conditions].fillna(0)

    fling = [column for column in dataframe if column.startswith("fling.")]
    fling_dataframe = dataframe[fling].fillna(0)
    fling_dataframes = [
        onehot_encode(fling_dataframe[field], lambda x: str(x)) for field in fling
    ]

    ons = [s for s in dataframe if re.match(r"on[A-Z]", s)]
    ons_dataframe = dataframe[ons].fillna(0)

    iss = [s for s in dataframe if re.match(r"is[A-Z]", s) and s not in _EXCLUDE]
    iss_dataframe = dataframe[iss].fillna(0)

    boosts = [column for column in dataframe if column.startswith("boosts.")]
    boosts_dataframe = dataframe[boosts].fillna(0)
    boosts_dataframes = [onehot_encode(boosts_dataframe[field]) for field in boosts]

    encodings = [
        onehot_encode(dataframe["id"]),
        iss_dataframe,
        ons_dataframe,
        conditions_dataframe,
        *fling_dataframes,
        *boosts_dataframes,
    ]

    concat = concat_encodings(encodings)

    object_columns = concat.dtypes[concat.dtypes == object].index
    encodings = [
        concat.drop(object_columns.tolist(), axis=1),
        *[onehot_encode(concat[column], lambda x: str(x)) for column in object_columns],
    ]

    return concat_encodings(encodings)


def get_conditions(dataframe: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(data=np.zeros((len(dataframe), 1)))


def get_typechart(dataframe: pd.DataFrame) -> pd.DataFrame:
    damage_taken = [col for col in dataframe.columns if col.startswith("damageTaken")]
    damage_taken_dataframe = dataframe[damage_taken]
    damage_taken_values = damage_taken_dataframe.values

    damage_taken_values = (
        (damage_taken_values == 0) * 1
        + (damage_taken_values == 1) * 2
        + (damage_taken_values == 2) * 0.5
        + (damage_taken_values == 3) * 0
    )

    type_names = list(map(lambda x: x.split(".")[-1], damage_taken_dataframe.columns))

    damage_taken_dataframe = pd.DataFrame(
        data=damage_taken_values, columns=type_names, index=type_names
    )
    encodings = [damage_taken_dataframe]
    return concat_encodings(encodings)


def main():
    with open("pokemb/data/data.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    output_obj = {}

    for gen, gendata in reversed(data.items()):
        output_obj[gen] = {}

        typechart_encodings = get_typechart(get_dataframe(gendata["typechart"]))

        for key, records in gendata.items():
            dataframe = get_dataframe(records)
            if key == "species":
                encodings = get_species(dataframe, typechart_encodings)
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

            output_obj[gen][key] = encodings.values.astype(np.float32)
            assert len(dataframe) == len(encodings)
            print(gen, key, encodings.shape)

        print()

    with open("pokemb/data/encodings.pkl", "wb") as f:
        pickle.dump(output_obj, f)


if __name__ == "__main__":
    main()
