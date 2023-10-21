import math
import pandas as pd

from typing import List, Callable


def get_column_type(df: pd.DataFrame, column: str) -> str:
    unique_values = df[column].astype(str).unique()
    if len(unique_values) <= 1:
        return "ignore"

    dtype = df[column].dtype.name
    if dtype.startswith("int") or dtype.startswith("float"):
        column_type = "continuous"
    else:
        column_type = "categorical"
    return column_type


def get_column_types(df: pd.DataFrame) -> pd.DataFrame:
    col_types = {}
    for column in df.columns:
        col_type = get_column_type(df, column)
        if col_type not in col_types:
            col_types[col_type] = []
        col_types[col_type].append(column)

    cardinality = {}
    for col in col_types["categorical"]:
        cardinality[col] = len(df[col].astype(str).unique())
    return col_types, cardinality


def z_score_scale(
    df: pd.DataFrame,
    column: str,
    clip: bool = False,
    lower: int = None,
    higher: int = None,
) -> pd.DataFrame:
    series = df[column]
    if clip:
        series = series.clip(lower=lower, higher=higher)
    mean = series.mean()
    std = series.std()
    transformed = (series - mean) / std
    df[f"{column}: zscore"] = transformed
    return df


def onehot_encode(df: pd.DataFrame, column: str) -> pd.DataFrame:
    series = df[column].astype(str)
    unique_categories = [
        category for category in sorted(series.unique()) if category not in ["nan"]
    ]
    records = series.map(
        lambda category: {
            f"{column}: {unique_category}": int(unique_category == category)
            for unique_category in unique_categories
        }
    ).tolist()
    new_df = pd.DataFrame.from_records(records)
    return concat_dfs(df, new_df)


def lambda_onehot_encode(df: pd.DataFrame, column: str, fn: Callable) -> pd.DataFrame:
    series = df[column].astype(float)
    max_value = series.max()
    max_lambda_value = fn(max_value)
    df[column] = series.map(lambda value: min(fn(value), max_lambda_value))
    return onehot_encode(df, column)


def sqrt_onehot_encode(df: pd.DataFrame, column: str) -> pd.DataFrame:
    return lambda_onehot_encode(df, column, fn=lambda x: int(x**0.5))


def log_onehot_encode(df: pd.DataFrame, column: str) -> pd.DataFrame:
    return lambda_onehot_encode(df, column, fn=lambda x: int(math.log(x)))


def multihot_encode(df: pd.DataFrame, column: str) -> pd.DataFrame:
    all_categories = set()
    series = df[column]
    series.map(
        lambda categories: [
            all_categories.add(categories)
            for categories in (categories if isinstance(categories, list) else [])
        ]
    )
    all_categories = list(sorted(all_categories))
    records = series.map(
        lambda categories: {
            f"{column}: {unique_category}": unique_category
            in (categories if isinstance(categories, list) else [])
            for unique_category in all_categories
        }
    ).tolist()
    new_df = pd.DataFrame.from_records(records)
    return concat_dfs(df, new_df)


def json_norm_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    if column not in df.columns:
        return df
    json_df = pd.json_normalize(df[column])
    json_df = json_df[sorted(json_df.columns)]
    json_df = json_df.fillna(0)
    json_df.columns = [f"{column}: {col}" for col in json_df.columns]
    return concat_dfs(df, json_df)


def concat_dfs(*dfs: List[pd.DataFrame], axis: int = 1) -> pd.DataFrame:
    new_dfs = []
    for df in dfs:
        try:
            df = df.reset_index()
        except:
            pass
        new_dfs.append(df)
    concat_df = pd.concat(new_dfs, axis=axis)
    for column in ["level_0", "index"]:
        try:
            concat_df = concat_df.drop([column], axis=1)
        except:
            pass
    return concat_df


def toid(string: str) -> str:
    return "".join([c for c in string if c.isalnum()]).lower().strip()
