"""Data cleaning module."""

import pandas as pd


def load_data(path):
    """Load dataset."""

    df = pd.read_csv(path)

    df.drop_duplicates(inplace=True)

    return df
