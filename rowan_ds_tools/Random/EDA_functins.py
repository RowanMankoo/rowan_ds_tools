import pandas as pd


def isOneToOne(df, col1, col2):
    """Checks if col1 and col2 have a 1-1 relationship in a pandas df"""
    first = df.drop_duplicates([col1, col2]).groupby(col1)[col2].count().max()
    second = df.drop_duplicates([col1, col2]).groupby(col2)[col1].count().max()
    return first + second == 2
