import pandas as pd
import ast

"""
Please upload any useful data cleaning functions you tend to use regularly
"""


def __turn_string_to_dict(d):
    """if already a dict return dict, if string return dict"""
    try:
        # If d is a string
        d = ast.literal_eval(d)
    except (ValueError):
        # In case d is already a dictionary data type
        pass
    return d


def check_json_col_names(df, col):
    """Checks if JSON based column has consistent varibles among itself across all rows

    Args:
        df (_type_): _description_
        col (_type_): _description_
    """

    df[col].apply(__turn_string_to_dict)
    initial_keys = df[col][0].keys()
    for _, row in df.iterrows():
        if initial_keys != row[col].keys():
            print(
                f"The JSON column '{col}' does NOT have consistent varibles across all entries"
            )
            return
    print(f"The JSON column '{col}' DOES have consistent varibles across all entries")


def turn_json_col_to_features(df, col_name):
    """In a pandas df expand a dictionary based column and drop the original"""
    # Converts column to dictionary if it isn't already
    df[col_name].apply(__turn_string_to_dict)

    for i, row in df.iterrows():

        d = row[col_name]
        # If nan value then continue
        if isinstance(d, float):
            continue

        for feature_name, feature_value in d.items():
            df.loc[i, col_name + "_" + feature_name] = str(feature_value)

    # Remove old JSON column
    df.drop(columns=col_name, inplace=True)
