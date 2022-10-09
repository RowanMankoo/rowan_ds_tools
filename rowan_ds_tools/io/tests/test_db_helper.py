import pytest
from rowan_ds_tools.io import db_helper
import pandas as pd
import numpy as np
import sqlalchemy


postgress = db_helper.PostgressHelper("pytests")


def test_query():
    query = "select * from example_1"
    df = postgress.query(query)

    result = df.equals(
        pd.DataFrame(
            np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), columns=["a", "b", "c"]
        )
    )
    assert result


def test_upload_to_db_and_alter_db():

    # Upload to DB test
    df = pd.DataFrame({"calories": [420, 380, 390], "duration": [50, 40, 45]})
    postgress.upload_to_db(df, "temp_1", if_exists="replace")

    # Check df's match
    df_queried = postgress.query("select * from temp_1", dequote=False)
    assert df.equals(df_queried)
    "dataframes do not match"

    # test we can drop the table
    query = "drop table temp_1"
    postgress.alter_db(query)

    # Check the table is deleted
    with pytest.raises(sqlalchemy.exc.ProgrammingError):
        df_queried = postgress.query("select * from temp_1")


def test_query_column_names():

    assert (postgress.query_column_names("example_1") == ["a", "b", "c"]).any()
    "Column name query failed"


def test_check_columns_align():

    df = pd.DataFrame(
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), columns=["a", "b", "c"]
    )

    postgress.check_columns_align(df, "example_1")

    with pytest.raises(ValueError):
        postgress.check_columns_align(df, "example_1", dequote=False)
