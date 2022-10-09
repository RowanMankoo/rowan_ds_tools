import functools
import time

import pandas as pd
import sqlalchemy as sqa

from ..utils._param_validation import validate_params


def _db_connector_decorator(func):
    """Decorator which

    1. sets up connections to the db
    2. peforms decorated function
    3. closes connections to the DB

    Args:
        func (function): function to interact with the DB

    Returns:
        _type_: _description_
    """

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):

        # ---------- Sets up DB connection ----------
        try:
            engine = sqa.create_engine(self.connstring)
            conn = engine.connect()
        except:
            # Sometimes it doesn't connect on first try
            time.sleep(4)
            engine = sqa.create_engine(self.connstring)
            conn = engine.connect()

        # ---------- DB interaction ----------
        try:
            out = func(self, *args, **kwargs, conn=conn)

        except Exception as e:
            raise (e)

        # ---------- Terminate db connection ----------
        finally:
            conn.close()
            engine.dispose()

        return out

    return wrapper


def strip_column_wrappers(df):
    return [x.replace('"', "") for x in df.columns]


def add_column_wrappers(df):
    return ['"' + x + '"' for x in df.columns]


class PostgressHelper:
    @validate_params({"conn_string": [str]})
    def __init__(self, conn_string):
        self.connstring = conn_string

    def get_db_engine(self):

        engine = sqa.create_engine(self.connstring)
        return engine

    @_db_connector_decorator
    @validate_params({"dequote": ["boolean"], "query": [str]})
    def query(self, query, dequote=True, **kwargs):
        """Queries the postgress DB and return the quieried result

        Args:
            query (str): query string
            dequote (bool, optional): option to dequote columns or not. Defaults to True.

        Returns:
            _type_: _description_
        """
        df = pd.read_sql(query, con=kwargs["conn"])
        if dequote:
            df.columns = strip_column_wrappers(df)

        return df

    @_db_connector_decorator
    @validate_params({"query": [str]})
    def alter_db(self, query, **kwargs):
        """Peforms query to alter the postgress DB, such as drop drop table or rows

        Args:
            query (str): query to alter db
        """
        kwargs["conn"].execute(query)
        print("query executed")

        return

    @_db_connector_decorator
    @validate_params({"df": [pd.core.frame.DataFrame], "table_name": [str]})
    def upload_to_db(
        self,
        df,
        table_name,
        if_exists,
        index=False,
        index_label=None,
        chunksize=None,
        dtype=None,
        method=None,
        add_quotes=True,
        **kwargs,
    ):

        """Uploads data to the database

        Args:
            df (pd.core.frame.DataFrame): dataframe to upload
            table_name (str): name of table to upload into
            if_exists (str): .to_sql arg
            index (bool): .to_sql arg
            index_label (str): .to_sql arg
            chunksize (int): .to_sql arg
            dtype (): .to_sql arg
            method (): .to_sql arg
            add_quotes (bool): whether to add quotes to df in ordder to insert into pandas df


        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        # TODO add if statements on if_exists to add this in
        # self.check_columns_align(df, table_name, dequote=add_quotes)
        if add_quotes:
            df.columns = add_column_wrappers(df)

        df.to_sql(
            table_name,
            con=kwargs["conn"],
            if_exists=if_exists,
            index=index,
            index_label=index_label,
            chunksize=chunksize,
            dtype=dtype,
            method=method,
        )
        print("upload completed")

        return

    @validate_params({"table_name": [str]})
    def query_column_names(self, table_name, dequote=True):
        """queries postgress DB to get the column names from a specified table

        Args:
            table_name (str): name of table to query

        Returns:
            (list): list of column names in table
        """

        column_name_query = (
            "SELECT column_name FROM information_schema.columns where table_name = '"
            + table_name
            + "' order by column_name"
        )
        if dequote:
            column_names = (
                self.query(column_name_query)["column_name"]
                .apply(lambda x: x.replace('"', ""))
                .values
            )
        else:
            column_names = self.query(column_name_query)["column_name"].values

        return column_names

    @validate_params({"df": [pd.core.frame.DataFrame], "table_name": [str]})
    def check_columns_align(self, df, table_name, dequote=True):
        """Checks names in local df matches names in postgres table

        Args:
            df (pd.core.frame.DataFrame): local df
            table_name (str): name of table in pandas df
        """

        db_cols = set(self.query_column_names(table_name, dequote=dequote))
        df_cols = set(df.columns)

        db_cols_not_in_df = db_cols.difference(df_cols)
        df_cols_not_in_db = df_cols.difference(db_cols)

        if len(db_cols_not_in_df) + len(df_cols_not_in_db) != 0:
            raise ValueError(
                f"The following columns {db_cols_not_in_df} are in the db but not in the df \n The following columns {df_cols_not_in_db} are in the df but not in the db"
            )
        else:
            print("All columns align!")
            return
