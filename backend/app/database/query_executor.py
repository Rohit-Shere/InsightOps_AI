import pandas as pd
from sqlalchemy import text
from app.database.connection import engine

def fetch_dataframe(query: str, params: dict | None = None):
    """
    Execute a parameterized analytical query
    and return the result as a Pandas DataFrame.
    """
    with engine.connect() as connection:
        return pd.read_sql_query(
            text(query),
            connection,
            params=params or {}
        )