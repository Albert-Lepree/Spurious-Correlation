"""
Simple Databricks query interface.
Usage:
    connection = init_databricks()
    df = query_databricks(connection, sql_query)
    connection.close()
"""

import pandas as pd
from databricks import sql
import os
from dotenv import load_dotenv

load_dotenv()

def init_databricks():
    """
    Initialize Databricks connection.
    Requires environment variables:
        DATABRICKS_SERVER_HOSTNAME
        DATABRICKS_HTTP_PATH
        DATABRICKS_TOKEN
    
    Returns:
        connection object
    """
    connection = sql.connect(
        server_hostname=os.getenv("DATABRICKS_SERVER_HOSTNAME"),
        http_path=os.getenv("DATABRICKS_HTTP_PATH"),
        access_token=os.getenv("DATABRICKS_TOKEN")
    )
    
    return connection


def query_databricks(connection, sql_query):
    """
    Execute SQL query and return pandas DataFrame.
    
    Args:
        connection: Databricks connection object
        sql_query: SQL query string
    
    Returns:
        pandas DataFrame with query results
    """
    cursor = connection.cursor()
    cursor.execute(sql_query)
    
    # Fetch results
    result = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]
    
    cursor.close()
    
    # Convert to DataFrame
    df = pd.DataFrame(result, columns=columns)
    
    return df


