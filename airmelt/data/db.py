from functools import reduce
import pandas as pd
import pyodbc
from typing import Dict
import pymysql
import sqlparse
import mysql.connector
from sqlalchemy import create_engine
from sqlalchemy.engine.url import URL
import logging
import csv

from airmelt_data import gcp
from airmelt_system import toolbox


SUPPRTED_DB_TYPES = ["bigquery", "mssql", "mysql"]


def create_mysql_engine(dbname, prod=True, driver="pymysql"):
    """
    Creates and returns a MySQL database engine.

    Args:
        dbname (str): The name of the database.
        prod (bool, optional): Whether to use production settings. Defaults to True.
        driver (str, optional): The name of the database driver to use. Defaults to 'pymysql'.

    Returns:
        sqlalchemy.engine.Engine: The database engine.
    """
    db_config = toolbox.open_system_config(prod=prod, config_type="DB_CONFIG")[dbname]
    db_url = URL(
        drivername="mysql+{}".format(driver),
        username=db_config.get("username"),
        password=db_config.get("password"),
        host=db_config.get("hostname"),
        database=db_config.get("db"),
    )
    engine = create_engine(db_url)
    return engine


def create_mssql_connection(
    dbname, prod=True, driver="{ODBC Driver 17 for SQL Server}", driver_type="pyodbc"
):
    """
    Creates and returns a connection to a Microsoft SQL Server database.

    Args:
        dbname (str): The name of the database.
        prod (bool, optional): Whether to use production settings. Defaults to True.
        driver (str, optional): The name of the ODBC driver to use. Defaults to "{ODBC Driver 17 for SQL Server}".
        driver_type (str, optional): The type of driver to use. Defaults to "pyodbc".

    Returns:
        pyodbc.Connection: The database connection.
    """
    db_config = toolbox.open_system_config(prod=prod, config_type="DB_CONFIG")[dbname]
    if driver_type == "pyodbc":
        connection = pyodbc.connect(
            driver=driver,
            server=db_config["server"],
            database=db_config["database"],
            uid=db_config["username"],
            pwd=db_config["password"],
        )
    else:
        raise ValueError("Invalid driver type: {}".format(driver_type))
    return connection


def create_pymysql_connection(
    dbname: str, prod: bool = True, **kwargs
) -> pymysql.connections.Connection:
    """
    Returns a connection object to a MySQL database.

    Args:
        dbname (str): The name of the database to connect to.
        prod (bool): Whether to use production database configuration or not.
        **kwargs: Additional parameters to pass to pymysql.connect().

    Returns:
        A connection object to the specified database.
    """
    logger = logging.getLogger(__name__)
    db_config: Dict = toolbox.open_system_config(prod=prod, config_type="DB_CONFIG")[
        dbname
    ]
    conn: pymysql.connections.Connection = None
    try:
        conn = pymysql.connect(
            host=db_config.get("hostname"),
            port=db_config.get("port"),
            user=db_config.get("username"),
            passwd=db_config.get("password"),
            db=db_config.get("db"),
            **kwargs,
        )
        return conn
    except pymysql.Error:
        pass
    try:
        conn = mysql.connector.connect(
            host=db_config.get("hostname"),
            user=db_config.get("username"),
            password=db_config.get("password"),
        )
        return conn
    except mysql.connector.Error:
        logger.error(
            "Error connecting using mysql.connector to database: \n host: {host}; user: {user}; db: {db}".format(
                host=db_config.get("hostname"),
                user=db_config.get("username"),
                db=db_config.get("db"),
            )
        )
        raise


def execute_query(dbEngine, database, query, rowCnt=1000000, prod=True, **kwargs):
    batchList = []
    if dbEngine == "mysql":
        connection = create_pymysql_connection(database, prod)
        cursor = connection.cursor()
        cursor.execute(query)
        while True:
            batch = cursor.fetchmany(rowCnt)
            if batch:
                batchList.extend(batch)
            if not batch:
                break
        return batchList, [column[0] for column in cursor.description]
    elif dbEngine == "mssql":
        connection = create_mssql_connection(database, prod)
        cursor = connection.cursor()
        cursor.execute(query)
        while True:
            batch = cursor.fetchmany(rowCnt)
            if batch:
                batchList.extend(batch)
            if not batch:
                break
        return batchList, [column[0] for column in cursor.description]


class Database(object):
    def __init__(self, db_type: str, **kwargs) -> None:
        """
        db_type: Database types, as appear in SUPPRTED_DB_TYPES
        dbname: Database name, optional (required for MySQL and MSSQL)
        """
        if db_type not in SUPPRTED_DB_TYPES:
            raise ValueError(
                f"Invalid db_type. Supported types are: {', '.join(SUPPRTED_DB_TYPES)}"
            )
        self.db_type = db_type
        if self.db_type == "bigquery":
            if "project_id" in kwargs:
                bq_project_id = kwargs.get("project_id")
            elif "bq_project_id" in kwargs:
                bq_project_id = kwargs.get("bq_project_id")
            else:
                raise ValueError("project_id is required for bigquery db_type")

            self.client = gcp.Bigquery(bq_project_id=bq_project_id)
        elif self.db_type == "mssql":
            if "dbname" in kwargs:
                self.dbname = kwargs.get("dbname")
            else:
                raise ValueError("dbname is required for mssql db_type")
        elif self.db_type == "mysql":
            if "dbname" in kwargs:
                self.dbname = kwargs.get("dbname")
            else:
                raise ValueError("dbname is required for mysql db_type")


class Query(Database):
    def __init__(self, db_type: str, query_string: str, **kwargs) -> None:
        """
        db_type: Database types, as appear in SUPPRTED_DB_TYPES
        query_string: SQL Query
        """
        super().__init__(db_type, **kwargs)
        if not sqlparse.parse(query_string):
            raise ValueError("Invalid SQL query")
        self.query_string = query_string
        self.logger = logging.getLogger(__name__)

    def to_dataframe(self) -> pd.DataFrame:
        if self.db_type == "bigquery":
            try:
                result = self.client.query_to_dataframe(self.query_string)
                if result.empty:
                    print("The query returned no results")
                    return None
                return result
            except Exception as e:
                print(f"An error occurred while executing the query: {e}")
                return None
        elif self.db_type in ["mssql", "mysql"]:
            try:
                result, columns = execute_query(
                    database=self.dbname, dbEngine=self.db_type, query=self.query_string
                )
                df = pd.DataFrame(result, columns=columns)
                return df
            except Exception as e:
                self.logger.error("Result: \n {}".format(result))
                self.logger.error("Columns: \n {}".format(columns))
                print(f"An error occurred while executing the query: {e}")
                raise

    def to_csv(self, filename: str, **kwargs) -> None:
        if self.db_type == "bigquery":
            try:
                result, columns = self.client.query2list(self.query_string)
                if result.empty:
                    print("The query returned no results")
                    return None
                return result
            except Exception as e:
                print(f"An error occurred while executing the query: {e}")
                return None
        elif self.db_type in ["mssql", "mysql"]:
            try:
                result, columns = execute_query(
                    database=self.dbname, dbEngine=self.db_type, query=self.query_string
                )
            except Exception as e:
                self.logger.error("An error occurred while executing the query: {e}")
                print(f"An error occurred while executing the query: {e}")
                raise
        with open(filename, "w") as f:
            # using csv.writer method from CSV package
            write = csv.writer(f)

            write.writerow(columns)
            write.writerows(result)
        return len(result)


class QueryList(Database):
    def __init__(self, db_type: str, query_list: list, **kwargs) -> None:
        """
        db_type: Database types, as appear in SUPPRTED_DB_TYPES
        query_list: List SQL queries
        """
        super().__init__(db_type, **kwargs)
        for query in query_list:
            if not sqlparse.parse(query):
                raise ValueError("Invalid SQL query")
        self.query_list = query_list

    def to_dataframes_list(self, **kwargs) -> list:
        df_list = list()
        for query in self.query_list:
            try:
                df_list.append(self.client.query_to_dataframe(query))
            except Exception as e:
                print(f"An error occurred while executing the query: {e}")
                df_list.append(None)
        return df_list

    def to_dataframe(self, key_list: list = None):
        df_list = self.to_dataframes_list()
        if key_list is None:
            raise ValueError("key_list is required for to_dataframe method")
        return reduce(
            lambda x, y: pd.merge(
                x,
                y,
                how="inner",
                left_on=key_list,
                right_on=key_list,
            ),
            df_list,
        )
