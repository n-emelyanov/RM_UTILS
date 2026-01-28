import os
from typing import Callable, Any
import sqlparse
import sqlalchemy
from sshtunnel import SSHTunnelForwarder
import clickhouse_connect

import pandas as pd
from abc import  ABC, abstractmethod



class BaseSQLConnector(ABC):
    """Базовый класс для всех коннекторов к БД"""

    @abstractmethod
    def execute_multiple_query(self, query: str) -> None:
        pass

    @abstractmethod
    def dataframe_from_sql(self, query: str) -> pd.DataFrame:
        pass

    @abstractmethod
    def dataframe_to_sql(self, data: pd.DataFrame, **kwargs) -> None:
        pass


class SQLQueryProcessor:
    """Обработка SQL запросов"""

    @staticmethod
    def parse_sql_file(query: str) -> list[str]:
        """Парсит SQL файл или строку на отдельные команды"""
        if os.path.exists(query):
            with open(query, "r") as sql:
                query = sql.read()

        return [cmd.strip() for cmd in sqlparse.split(query) if cmd.strip()]


class DataFrameUtils:
    """Утилиты для работы с DataFrame"""

    @staticmethod
    def fix_dataframe_dtypes(data: pd.DataFrame) -> pd.DataFrame:
        """Автоматическое преобразование неподдерживаемых типов данных"""
        data = data.copy()

        for column in data.columns:
            dtype = str(data[column].dtype)

            if dtype.startswith('uint'):
                if dtype == 'uint8':
                    data[column] = data[column].astype('int16')
                elif dtype == 'uint16':
                    data[column] = data[column].astype('int32')
                elif dtype == 'uint32':
                    data[column] = data[column].astype('int64')
                elif dtype == 'uint64':
                    if data[column].max() > 2**63 - 1:
                        data[column] = data[column].astype('float64')
                    else:
                        data[column] = data[column].astype('int64')
            elif dtype == 'bool':
                data[column] = data[column].astype('boolean')
            elif 'datetime64[ns]' in dtype:
                data[column] = pd.to_datetime(data[column])

        return data


class PostgreSQLConnector(BaseSQLConnector):
    """
    Класс для подключения к базе данных ClickHouse.

    Parameters
    ----------
    credentials : dict
        Словарь, содержащий:
         {username: Имя пользователя,
          password: Пароль}

    Examples
    --------
    >>>> credentials = {
                            "username": "your_username",
                            "password": "your_password",
                        }
    >>>> p_db_conn = Postgre_DB_Connector(credentials)
    >>>> sql = "select * from some_table"
    >>>> data = p_db_conn.from_sql_to_dataframe(sql)
    """


    def __init__(self, credentials):
        self.credentials = credentials
        self.engine = self._create_engine()

    def _create_engine(self) -> sqlalchemy.engine.Engine:
        """Создает SQLAlchemy engine"""
        return sqlalchemy.create_engine(
            f"postgresql+psycopg2"
            f"://{self.credentials['username']}"
            f":{self.credentials['password']}"
            f"@10.20.31.61"
            f":5432"
            f"/paymart_superset",
            pool_pre_ping=True,
            pool_recycle=3600,
            echo=False,
        )

    def connection_test(self) -> bool:
        """Тестирование подключения к БД"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(sqlalchemy.text("SELECT 1 as test"))
                row = result.fetchone()
                return row.test == 1
        except Exception:
            return False


    def execute_multiple_query(self, query: str, verbose: bool = False) -> None:
        """Выполняет множественные sql-запросы"""
        sql_commands = SQLQueryProcessor.parse_sql_file(query)

        with self.engine.connect() as connection:
            for command in sql_commands:
                if verbose:
                    print("Execute: ", command)
                connection.execute(sqlalchemy.text(command))
                connection.commit()


    def dataframe_from_sql(self, query: str) -> pd.DataFrame:
        """Выполняет sql-запрос и возвращает DataFrame."""
        return pd.read_sql_query(query, con=self.engine)


    def dataframe_to_sql(self,
                        data: pd.DataFrame,
                        table_name: str,
                        schema: str,
                        index: bool = False,
                        if_exists: str = 'replace',
                        **kwargs) -> None:
        """Сохраняет DataFrame в таблицу в БД"""
        data = DataFrameUtils.fix_dataframe_dtypes(data)

        data.to_sql(
            name=table_name,
            con=self.engine,
            schema=schema,
            if_exists=if_exists,
            index=index,
            **kwargs
        )

    def get_engine(self):
        """Получить SQLAlchemy engine"""
        return self.engine


class ClickSQLConnector(BaseSQLConnector):
    """
    Класс для подключения к базе данных ClickHouse.

    Parameters
    ----------
    credentials : dict
        Словарь, содержащий:
         {username: Имя пользователя,
          password: Пароль
          ssh_pkey: Путь до ключа ssh}

    Examples
    --------
    >>>> credentials = {
                            "username": "your_username",
                            "password": "your_password",
                            "ssh_pkey": "/../.ssh/your_private_ssh_key",
                        }
    >>>> db_conn = DB_Connector(credentials)
    >>>> sql = "select * from some_table"
    >>>> data = db_conn.from_sql_to_dataframe(sql)
    """

    # Маппинг типов данных
    TYPE_MAPPING = {
        'int64': 'Int64', 'int32': 'Int32', 'int16': 'Int16', 'int8': 'Int8',
        'uint64': 'UInt64', 'uint32': 'UInt32', 'uint16': 'UInt16', 'uint8': 'UInt8',
        'float64': 'Float64', 'float32': 'Float32',
        'bool': 'Bool', 'boolean': 'Bool',
        'object': 'String', 'category': 'String',
        'datetime64[ns]': 'DateTime', 'datetime64[ms]': 'DateTime',
        'datetime64[us]': 'DateTime', 'datetime64[s]': 'DateTime',
        'date': 'Date', 'timestamp': 'DateTime',
    }

    def __init__(self, credentials: dict):
        self.credentials = credentials
        self.ssh_config = {
            "ssh_address_or_host": ("jumphost.udata.uz", 22),
            "ssh_username": credentials["username"],
            "ssh_pkey": credentials["ssh_pkey"],
            "remote_bind_address": ("clickhouse.data.uz", 8123),
            "local_bind_address": ("127.0.0.1", 0),
        }

    def _execute_with_tunnel(self, operation: Callable) -> Any:
        """Выполняет операции через SSH-туннель"""
        with SSHTunnelForwarder(**self.ssh_config) as tunnel:
            with clickhouse_connect.get_client(
                host=tunnel.local_bind_address[0],
                port=tunnel.local_bind_port,
                username=self.credentials["username"],
                password=self.credentials["password"],
            ) as client:
                return operation(client)

    def _create_table(self, client, table_name: str, columns_with_types: list) -> None:
        """Создает таблицу в ClickHouse"""
        columns_definition = ", ".join(f"`{col}` {typ}" for col, typ in columns_with_types)
        create_table_query = f"CREATE TABLE IF NOT EXISTS {table_name} ({columns_definition}) ENGINE = MergeTree() ORDER BY tuple()"
        client.query(create_table_query)

    def _prepare_data_for_clickhouse(self, data: pd.DataFrame) -> pd.DataFrame:
        """Подготавливает DataFrame для загрузки в ClickHouse"""
        data = data.copy()

        # Преобразуем period типы в строки
        for col in data.columns:
            dtype_str = str(data[col].dtype)
            if 'period' in dtype_str:
                data[col] = data[col].astype(str)
                print(f"Столбец {col} имел тип 'period', преобразован в String")

        # Обработка строковых колонок
        for col in data.columns:
            if data[col].dtype == 'object':
                data[col] = data[col].astype(str).replace(['nan', 'None', '<NA>'], '')

        # Замена NaN/None
        return data.where(pd.notnull(data), None)


    def connection_test(self) -> bool:
        """Тестирование подключения к БД"""
        try:
            result = self._execute_with_tunnel(lambda client: client.query("SELECT 1 as test"))
            return True
        except Exception:
            return False

    def execute_multiple_query(self, query: str, verbose: bool = False) -> None:
        """Выполняет множественные sql-запросы"""
        sql_commands = SQLQueryProcessor.parse_sql_file(query)

        for command in sql_commands:
            if verbose:
                print("Execute: ", command)
            self._execute_with_tunnel(lambda client: client.query(command))

    def dataframe_from_sql(self, query: str) -> pd.DataFrame:
        """Выполняет sql-запрос и возвращает DataFrame"""
        return self._execute_with_tunnel(lambda client: client.query_df(query))

    def dataframe_to_sql(self,
                        data: pd.DataFrame,
                        schema: str,
                        table_name: str,
                        batch_size: int = 10000,
                        if_exists: str = 'drop') -> None:
        """Загружает DataFrame в ClickHouse"""
        table_name = f"{schema}.{table_name}"

        # Определяем типы колонок для ClickHouse
        columns_with_types = []
        for col, dtype in zip(data.columns, data.dtypes):
            ch_type = self.TYPE_MAPPING.get(str(dtype), 'String')
            columns_with_types.append((col, ch_type))

        # Проверяем неизвестные типы
        unique_dtypes = set(str(dtype) for dtype in data.dtypes)
        missing_types = unique_dtypes - set(self.TYPE_MAPPING.keys())
        if missing_types:
            print(f"Предупреждение: неизвестные типы {missing_types}, будут преобразованы в String")

        # Удаляем таблицу если нужно
        if if_exists in ['drop', 'replace']:
            self._execute_with_tunnel(lambda client: client.query(f"DROP TABLE IF EXISTS {table_name}"))
            self._execute_with_tunnel(lambda client: self._create_table(client, table_name, columns_with_types))

        # Подготавливаем данные
        data = self._prepare_data_for_clickhouse(data)

        # Загрузка данных батчами
        for i in range(0, data.shape[0], batch_size):
            chunk = data.iloc[i:i+batch_size]
            print(f"Загружается батч {i//batch_size + 1}, строк: {chunk.shape[0]}")

            self._execute_with_tunnel(lambda client: client.insert(
                table_name,
                chunk.values.tolist(),
                column_names=chunk.columns.tolist()
            ))
