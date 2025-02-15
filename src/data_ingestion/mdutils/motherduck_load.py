
import pyarrow as pa
import polars as pl
import duckdb

PYARROW_TYPE_CONVERSION = {"String": pa.string(), "Integer": pa.int64(), "Float": pa.float64()}


class ArrowTableLoadingBuffer:

    def __init__(
        self, duck_conn: duckdb.DuckDBPyConnection, data_schema: list[dict],  md_schema: str,
        md_table: str, chunk_size: int = 100_000, delete_before_load: bool = True, custom_delete_string: str = None
    ):
        self.duck_conn = duck_conn
        self.data_schema = data_schema
        self.md_schema = md_schema
        self.md_table = md_table
        self.chunk_size = chunk_size
        self.delete_before_load = delete_before_load
        self.custom_delete_string = custom_delete_string
        self.pyarrow_schema = self._pyarrow_schema_generation()

    def motherduck_insert(self, df: pl.LazyFrame) -> None:
        """
        Method to perform MotherDuck insert operation by chunk. The default chunk size is 100_000.

        :param df: Polars LazyFrame object from prior workflows.
        :return: None
        """

        pa_table = df.collect().to_arrow()
        total_rows = pa_table.num_rows

        # Execute deletion before insert new data if set to True
        if self.delete_before_load:
            self._delete_ops()

        # Insert the data into MotherDuck in chunks
        for batch_start in range(0, total_rows, self.chunk_size):
            batch_end = min(batch_start + self.chunk_size, total_rows)

            # Chunk pyarrow table and perform insertion to MotherDuck
            pa_chunk = pa_table.slice(batch_start, batch_end - batch_start)
            self._insert_ops(pa_chunk)

    def _pyarrow_schema_generation(self) -> pa.Schema:
        """
        Private method to generate pyarrow schema object using provided table schema.

        :return: Pyarrow schema object.
        """

        columns = [
            pa.field(schema["Column_Name"], PYARROW_TYPE_CONVERSION.get(schema["Column_Type"], pa.string()))
            for schema in self.data_schema
        ]
        return pa.schema(columns)

    def _insert_ops(self, insert_chunk: pa.Table) -> None:
        """
        Private method to perform MotherDuck insert operation, by using the pyarrow object chunk provided.

        :param insert_chunk: Pyarrow chunked object
        :return: None
        """

        self.duck_conn.register("buffer_table", insert_chunk)
        insert_string = f'INSERT INTO "{self.md_schema}".{self.md_table} SELECT * FROM buffer_table'
        self.duck_conn.execute(insert_string)

    def _delete_ops(self) -> None:
        """
        Private method to clear data from MotherDuck Table. User can provide custom delete string to delete data out
        from MotherDuck table.

        :return: None
        """

        # If a custom delete string is provided execute the deletion string
        if self.custom_delete_string:
            try:
                self.duck_conn.execute(self.custom_delete_string)
            except:
                raise Exception("Custom delete string is failed.")

        else:
            try:
                delete_string = f'DELETE FROM "{self.md_schema}".{self.md_table}'
                self.duck_conn.execute(delete_string)
            except:
                raise Exception("Failed to delete data from the specified table.")
