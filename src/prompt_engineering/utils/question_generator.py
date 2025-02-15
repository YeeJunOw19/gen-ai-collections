
import polars as pl
import random
from sqlalchemy.engine import Engine
from src.data_ingestion.mdutils import motherduck_setup


class QuestionGenerator:

    def __init__(self, duck_engine: Engine, database_schema: str, table_name: str, n: int = 300, seed: int = 3825):
        self.duck_engine = duck_engine
        self.database_schema = database_schema
        self.table_name = table_name
        self.n = n
        self.seed = seed
        self.random_ids = self._random_id_generator()

    def _random_id_generator(self) -> list[int]:
        """
        Generate a list of random ids within the existing ids in MotherDuck table.

        :return: A list of random ids
        """

        # From MotherDuck get a list of available IDs
        query_string = f'SELECT Id FROM "{self.database_schema}".{self.table_name}'
        ids = (
            motherduck_setup.md_read_table(
                duck_engine=self.duck_engine, md_schema=self.database_schema, md_table=self.table_name,
                keep_columns=None, custom_query=query_string
            )
            .collect()
            .to_series()
            .to_list()
        )

        # From the list, randomly pick the number of questions that the user has specified
        random.seed(self.seed)
        random_ids = random.sample(ids, self.n)

        return random_ids

    def qa_generator(self, answer_schema: str, answer_table: str) -> dict:
        """
        Using the list of random ids, get a list of questions and answers corresponding to those ids. Two additional
        parameters are required to be passed in because class instantiation assume query happens on base table.

        :param answer_schema: Specify the schema of the answer key table in MotherDuck
        :param answer_table: Specify the table name of the answer key table in MotherDuck
        :return: A dictionary containing Polars dataframes for questions and answers
        """

        # From MotherDuck, get the questions and answers and save them into a dictionary
        question_string = f'SELECT Id, QuestionAsked FROM "{self.database_schema}".{self.table_name}'
        answer_string = f'SELECT Id, ExtractedAnswer FROM "{answer_schema}".{answer_table}'

        query_df = (
            motherduck_setup.md_read_table(
                duck_engine=self.duck_engine, md_schema=self.database_schema, md_table=self.table_name,
                keep_columns=None, custom_query=question_string
            )
            .filter(pl.col("Id").is_in(self.random_ids))
        )

        answer_df = (
            motherduck_setup.md_read_table(
                duck_engine=self.duck_engine, md_schema=answer_schema, md_table=answer_table,
                keep_columns=None, custom_query=answer_string
            )
            .filter(pl.col("Id").is_in(self.random_ids))
        )

        return {"questions": query_df, "answers": answer_df}
