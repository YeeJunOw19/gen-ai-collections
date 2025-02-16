
import yaml
import polars as pl
from pathlib import Path
from src.data_ingestion.mdutils import motherduck_setup
from src.prompt_engineering.utils import question_generator

DATA_CONFIG = yaml.safe_load(open(Path(__file__).joinpath("..", "config.yaml").resolve(), mode="r"))["MotherDuck_Configurations"]


class MathQuestionData:

    def __init__(self, seed, n):
        self.md_database = DATA_CONFIG["MotherDuck_Database"]
        self.md_schema = DATA_CONFIG["MotherDuck_Schema"]
        self.md_question_table = DATA_CONFIG["Questions_Table"]
        self.md_answer_table = DATA_CONFIG["Answers_Table"]
        self.seed = seed
        self.n = n
        self.md = motherduck_setup.MotherDucking(self.md_database, True)
        self.generated_data = self._generate_questions_answers()

    def _generate_questions_answers(self) -> dict:
        # Create an instance of question generator
        qg = question_generator.QuestionGenerator(
            duck_engine=self.md.duckdb_engine, database_schema=self.md_schema,
            table_name=self.md_question_table, n=self.n, seed=self.seed
        )

        # Get a list of questions and answers for comparison
        question_ans = qg.qa_generator(self.md_schema, self.md_answer_table)

        # Create a list for questions and another list for ids associated with those questions
        question_df = question_ans["questions"].collect()
        ids = question_df.select("Id").to_series().to_list()
        questions = question_df.select("QuestionAsked").to_series().to_list()

        # Extract the Lazyframe for answers out of the dictionary
        answer_df = question_ans["answers"]

        return {"question_ids": ids, "questions_list": questions, "answer_df": answer_df}

    def opro_accuracy_evaluation(self, answer_keys: list[int | float]) -> float:
        # Join both OpenAI data and the actual answers into a dataframe and evaluate if the answers are the same
        openai_df = (
            pl.DataFrame(
                data={"Id": self.generated_data["question_ids"], "OpenAIAnswer": answer_keys},
                schema={"Id": pl.Int64, "OpenAIAnswer": pl.Float64}
            )
            .join(self.generated_data["answer_df"].collect(), on="Id", how="inner")
            .with_columns(pl.col("ExtractedAnswer").cast(pl.Float64).alias("ExtractedAnswer"))
            .with_columns(pl.when((pl.col("OpenAIAnswer") - pl.col("ExtractedAnswer") == 0)).then(1).otherwise(0).alias("IndCorrect"))
        )

        # Calculate accuracy
        total_rows = openai_df.height
        correct_rows = openai_df.filter(pl.col("IndCorrect") == 1).height
        accuracy = (correct_rows / total_rows) * 100

        return accuracy
