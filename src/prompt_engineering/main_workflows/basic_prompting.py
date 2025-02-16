
import os
import yaml
import asyncio
import polars as pl
from dagster import asset, AssetIn
from openai import AsyncOpenAI
from pathlib import Path
from src.prompt_engineering.utils import question_generator, openai_utils
from src.data_ingestion.mdutils import motherduck_setup
from src.prompt_engineering.database_preprocessing import setup_motherduck_tables as st

CONFIG = yaml.safe_load(open(Path(__file__).joinpath("..", "config.yaml").resolve(), mode="r"))["Basic_Prompting_Configurations"]
OPEN_AI_KEY = os.environ.get("OPENAI_API_KEY")


@asset(deps=[st.prompt_engineering_preprocessing])
async def basic_prompting() -> dict:
    """
    Dagster asset to asynchronously run basic prompt engineering from OpenAI API. The inputs and results are saved in
    a dictionary to be used downstream.

    :return: A dictionary containing inputs for downstream assets
    """

    # Get instances of engines required
    md = motherduck_setup.MotherDucking(CONFIG["MotherDuck_Database"], True)
    qg = question_generator.QuestionGenerator(
        duck_engine=md.duckdb_engine,
        database_schema=CONFIG["MotherDuck_Schema"],
        table_name=CONFIG["MotherDuck_Table"],
        n=CONFIG["n"],
        seed=CONFIG["seed"]
    )
    client = AsyncOpenAI(api_key=OPEN_AI_KEY)

    # Get a list of questions and answers for comparison
    qa_dict = qg.qa_generator(CONFIG["Answer_Schema"], CONFIG["Answer_Table"])
    questions_df = qa_dict["questions"].collect()
    ids = questions_df.select("Id").to_series().to_list()
    questions = questions_df.select("QuestionAsked").to_series().to_list()
    questions = openai_utils.answer_style_parser(questions)

    # With the list of question, asynchronously send them to OpenAI and get answers back
    tasks = [
        openai_utils.chat_completion(
            openai_client=client,
            prompt=prompt,
            model=CONFIG["OpenAI_Model"],
            temperature=CONFIG["OpenAI_Temperature"],
        )
        for prompt in questions
    ]
    results = await asyncio.gather(*tasks)
    return {"ids": ids, "questions": questions, "results": results, "answers": qa_dict["answers"]}


@asset(ins={"openai_output": AssetIn(key="basic_prompting")})
def basic_scoring(openai_output):
    """
    Dagster asset that takes in the the output from the previous asset, and calculates the scores from basic level
    prompt engineering.

    :param openai_output: The output dictionary from previous asset
    :return: A Polars dataframe containing the scores and ready for dim-fact modeling
    """

    # From the results, extract the actual answer from OpenAI response
    client_answers = openai_utils.answer_extractor(openai_output["results"])

    # Create a dataframe to score and save the data into MotherDuck as a record
    df = openai_utils.scoring_function(
        ids=openai_output["ids"],
        openai_answers=client_answers,
        scoring_df=openai_output["answers"],
        matching_cols=["RowId", "Id"],
        model_name=CONFIG["OpenAI_Model"],
        model_temperature=CONFIG["OpenAI_Temperature"],
        run_method=CONFIG["Prompt_Engineering_Method"],
    )

    return {"df": df, "config": CONFIG}
