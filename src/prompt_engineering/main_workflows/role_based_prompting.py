
import yaml
import os
import asyncio
from pathlib import Path
from openai import AsyncOpenAI
from dagster import asset, AssetIn
from src.data_ingestion.mdutils import motherduck_setup
from src.prompt_engineering.utils import question_generator, openai_utils
from src.prompt_engineering.main_workflows import basic_prompting
from src.env_vars import OPEN_AI_KEY

CONFIG = yaml.safe_load(open(Path(__file__).joinpath("..", "config.yaml").resolve(), mode="r"))["Role_Based_Prompting_Configurations"]
EXTERNAL_SCRIPTS = Path(__file__).joinpath("..", "..", "..", "external_scripts").resolve()


@asset(deps=[basic_prompting.basic_prompting])
async def role_prompting() -> dict:
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
    questions = openai_utils.answer_style_parser(questions, prompt_style=CONFIG["Prompt_Engineering_Method"])

    # Read in the system prompt and combine the entire prompt into one string
    system_prompt =  EXTERNAL_SCRIPTS.joinpath(CONFIG["Context_Template"]).resolve()
    with open(system_prompt, mode="r") as f:
        content = f.read().split("\n")
        combined = " ".join(content)

    # With the list of question, asynchronously send them to OpenAI and get answers back
    tasks = [
        openai_utils.chat_completion(
            openai_client=client,
            prompt=prompt,
            model=CONFIG["OpenAI_Model"],
            temperature=CONFIG["OpenAI_Temperature"],
            prompt_style=CONFIG["Prompt_Engineering_Method"],
            role_input=combined
        )
        for prompt in questions
    ]
    results = await asyncio.gather(*tasks)
    return {"ids": ids, "questions": questions, "results": results, "answers": qa_dict["answers"]}


@asset(ins={"openai_output": AssetIn(key="role_prompting")})
def role_scoring(openai_output):
    """
    Dagster asset that takes in the output from the previous asset, and calculates the scores from basic level
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
