
import asyncio
from openai import AsyncOpenAI
from dagster import asset, AssetIn
from src.prompt_engineering.opro_implementation.opro_training import OPEN_AI_KEY, OPEN_AI_CONFIG, OPRO_CONFIG
from src.prompt_engineering.opro_implementation import opro_training, opro_data_object as to, opro_openai_client
from src.prompt_engineering.utils import openai_utils


@asset(ins={"training_results": AssetIn(key="opro_training")})
async def opro_testing(training_results: dict):
    # Create an instance of MathQuestion object and pull in all the data needed
    math_data = to.MathQuestionData(seed=OPEN_AI_CONFIG["Testing_Seed"], n=OPEN_AI_CONFIG["Testing_Samples"])
    question_ids = math_data.generated_data["question_ids"]
    questions_list = math_data.generated_data["questions_list"]
    answer_df = math_data.generated_data["answer_df"]
    client = AsyncOpenAI(api_key=OPEN_AI_KEY)

    # Create the system prompt and adding the highest scoring result from training
    evaluation_prompt = opro_training.text_formatter(
        concat_method=OPRO_CONFIG["Evaluation_Prompt"]["Concatenation"],
        texts=OPRO_CONFIG["Evaluation_Prompt"]["Prompts"]
    )
    system_prompt = "\n".join([evaluation_prompt, training_results["opro_contexts"][0]])

    # Using this system prompt run through all the questions and score the answers
    tasks = [
        opro_openai_client.opro_prompt_evaluation(
            client=client, system_prompt_string=system_prompt, user_prompt_string=prompt,
            temperature=OPEN_AI_CONFIG["OpenAI_Temperature"], model=OPEN_AI_CONFIG["OpenAI_Model"]
        )
        for prompt in questions_list
    ]
    results = await asyncio.gather(*tasks)
    answer_keys = openai_utils.answer_extractor(results)

    # Calculate the accuracy of the prompt and return a dataframe for data modeling and ingestion
    df = openai_utils.scoring_function(
        ids=question_ids,
        openai_answers=answer_keys,
        scoring_df=answer_df,
        matching_cols=["RowId", "Id"],
        model_name=OPEN_AI_CONFIG["OpenAI_Model"],
        model_temperature=OPEN_AI_CONFIG["OpenAI_Temperature"],
        run_method=OPEN_AI_CONFIG["Prompt_Engineering_Method"],
    )

    return {"df": df, "config": OPEN_AI_CONFIG}
