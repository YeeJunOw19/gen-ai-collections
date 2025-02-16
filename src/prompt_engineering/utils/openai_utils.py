
import asyncio
import re
import backoff
import openai
import polars as pl
from datetime import datetime


def answer_style_parser(questions: list[str], prompt_style: str = "basic") -> list[str]:
    # For each of the questions, add how OpenAI should style its answers
    parsed_questions = []
    for question in questions:
        if prompt_style == "basic":
            temp_str = f'{question} -- Please output only the final numerical number at the end with the prefix ####. For example: ####[ANSWER]'
            parsed_questions.append(temp_str)

        else:
            parsed_questions.append(question)

    return parsed_questions


def answer_extractor(answers: list[str]) -> list[int | float]:
    # Create a new list that only contains the correct answer output
    numerical_answers = []
    for answer in answers:
        try:
            num_ans = answer.split("####")[-1]
        except IndexError:
            num_ans = answer

        num_ans = num_ans.strip()
        numerical_answers.append(num_ans)

    # Replace all non-numerical character with ""
    cleaned_answers = []
    for answer in numerical_answers:
        x = re.sub(r"[a-zA-Z$%/'=+, ]", "", answer).strip()
        x = re.sub(r"\.$", "", x)
        x = re.sub(r"-$", "", x)

        # If there is somehow a line split character from OpenAI, split it and take the last number in the list
        line_split = x.split("\n")
        x = line_split[-1]

        # Convert the string into number
        if "." in x:
            number = float(x)
        else:
            number = int(x)

        cleaned_answers.append(number)

    return cleaned_answers


@backoff.on_exception(backoff.expo, openai.RateLimitError)
async def chat_completion(
    openai_client, prompt: str, model: str, temperature: float,
    prompt_style: str = "basic", role_input: str | None = None, examples_input: str | None = None,
) -> str:
    # Create a base prompt message and incrementally add to the list
    messages = [{"role": "user", "content": prompt}]

    # Based on the different prompting strategy, create different messages
    if prompt_style == "Role-based Prompting":
        new_addition = {"role": "system", "content": role_input}
        messages.append(new_addition)

    elif prompt_style == "Chain of Thought Prompting":
        concat_content = role_input + "\n" + prompt
        new_addition = {"role": "system", "content": concat_content}
        messages.append(new_addition)

    # Pass in the message into OpenAI API
    response = await openai_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )

    return response.choices[0].message.content


def scoring_function(
    ids: list[int], openai_answers: list[float], scoring_df: pl.LazyFrame, matching_cols: list[str],
    model_name: str, model_temperature: float, run_method: str
) -> pl.DataFrame:
    # Create a base dataframe from the results of OpenAI
    df = pl.DataFrame(
        data={"RowId": ids, "OpenAIAnswer": openai_answers},
        schema={"RowId": pl.Int64, "OpenAIAnswer": pl.Float64},
    )

    # Join the scoring dataframe and perform scoring and metadata coding
    df = (
        df
        .join(scoring_df.collect(), left_on=matching_cols[0], right_on=matching_cols[1], how="inner")
        .with_columns(
            pl.col("OpenAIAnswer").cast(pl.Float64).alias("OpenAIAnswer"),
            pl.lit(model_name).alias("ModelName"),
            pl.lit(model_temperature).alias("ModelTemperature"),
            pl.lit(run_method).alias("RunStyle"),
        )
        .with_columns(
            pl.when((pl.col("OpenAIAnswer") - pl.col("ExtractedAnswer")) == 0).then(1).otherwise(0).alias("IndCorrect"),
            pl.lit(datetime.now().date()).alias("RunDate")
        )
    )

    return df
