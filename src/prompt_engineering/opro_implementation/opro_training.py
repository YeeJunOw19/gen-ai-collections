
import yaml
import os
import asyncio
import logging
from openai import AsyncOpenAI, OpenAI
from pathlib import Path
from src.prompt_engineering.opro_implementation import opro_data_object as to, opro_openai_client
from src.prompt_engineering.utils import openai_utils

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)

OPEN_AI_KEY = os.environ.get("OPENAI_API_KEY")
PROMPT_CONFIG = yaml.safe_load(open(Path(__file__).joinpath("..", "config.yaml").resolve(), mode="r"))["OPRO_Meta_Prompts"]
OPEN_AI_CONFIG = yaml.safe_load(open(Path(__file__).joinpath("..", "config.yaml").resolve(), mode="r"))["OpenAI_Configurations"]
OPRO_CONFIG = yaml.safe_load(open(Path(__file__).joinpath("..", "config.yaml").resolve(), mode="r"))["OPRO_Meta_Prompts"]


def _text_formatter(concat_method: str, texts: list[str]) -> str:
    if concat_method == "Same Line":
        return " ".join(texts)
    else:
        return "\n".join(texts)


def _list_sorter(list1: list, list2: list) -> tuple:
    combined = list(zip(list1, list2))
    sorted_combined = sorted(combined, key=lambda x: x[0], reverse=True)
    sorted_list1, sorted_list2 = zip(*sorted_combined)
    sorted_list1 = list(sorted_list1)
    sorted_list2 = list(sorted_list2)

    return sorted_list1, sorted_list2


async def opro_training() -> dict:
    # Create an instance of MathQuestion object and pull in all the data needed
    math_data = to.MathQuestionData(seed=OPEN_AI_CONFIG["Training_Seed"], n=OPEN_AI_CONFIG["Training_Samples"])
    question_ids = math_data.generated_data["question_ids"]
    questions_list = math_data.generated_data["questions_list"]
    answer_df = math_data.generated_data["answer_df"]
    logging.info("OPRO Training: Acquired training data from MotherDuck.")

    # Create instances of OpenAI objects
    sync_client = OpenAI(api_key=OPEN_AI_KEY)
    async_client = AsyncOpenAI(api_key=OPEN_AI_KEY)
    logging.info("OPRO Training: Set up clients for OpenAI API calls.")

    # Initialize OPRO contexts and OPRO scoring
    opro_contexts = []
    opro_scores = []

    # Initialize all meta prompts and context that will not change throughout the entire run
    starting_meta = _text_formatter(
        concat_method=OPRO_CONFIG["Starting_Meta_Instruction"]["Concatenation"],
        texts=OPRO_CONFIG["Starting_Meta_Instruction"]["Prompts"],
    )

    problem_meta = _text_formatter(
        concat_method=OPRO_CONFIG["Problem_Statement_Instruction"]["Concatenation"],
        texts=OPRO_CONFIG["Problem_Statement_Instruction"]["Prompts"]
    )

    text_description = _text_formatter(
        concat_method=OPRO_CONFIG["Examples_Statements"]["Concatenation"],
        texts=OPRO_CONFIG["Examples_Statements"]["Prompts"]
    )

    request_prompt = _text_formatter(
        concat_method=OPRO_CONFIG["Prompt_Generation"]["Concatenation"],
        texts=OPRO_CONFIG["Prompt_Generation"]["Prompts"]
    )

    evaluation_prompt = _text_formatter(
        concat_method=OPRO_CONFIG["Evaluation_Prompt"]["Concatenation"],
        texts=OPRO_CONFIG["Evaluation_Prompt"]["Prompts"]
    )
    logging.info("OPRO Training: Created baseline prompt strings for OPRO algorithm.")

    # Create a loop to loop through OPRO algorithm and preform prompt optimization
    for i in range(0, OPEN_AI_CONFIG["Max_Run"]):
        logging.info(f"OPRO Training: Initiating training loop {i + 1}.")

        # Concatenate all meta-instructions, descriptions, Q&A pairs, and historical results into a prompting string
        if not opro_contexts:
            prompt_string = "\n".join([starting_meta, problem_meta, text_description, request_prompt])

        else:
            # Create sorted lists of OPRO before moving forward
            opro_scores_sorted, opro_context_sorted = _list_sorter(opro_scores, opro_contexts)

            solution_score = [f"Sentence: {value}\nScore: {str(opro_scores_sorted[idx])}" for idx, value in enumerate(opro_context_sorted)]
            solution_score_pairs = "\n".join(solution_score)
            prompt_string = "\n".join([starting_meta, solution_score_pairs, problem_meta, request_prompt])

        # Call OpenAI API and get suggested prompts to be used next and combined each one of them with the evaluation prompts
        opro_response = opro_openai_client.opro_prompt_output(
            client=sync_client, prompt_string=prompt_string,
            model=OPEN_AI_CONFIG["OpenAI_Model"], temperature=OPEN_AI_CONFIG["OpenAI_Temperature"]
        )
        system_prompts = ["\n".join([evaluation_prompt, x]) for x in opro_response]

        # With the list of question, asynchronously send them to OpenAI and get answers back
        current_contexts = []
        current_scores = []
        for idx, system_prompt in enumerate(system_prompts):
            tasks = [
                opro_openai_client.opro_prompt_evaluation(
                    client=async_client, system_prompt_string=system_prompt, user_prompt_string=prompt,
                    temperature=OPEN_AI_CONFIG["OpenAI_Temperature"], model=OPEN_AI_CONFIG["OpenAI_Model"]
                )
                for prompt in questions_list
            ]
            results = await asyncio.gather(*tasks)

            # From the answers provided, parse out the actual numerical answer and evaluate the prompt accuracy
            answer_keys = openai_utils.answer_extractor(results)
            accuracy = math_data.opro_accuracy_evaluation(answer_keys)

            # Save the prompt and its respective accuracy
            current_contexts.append(opro_response[idx])
            current_scores.append(accuracy)

        # Sort both lists and only save the top prompt into the running lists
        sorted_contexts, sorted_scores = _list_sorter(current_contexts, current_scores)
        opro_contexts.append(sorted_contexts[0])
        opro_scores.append(sorted_scores[0])

        logging.info(f"OPRO Training: Completed training loop {i + 1} with highest score: {sorted_scores[0]}.")

    logging.info(f"OPRO Training: Completed OPRO training process and returning OPRO training results.")
    final_sorted_scores, final_sorted_context = _list_sorter(opro_scores, opro_contexts)
    return {"opro_contexts": final_sorted_context, "opro_scores": final_sorted_scores}
