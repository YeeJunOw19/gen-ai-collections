
import yaml
import os
from pathlib import Path
from multiprocessing.pool import ThreadPool
from src.data_ingestion.mdutils import motherduck_setup
from src.fine_tuning.llama_object import llama_instruct

CONFIG = yaml.safe_load(open(Path(__file__).joinpath("..", "config.yaml").resolve(), mode="r"))
DATA_LOCATION = Path(__file__).joinpath("..", "..", "..", "..", "data_dump").resolve()
PROMPT_LOCATION = Path(__file__).joinpath("..", "..", "..", "external_scripts").resolve()


def _concurrent_qa(system_prompt: str, answer_prompt: str, model_name: str) -> str:
    # Create an instance of LLama model
    llama = llama_instruct.LlamaInstruct(model_name)

    # Pass in prompts and get answer back from Llama
    answer = llama.llama_answering(system_prompt, answer_prompt)
    return answer


def evaluation_run(run_key: str, question_lists: list | None = None, mode: str = "remote") -> list[str]:
    config = CONFIG[run_key]

    # Get the save file location and make sure that the folder exists. If not, create one
    SAVE_LOC = DATA_LOCATION.joinpath(config["Data_Save_Location"]).resolve()
    os.makedirs(SAVE_LOC, exist_ok=True)

    if not question_lists:
        # Set up MotherDuck instance and query a list of questions from MotherDuck
        md = motherduck_setup.MotherDucking(config["MotherDuck_Database"], True)
        schema_name = config["MotherDuck_Schema"]
        table_name = config["MotherDuck_Table"]
        md_seed = config["Seed"]
        top_n = config["Top_N"]
        run_mode = config["Run_Mode"]

        query_string = (
            f"""
            SELECT setseed({md_seed});
            SELECT * 
            FROM "{schema_name}".{table_name} 
            WHERE DataSplit = '{run_mode}' AND QuestionInput = '' AND Id IN (67, 92, 181, 516, 7708)
            ORDER BY RANDOM() 
            LIMIT {top_n};
            """
        )
        question_lists = (
            motherduck_setup.md_read_table(
                duck_engine=md.duckdb_engine, md_schema=schema_name, md_table=table_name,
                keep_columns=None, custom_query=query_string
            )
            .select("QuestionAsked")
            .collect()
            .to_series()
            .to_list()
        )

    # Create the system prompt and user prompts from the list above
    prompt_config = yaml.safe_load(open(PROMPT_LOCATION.joinpath(config["Llama_Instruct_Prompt_Script"]).resolve(), mode="r"))["Llama_Instruct_Prompts"]
    system_config = prompt_config["System_Prompt"]
    user_config = prompt_config["User_Prompt"]

    system_prompts = system_config["Prompt"]
    system_prompt = "\n".join(system_prompts) if system_config["Separator"] == "New Line" else " ".join(system_prompts)

    user_prompts = user_config["Prompt"]
    user_prompt = "\n".join(user_prompts) if user_config["Separator"] == "New Line" else " ".join(user_prompts)

    # Create parameters for concurrency
    prompt_questions = [user_prompt + "\n" + question for question in question_lists]
    system_lists = [system_prompt] * len(prompt_questions)

    # Get the model from either remote repository or local repository
    if mode == "remote":
        model_names = [config["LLama_Model_Name"]] * len(prompt_questions)

    else:
        model_repo = Path(__file__).joinpath("..", "..", "..", "..", "data_dump", "fine_tuned_models").resolve()
        model_path = model_repo.joinpath(config["LLama_Model_Name"]).resolve().__str__()
        model_names = [model_path] * len(prompt_questions)

    with ThreadPool(processes=1) as pool:
        args = list(zip(system_lists, prompt_questions, model_names))
        results = pool.starmap(_concurrent_qa, args)

    # Save the data into a folder for review
    qa_list = []
    for idx, value in enumerate(results):
        write_string = f"Question: {prompt_questions[idx]}\n===============\nAnswer: {value}\n\n"
        qa_list.append(write_string)
        with open(SAVE_LOC.joinpath(f"Question_{idx+1}.txt"), "w") as file:
            file.write(write_string)

    return qa_list
