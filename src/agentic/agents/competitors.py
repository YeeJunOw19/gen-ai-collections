
import yaml
from pathlib import Path
from openai import OpenAI
from src.fine_tuning.llama_object import llama_instruct
from src.env_vars import OPEN_AI_KEY

CONFIG_FILE = Path(__file__).joinpath("..", "config.yaml").resolve()
CONFIG = yaml.safe_load(open(CONFIG_FILE, mode="r"))["Competitors_Configurations"]


def _prompt_message(question: str, config_key: str) -> dict:
    # Read in the prompt configurations
    prompt_filename = CONFIG["Prompts_Script"]
    prompt_file = Path(__file__).joinpath("..", "..", "..", "external_scripts", prompt_filename).resolve()
    prompt_config = yaml.safe_load(open(prompt_file, mode="r"))[config_key]

    # Create the system prompt
    system_prompts = prompt_config["System_Prompt"]["Prompt"]
    separator = prompt_config["System_Prompt"]["Separator"]
    system_prompt = ("\n" if separator == "New Line" else " ").join(system_prompts)

    # Create base user prompt
    user_prompts = prompt_config["User_Prompt"]["Prompt"]
    separator = prompt_config["User_Prompt"]["Separator"]
    user_prompt = ("\n" if separator == "New Line" else " ").join(user_prompts) + f"\nQuestion: {question}"

    return {"system_prompt": system_prompt, "user_prompt": user_prompt}


def competitor_generation(question: str, competitor_name: str) -> str:
    # Get the location of the model
    if not CONFIG[competitor_name]["Remote_Model"]:
        model_folder = Path(__file__).joinpath("..", "..", "..", "..", "data_dump", "fine_tuned_models").resolve()
        model_path = Path(model_folder).joinpath(CONFIG[competitor_name]["Model_Name"]).resolve().__str__()
    else:
        model_path = CONFIG[competitor_name]["Model_Name"]

    # Create a Llama model object
    llama = llama_instruct.LlamaInstruct(model_path)

    # Create the prompt and run it through Llama
    message_prompt = _prompt_message(question, "Llama_Instruct_Prompts")
    output = llama.llama_answering(message_prompt["system_prompt"], message_prompt["user_prompt"])

    return output


def code_extractor(answer: str) -> str:
    # Create OpenAI instance
    client = OpenAI(api_key=OPEN_AI_KEY)

    # Create the prompt message and send it to OpenAI
    grading_model = CONFIG["Extraction_Model"]
    message_prompt = _prompt_message(answer, "Llama_Code_Extraction_Prompts")
    output = client.chat.completions.create(
        model=grading_model,
        messages=[
            {"role": "system", "content": message_prompt["system_prompt"]},
            {"role": "user", "content": message_prompt["user_prompt"] + f"\nAnswer: {answer}"}
        ]
    )

    return output.choices[0].message.content
