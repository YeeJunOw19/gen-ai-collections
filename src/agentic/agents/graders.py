
import yaml
from openai import OpenAI
from pydantic import BaseModel, Field
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from src.agentic.agents.competitors import CONFIG
from src.env_vars import OPEN_AI_KEY


class AnswerRelevance(BaseModel):
    relevance: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


def _grader_prompt(question: str | None, answer: str, config_key: str) -> dict:
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
    user_prompt = ("\n" if separator == "New Line" else " ").join(user_prompts)

    # Add in original question and generated answer from LLM
    if question is not None:
        user_prompt += f"\nQuestion: {question}\nAnswer: {answer}"
    else:
        user_prompt += f"\n{answer}"

    return {"system_prompt": system_prompt, "user_prompt": user_prompt}


def answer_relevancy(question: str, answer: str) -> AnswerRelevance:
    # Create OpenAI instance
    grading_model = CONFIG["Answer_Grader_Model"]
    client = ChatOpenAI(api_key=OPEN_AI_KEY, model=grading_model, temperature=0)
    structure_output = client.with_structured_output(AnswerRelevance, method="function_calling")

    # Create the prompt message and send it to OpenAI
    message_prompt = _grader_prompt(question, answer, "Llama_Code_Answering_Check_Prompt")
    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", message_prompt["system_prompt"]),
            ("human", message_prompt["user_prompt"])
        ]
    )

    retrieved_answer = grade_prompt | structure_output
    output = retrieved_answer.invoke({})

    return output


def answer_grader(answer: str) -> str:
    # Create OpenAI instance
    client = OpenAI(api_key=OPEN_AI_KEY)

    # Create the prompt message and send it to OpenAI
    grading_model = CONFIG["Answer_Quality_Model"]
    message_prompt = _grader_prompt(None, answer, "Llama_Code_Quality_Prompt")
    output = client.chat.completions.create(
        model=grading_model,
        messages=[
            {"role": "system", "content": message_prompt["system_prompt"]},
            {"role": "user", "content": message_prompt["user_prompt"]}
        ]
    )

    return output.choices[0].message.content
