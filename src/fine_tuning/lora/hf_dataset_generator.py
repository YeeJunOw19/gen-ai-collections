
import yaml
import polars as pl
from pathlib import Path
from datasets import Dataset
from src.fine_tuning.llama_object import llama_instruct

YAML_FILE = Path(__file__).joinpath("..", "config.yaml").resolve()
CONFIG = yaml.safe_load(open(YAML_FILE, mode="r"))["LoRA_Fine_Tuning_Configurations"]


def generate_hf_data(df: pl.DataFrame) -> Dataset:
    # Create a Hugging Face dataset of questions and answers
    qa_dict = {
        "question": df.select("QuestionAsked").to_series().to_list(),
        "answer": df.select("OutputAnswer").to_series().to_list()
    }
    training_data = Dataset.from_dict(qa_dict)

    # Create a training dataset using tokenizer
    llama = llama_instruct.LlamaInstruct(CONFIG["LLama_Model_Name"])
    mapped_training = training_data.map(llama.qa_tokenizing, batched=False, remove_columns=['question', 'answer'])

    return mapped_training
