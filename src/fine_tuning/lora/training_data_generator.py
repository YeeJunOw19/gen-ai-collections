
import yaml
from pathlib import Path
from datasets import Dataset
from src.data_ingestion.mdutils import motherduck_setup
from src.fine_tuning.llama_object import llama_instruct

YAML_FILE = Path(__file__).joinpath("..", "config.yaml").resolve()
CONFIG = yaml.safe_load(open(YAML_FILE, mode="r"))["LoRA_Fine_Tuning_Configurations"]


def generate_training_data() -> Dataset:
    # Set up MotherDuck and get data
    md = motherduck_setup.MotherDucking(CONFIG["MotherDuck_Database"], True)

    sql_string = (
        f"""
        SELECT setseed({CONFIG["Seed"]});
        SELECT * 
        FROM "{CONFIG["MotherDuck_Schema"]}".{CONFIG["MotherDuck_Table"]} 
        WHERE DataSplit = '{CONFIG["Run_Mode"]}' AND QuestionInput = ''
        ORDER BY RANDOM() 
        LIMIT {CONFIG["Top_N"]};
        """
    )
    df = motherduck_setup.md_read_table(
        duck_engine=md.duckdb_engine, md_schema=CONFIG["MotherDuck_Schema"], md_table=CONFIG["MotherDuck_Table"],
        keep_columns=None, custom_query=sql_string
    ).collect()

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
