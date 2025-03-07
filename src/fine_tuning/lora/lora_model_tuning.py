
import yaml
from pathlib import Path
from src.data_ingestion.mdutils import motherduck_setup
from src.fine_tuning.llama_object import llama_tune
from src.fine_tuning.lora.hf_dataset_generator import generate_hf_data

YAML_FILE = Path(__file__).joinpath("..", "config.yaml").resolve()
CONFIG = yaml.safe_load(open(YAML_FILE, mode="r"))["LoRA_Fine_Tuning_Configurations"]


def lora_main() -> None:
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

    # Generate dataset for tuning
    tuning_data = generate_hf_data(df)

    # Create Llama object and prepare for tuning
    llama = llama_tune.LlamaTune(
        model_name=CONFIG["LLama_Model_Name"], r=CONFIG["LoRA_r"],
        alpha=CONFIG["LoRA_Alpha"], dropout=CONFIG["LoRA_Dropout"]
    )

    # Print the trainable parameters and start LoRA fine-tuning
    llama.print_trainable_parameters()
    llama.model_fine_tuning(
        seed=CONFIG["LoRA_Seed"], training_data=tuning_data,
        folder_name=CONFIG["LoRA_Model_Save"], model_name=CONFIG["LoRA_Model_Name"],
        warmup_steps=CONFIG["LoRA_WarmupSteps"], max_steps=CONFIG["LoRA_Max_Steps"]
    )


if __name__ == "__main__":
    lora_main()
