
import polars as pl
from pathlib import Path
from src.fine_tuning.llama_object import llama_instruct, llama_tune
from src.fine_tuning.lora.hf_dataset_generator import generate_hf_data

REMOTE_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"
SYSTEM_PROMPT = (
    f"""You are a Python coding expert and a helpful coding tutor.
    Your task is to answer Python coding questions accurately and clearly.
    """
)
USER_PROMPT = (
    f"""Solve this problem using Python coding language.
    Write a function in Python to print the Fibonacci series.
    """
)


def pre_fine_tune() -> None:
    # Create an object of the LLM model
    llama = llama_instruct.LlamaInstruct(REMOTE_MODEL)

    # Run the prompt through the model for testing
    prompt_output = llama.llama_answering(SYSTEM_PROMPT, USER_PROMPT)
    print("\n ================ \n Prompt Output:")
    print(prompt_output)


def model_fine_tune() -> None:
    # Read in the training data
    data_file = Path(__file__).joinpath("..", "data", "training_data.parquet").resolve()
    df = pl.read_parquet(data_file)

    # Generate LoRA training dataset
    df_tuning = generate_hf_data(df)
    output_folder = Path(__file__).joinpath("..", "data").resolve()

    # Tune the model
    llama = llama_tune.LlamaTune(model_name=REMOTE_MODEL)
    llama.print_trainable_parameters()

    print("Staring Fine Tuning...")
    llama.model_fine_tuning(
        seed=0, training_data=df_tuning, folder_name="models", model_name="SmolLM2-360M-Instruct-Python",
        warmup_steps=1, max_steps=20, output_folder=output_folder
    )
    print("Completed Fine Tuning...")


if __name__ == "__main__":
    pre_fine_tune()
    model_fine_tune()
