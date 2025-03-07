
import random
from src.fine_tuning.python_evaluation import llama_concurrency


def main_pre_tuned() -> None:
    # Define configuration key and run the Q&A process on pre-tuned LLM
    config_key = "Baseline_Python_Question_Answer"
    qa = llama_concurrency.evaluation_run(config_key)

    # Print out two results randomly
    for idx in range(0, 2):
        print(f"{idx}: The Python question and answer can is as below.\n")
        print(f"**************")
        print(f"{qa[idx]}\n")


if __name__ == "__main__":
    main_pre_tuned()
