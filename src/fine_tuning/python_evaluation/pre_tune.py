
import random
from src.fine_tuning.python_evaluation import llama_concurrency


def main_pre_tuned() -> None:
    # Define configuration key and run the Q&A process on pre-tuned LLM
    config_key = "Baseline_Python_Question_Answer"
    qa = llama_concurrency.evaluation_run(config_key)

    # Print out two results randomly
    random_qa = random.choice(qa)
    for idx, value in enumerate(random_qa):
        print(f"{idx}: The Python question and answer can is as below.\n")
        print(f"**************")
        print(f"{value}\n")


if __name__ == "__main__":
    main_pre_tuned()
