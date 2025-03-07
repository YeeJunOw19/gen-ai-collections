
from pathlib import Path
from src.fine_tuning.python_evaluation import llama_concurrency

TEST_QUESTIONS = Path(__file__).joinpath("..", "..", "..", "..", "data_dump", "pre_fine_tuning").resolve()


def post_tune() -> None:
    # Define configuration key and run the Q&A process on pre-tuned LLM
    config_key = "Fine_Tuned_Python_Question_Answer"

    # Read in all the previous questions and form a list
    questions = []
    for file in TEST_QUESTIONS.glob("*.txt"):
        with file.open(mode="r", encoding="utf-8") as f:
            content = f.read()
            q = content.split("===============")[0]
            q = q.replace("Question: ", "")
            questions.append(q)

    qa = llama_concurrency.evaluation_run(config_key, questions, mode="Local")

    # Print out two results randomly
    for idx in range(0, 2):
        print(f"{idx}: The Python question and answer can is as below.\n")
        print(f"**************")
        print(f"{qa[idx]}\n")


if __name__ == "__main__":
    post_tune()




