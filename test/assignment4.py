
import os
from pathlib import Path
from src.agentic.agentic_workflow import python_qa_quality

RETRY_LIMIT = 50
DATA_DUMP = Path(__file__).joinpath("..", "..", "data_dump", "agentic_tests").resolve()
TEST_QUESTIONS = [
    "Write a Python function to calculate the sum of all integers in a list.",
    "Write a Python function to fit a logistic regression model to a dataset.",
    "Write a Python function to calculate average of a list of numbers.",
    "Write a Python function to sort a list of integers in ascending order.",
    "Write a Python function to get the maximum value from a list of numbers."
]


def agentic_unit_test() -> None:
    # Create the folder to store test results if not exists
    os.makedirs(DATA_DUMP, exist_ok=True)

    # Compile the LangGraph and prepare to be used
    agent = python_qa_quality.agents_workflow().compile()

    # Loop through the test questions and run through the workflow and save the results
    for idx, value in enumerate(TEST_QUESTIONS):
        llama_output = agent.invoke(
            {"question": value, "competitor_name": "Competitor1"},
            config={"recursion_limit": RETRY_LIMIT}
        )
        smole_output = agent.invoke(
            {"question": value, "competitor_name": "Competitor2"},
            config={"recursion_limit": RETRY_LIMIT}
        )

        # Write the Python code and quality evaluation to a file
        with open(DATA_DUMP.joinpath(f"agentic_test_{idx + 1}.txt"), mode="w") as f:
            f.write(f"Llama Python Output\n")
            f.write("----------------------")
            f.write(f"\n{llama_output['answer']}\n")

            f.write(f"\nLlama Python Quality\n")
            f.write("----------------------")
            f.write(f"\n{llama_output['quality']}\n")

            f.write(f"\nSmol Python Output\n")
            f.write("----------------------")
            f.write(f"\n{smole_output['answer']}\n")

            f.write(f"\nSmol Python Quality\n")
            f.write("----------------------")
            f.write(f"\n{smole_output['quality']}\n")


if __name__ == "__main__":
    agentic_unit_test()
