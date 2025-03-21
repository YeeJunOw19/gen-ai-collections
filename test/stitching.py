
import os
from pathlib import Path
from src.agentic.agentic_workflow import advanced_rag_quality

RETRY_LIMIT = 50
DATA_DUMP = Path(__file__).joinpath("..", "..", "data_dump", "advanced_agentic_tests").resolve()
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
    basic_agent = advanced_rag_quality.basic_agentic_workflow().compile()
    advanced_agent = advanced_rag_quality.advanced_agents_workflow().compile()

    # Loop through the test questions and run through the workflow and save the results
    for idx, value in enumerate(TEST_QUESTIONS):
        basic_output = basic_agent.invoke(
            {"question": value, "competitor_name": "Base_Model", "rag": False},
            config={"recursion_limit": RETRY_LIMIT}
        )
        basic_output_rag = basic_agent.invoke(
            {"question": value, "competitor_name": "Base_Model", "rag": True},
            config={"recursion_limit": RETRY_LIMIT}
        )
        advanced_output = basic_agent.invoke(
            {"question": value, "competitor_name": "Base_Model", "rag": True},
            config={"recursion_limit": RETRY_LIMIT}
        )
        advanced_output_ft = basic_agent.invoke(
            {"question": value, "competitor_name": "Competitor2", "rag": True},
            config={"recursion_limit": RETRY_LIMIT}
        )

        # Write the Python code and quality evaluation to a file
        with open(DATA_DUMP.joinpath(f"agentic_test_{idx + 1}.txt"), mode="w") as f:
            f.write(f"Basic SmolLM Output\n")
            f.write("----------------------")
            f.write(f"\n{basic_output['answer']}\n")

            f.write(f"\nBasic SmolLM Quality\n")
            f.write("----------------------")
            f.write(f"\n{basic_output['quality']}\n")

            f.write(f"\nBasic SmolLM with RAG Output\n")
            f.write("----------------------")
            f.write(f"\n{basic_output_rag['answer']}\n")

            f.write(f"\nBasic SmolLM with RAG Output Quality\n")
            f.write("----------------------")
            f.write(f"\n{basic_output_rag['quality']}\n")

            f.write(f"\nAdvanced Agentic RAG SmolLM Output\n")
            f.write("----------------------")
            f.write(f"\n{advanced_output['answer']}\n")

            f.write(f"\nAdvanced Agentic RAG SmolLM Quality\n")
            f.write("----------------------")
            f.write(f"\n{advanced_output['quality']}\n")

            f.write(f"\nAdvanced Agentic RAG Fine-tuned SmolLM Output\n")
            f.write("----------------------")
            f.write(f"\n{advanced_output_ft['answer']}\n")

            f.write(f"\nAdvanced Agentic RAG Fine-tuned SmolLM Quality\n")
            f.write("----------------------")
            f.write(f"\n{advanced_output_ft['quality']}\n")




if __name__ == "__main__":
    agentic_unit_test()
