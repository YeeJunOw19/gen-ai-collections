
from langgraph.graph import END, StateGraph, START
from typing_extensions import TypedDict
from src.agentic.agents import competitors, graders


class GraphState(TypedDict):
    question: str
    answer: str
    competitor_name: str
    quality: str


# --------------------------------------------------------------------------------------------------------------------
# Nodes Creation
# --------------------------------------------------------------------------------------------------------------------

def code_generation(state):
    print("----------------- Generating Code  -----------------")
    question = state["question"]
    competitor_name = state["competitor_name"]

    answer = competitors.competitor_generation(question, competitor_name)
    return {"answer": answer}


def code_extraction(state):
    print("----------------- Extracting Code  -----------------")
    answer = state["answer"]

    answer = competitors.code_extractor(answer)
    return {"answer": answer}


def code_relevance(state):
    print("----------------- Evaluating Code Relevance  -----------------")
    question = state["question"]
    answer = state["answer"]

    relevance = graders.answer_relevancy(question, answer)
    relevance_bool = relevance.relevance

    # Check if the python code is relevant to the question asked
    return "relevant" if relevance_bool else "irrelevant"


def code_quality(state):
    print("----------------- Evaluating Code Quality  -----------------")
    answer = state["answer"]

    quality = graders.answer_grader(answer)
    return {"quality": quality}


# --------------------------------------------------------------------------------------------------------------------
# LangGraph Connections
# --------------------------------------------------------------------------------------------------------------------

def agents_workflow():
    # Initialize the workflow
    workflow = StateGraph(GraphState)

    # Define all the nodes for this workflow
    workflow.add_node("code_generation", code_generation)
    workflow.add_node("code_extraction", code_extraction)
    workflow.add_node("code_relevance", code_relevance)
    workflow.add_node("code_quality", code_quality)

    # Build out the workflow graph
    workflow.add_edge(START, "code_generation")
    workflow.add_edge("code_generation", "code_extraction")
    workflow.add_conditional_edges(
        source="code_extraction",
        path=code_relevance,
        path_map={
            "relevant": "code_quality",
            "irrelevant": "code_generation",
        }
    )
    workflow.add_edge("code_quality", END)

    return workflow
