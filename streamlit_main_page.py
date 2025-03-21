
import streamlit as st
from src.agentic.agentic_workflow import advanced_rag_quality

# Setup page configuration
st.set_page_config(page_title="Python Code Evaluation", page_icon=":robot_face:", layout="wide")

# Setup run configuration for LangGraph
RECURSION_LIMIT = 50


# --------------------------------------------------------------------------------------------------------------------
# Application Main Page
# --------------------------------------------------------------------------------------------------------------------

# Title of this main page
st.write("### Evaluating Python Code Generation of Local LLMs :robot_face:")

# Add markdown to describe the page and what is the purpose of this page
st.markdown(
    """
    With the advancement in Large Language Models (LLMs), the ability to generate code is increasingly important. The
    primary concern regarding code generation using cloud LLMs from major providers is the sharing of sensitive data and
    information with third parties that may be untrusted. This lead to an explosion of many pre-trained LLMs that can be
    downloaded locally and used for code generation.

    This in turns sprout a different problem, where the LLMs are used to generate code that is not as high quality. This
    lead to the need for fine-tuning these open-source LLMs to improve their performance. This application aims to compare
    two open-source model: LLama Instruct (base) and SmolLM (fine-tuned), and evaluate the performance of both of these
    models. The code generated from these two models may not be production ready or up to standard, but this serves as a
    baseline for evaluating whether to continue the future work of fine-tuning these models.

    ---
    """
)

# Starting point of the application
st.markdown(
    """
    To start, provide a question that is Python related that you would like to ask the LLMs to solve.
    """
)
input_question = st.text_input("Your Question:")
st.markdown("---")


# --------------------------------------------------------------------------------------------------------------------
# Generate Responses from the two LLM Models
# --------------------------------------------------------------------------------------------------------------------

# Create and compile a LangGraph
if input_question:
    # Compile both basic and advanced RAG graphs
    basic_rag = advanced_rag_quality.basic_agentic_workflow().compile()
    advanced_rag = advanced_rag_quality.advanced_agents_workflow().compile()

    # ----------------------------------------------------------------------------------------------------------------
    # Model 1: Basic LLM without RAG
    # ----------------------------------------------------------------------------------------------------------------

    basic_rag_output = basic_rag.invoke(
        {"question": input_question, "competitor_name": "Base_Model", "rag": False},
        config={"recursion_limit": RECURSION_LIMIT}
    )

    st.write("#### 1.) Basic SmolLM Model")
    st.markdown(
        """
        In this section, we provide the basic SmolLM model with a Python related question, without any additional context.
        There will be no recursion on this section, since we are only asking the model to generate a response based on the
        question we have asked. The code and quality evaluation are as follows.
        """
    )

    st.markdown("_Basic SmolLM Generated Code_")
    st.code(basic_rag_output["answer"], language="python")
    st.markdown("_Basic SmolLM Code Evaluation_")
    st.code(basic_rag_output["quality"], language="markdown")
    st.markdown("---")

    # ----------------------------------------------------------------------------------------------------------------
    # Model 2: Basic LLM with simple RAG
    # ----------------------------------------------------------------------------------------------------------------

    simple_rag_output = basic_rag.invoke(
        {"question": input_question, "competitor_name": "Base_Model", "rag": True},
        config={"recursion_limit": RECURSION_LIMIT}
    )

    st.write("#### 2.) SmolLM Model with Simple RAG")
    st.markdown(
        """
        In this section, we provide the basic SmolLM model with a Python related question, but we have also added a simple
        RAG with context retrieval. There will be no recursion on this section, since we are only asking the model to
        generate a response based on the question we have asked. The code and quality evaluation are as follows.
        """
    )

    st.markdown("_SmolLM with Simple RAG Generated Code_")
    st.code(simple_rag_output["answer"], language="python")
    st.markdown("_SmolLM with Simple RAG Evaluation_")
    st.code(simple_rag_output["quality"], language="markdown")
    st.markdown("---")

    # ----------------------------------------------------------------------------------------------------------------
    # Model 3: Advanced RAG with SmolLM model without Fine Tuning
    # ----------------------------------------------------------------------------------------------------------------

    adv_rag_output = advanced_rag.invoke(
        {"question": input_question, "competitor_name": "Base_Model", "rag": True},
        config={"recursion_limit": RECURSION_LIMIT}
    )

    st.write("#### 3.) SmolLM Model with Advanced Agentic RAG")
    st.markdown(
        f"""
        This section is pretty similar to the previous section, except the only difference is that we used RAG on the
        basic SmolLM model, and with recursion. The recursion involves ChatGPT checking if the code is relevant or not.
        If the code is not relevant, the pipeline will restart go through the process again to answer the user's question.
        The recursion limit is set to {RECURSION_LIMIT}. The code and quality evaluation are as follows.
        """
    )

    st.markdown("_SmolLM with Advanced Agentic RAG Generated Code_")
    st.code(adv_rag_output["answer"], language="python")
    st.markdown("_SmolLM with Advanced Agentic RAG Evaluation_")
    st.code(adv_rag_output["quality"], language="markdown")
    st.markdown("---")

    # ----------------------------------------------------------------------------------------------------------------
    # Model 4: Advanced RAG with Fine-tuned SmolLM model
    # ----------------------------------------------------------------------------------------------------------------

    adv_tuned_output = advanced_rag.invoke(
        {"question": input_question, "competitor_name": "Competitor2", "rag": True},
        config={"recursion_limit": RECURSION_LIMIT}
    )

    st.write("#### 4.) Fine-tuned SmolLM Model with Advanced Agentic RAG")
    st.markdown(
        f"""
        This section is pretty similar to the previous section, except the only difference is that we used RAG on the
        fine-tuned SmolLM model, and with recursion. The recursion involves ChatGPT checking if the code is relevant or not.
        If the code is not relevant, the pipeline will restart go through the process again to answer the user's question.
        The recursion limit is set to {RECURSION_LIMIT}. The code and quality evaluation are as follows.
        """
    )

    st.markdown("_Fine-tuned SmolLM with Advanced Agentic RAG Generated Code_")
    st.code(adv_tuned_output["answer"], language="python")
    st.markdown("_Fine-tuned SmolLM with Advanced Agentic RAG Evaluation_")
    st.code(adv_tuned_output["quality"], language="markdown")
