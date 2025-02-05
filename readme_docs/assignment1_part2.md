## Assignment 1 Part 2

### Introduction

This assignment showcases the use of RAG by including relevant context into LLM prompt, enriching 
and improving the answers from LLM. Five different questions are sent to ChatGPT 3.5 Turbo and requesting
for an answer, but 4 different methods are being compared to see the accuracy of the answer. These methods are:
- **Using only LLM**: Sending questions directly to LLM
- **Simple RAG Setup**: Sending a question, along with context from latest news to enrich LLM response
- **RAG with HyDE**: Using LLM to generate a hypothetical answer and use that to get the context from latest news
- **RAG with Reranking**: Similar to simple RAG setup, but before sending the context to LLM, reranking is done
on the context itself to extract only the most similar context to the original question

Before we jump into the meat of this assignment, I have made some improvements from Assignment 1 Part 1 that
I would like to call out beforehand.

---

### Prerequisites

1. The *requirements.txt* file will have all the Python packages to be installed before running the code in this
repository.
2. It is recommended to use Python 3.11+ for running code in this repository.

Because the all data is not saved locally in this repository, and the workflow / code do not create copies of data into any
local machines, these are the environment variables and API keys that you will need if you want to run the code in this repository.
If you send me a request for any of these keys, I am more than happy to generate a new one for you and have you run it on your
machine locally.

1. **PINECONE_API_KEY** - API Key to authenticate and login into Pinecone
2. **MOTHERDUCK_TOKEN** - MotherDuck database login and authentication token
3. **HUGGING_FACE_API** - Hugging Face authentication token
4. **OPENAI_API_KEY** - OpenAI API key

The link to the dataset that I am using in this assignment can be found here: https://huggingface.co/datasets/heegyu/news-category-dataset.

---

### Lessons Learned

The workflow, data, and structure of the project from Assignment 1 Part 1 is less than ideal to me. It gets
the job done but the tool choice and setup are not what I would ultimately go for if I were to build a somewhat
decent LLM application. I have summarized these into "lessons learned" points that I want to fix and improve on
in this assignment.

1. Unstable Turso Database
   - Turso is built on the fork of SQLite, i.e. libSQL, and it is still in early development stage
   - Turso is also a good solution as a *edge database*, not an analytical database
   - While Turso has always been a database tool that I want to try out, using it in Assignment 1 Part 1 proves to be 
   a bad decision due to the above mentioned reasons
   - Turso has announced a complete re-write of its service in Rust, so it will have to be a future project to use this
   service


2. Unsuitable Data
   - In Assignment 1 Part 1, the data being used is coming from a Kaggle dataset for Sentiment and Emotional Analysis
   - While this dataset is a great use case for LLM, it does not serve a greater purpose down the road when we are trying to
   incorporate things like RAG and fine-tuning


3. Not so well-designed Workflow
   - The intention in Assignment 1 Part 1 is to create a sound workflow in ingesting data, performing vector embeddings, 
   and saving those vectors into Pinecone
   - The intention is also to showcase some orchestration tools and capabilities in the early stage of the process where
   data processing is essential
   - Handling the kinks and limitations in Turso database had taken up a lot of time and a bandage solution was implemented
   to handle data processing

---

### Improvements Built Upon Lessons Learned!

To tackle each of the aforementioned lessons learned, several steps have been taken in this assignment to address them.

1. The Introduction of MotherDuck
   - MotherDuck is an analytical data warehouse solution, build on top of DuckDB, a state-of-the-art in-process OLAP database
   management system
   - In the past, I had great success in using DuckDB and I have always wanted to use MotherDuck but could not find a use
   case until now
   - With MotherDuck, all data is truly stored in a cloud environment and no data is being stored locally on my PC


2. News Data from Hugging Face
   - I have pulled a new dataset from Hugging Face that is related to news headlines and news short description from Hugging Face
   - This dataset will be more suitable for this class, especially going into fine-tuning tasks and RAG
   - On top of that, Polars (an alternative to Pandas) has direct integration with Hugging Face, allowing the creation of a
   true "zero-copy workflow"


3. The Introduction of Dagster
    - Dagster is an alternative orchestration tool to Airflow, and unlike Airflow, it focuses on the star of any analytical
   workflows, which is *Data Assets*
    - In the past, I had great success with Dagster as an orchestration tool as well, so I wanted to introduce the fun into
   this project as well
    - With the addition of Dagster, the code structure has been restructured to follow a modular format, rather than pure Python
   scripts format

There are some other improvements made since Part 1 but the above three points are the primary improvements made to the project.

---

### Architecture of Assignment 1

Because I do not save data locally in my personal PC and all data lives in either an analytical database or a vector database
in the cloud, I will be hard to picture the current process and the current architecture. So, I have prepared a small architectural
diagram, summarizing Assignment 1 Part 1.

![Assignment 1 Architecture](/images/Assignment1_Part1_Architecture.drawio.png)

There are two Dagster workflows that you can run, and the nice thing about Dagster is that it is a CLI based Python package
so at the top level of this repository, these are the two commands that are needed to run Workflow 1 and Workflow 2, respectively.

*Workflow 1*
```console
dagster job execute -m src -j run_vector_embedding
```

*Workflow 2*
```console
dagster job execute -m src -j run_raw_data_save
```

Below is an example of output from running Workflow 2, signifying the successful execution of the entire workflow.

![Example Output](/images/Example_WF2_Output.png)

---

### RAG Implementation and Testing

In the module `rag_pipeline`, there are two Python scripts, one for custom retriever and formatter to retrieve context from
MotherDuck and format it into a usable format that ChatGPT can consume. The other script, `rag_chaining.py` has all the functionality
to generate the four RAG pipelines and produce an answer from ChatGPT, described in the introduction section.

In the `test` folder, there is a Jupyter Notebook named `assignment1_part2_testing.ipynb` with all the code and results
for Assignment 1 Part 2. This file can also serves as a basis on how to utilize the modularized RAG pipeline, and apply different
functionality to different prompt questions.

To recap, here are the four tests that have been done on the five sample questions:

1. Using only LLM
   - Directly input the question into ChatGPT without providing any additional enrichment context


2. Simple RAG Setup
   - Using the question prompted by user, search the vector database for the top 5 most similar headlines and news descriptions
   and use those as context to enrich ChatGPT answer
   - Both question and context will be used as prompt in ChatGPT


3. RAG with HyDE
   - Taking the original question asked, use ChatGPT to generate a hypothetical answer to that question
   - Then, taking that answer and search the vector database for top 5 most similar documents and use that as context for ChatGPT
   - Both question and enriched context will be used as prompt in ChatGPT


4. RAG with Reranking
   - Using the original question prompted, search the vector database for the top 10 most similar documents
   - Then from those 10 documents, `cross-encoder/ms-marco-MiniLM-L-6-v` is used to rerank them and the top 5 highest scoring
   documents are picked for forming the context for prompting
   - Both question and enriched context will be used as prompt in ChatGPT

Below is a sample output of the running the 4 functions above. A full version of the output are printed out in the notebook.

![Example Output](/images/Assignment1_Part2_Sample.png)

---

### Conclusion

In general, method 2 and method 3 performed similarly and both performed better than method 1. Since context is provided to enrich
the prompt, ChatGPT produces answers that are much more closer to recent events and news. Finally, method 4 performed the best
compared to the other three methods. The nature of my questions revolve around getting the latest information and latest updates,
so reranking performed the best makes the most sense.

**More details in the notebook!**
