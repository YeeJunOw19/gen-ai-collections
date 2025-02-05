## Sentiment and Emotion Analysis

### Overview
The Sentiment and Emotion Analysis Dataset is a meticulously curated collection of textual 
data designed to empower researchers, data scientists, and NLP enthusiasts to delve into 
the intricacies of human emotions and sentiments embedded in text.
With a blend of large-scale emotional diversity and sentiment categorization, 
this dataset offers a rich playground for building state-of-the-art machine learning and deep learning models.

*Source*: https://www.kaggle.com/datasets/kushagra3204/sentiment-and-emotion-analysis-dataset


### Project Setup

The project is setup with src folder containing the following module. Within each module, 
the submodules representing different workflows are structured as follows.
- *data_ingestion*
  - kaggle_ingestion: Contains workflow to get data from Kaggle through Kaggle API and save the data into a database
  - vector_embedding: Contains workflow for vector embedding of textual data and save vectors to Pinecone


### Prerequisite
To run the code and files in this repository, these are the API keys and environment variables that you will need.
- TURSO_URL: Link to TursoDB that currently contains a saved version of data from Kaggle
- TURSO_TOKEN: TursoDB API Token for SQLAlchemy connection
- TURSO_API_TOKEN: TursoDB Platform API key used for POST and GET requests
- PINECONE_API: Pinecone API key

The *requirements.txt* file will have all the Python packages to be installed before running the code in this
repository. It is recommended to use Python 3.11+ for running code in this repository.


### How to Run
Below are the scripts that can be ran in this repository.

*Note: Future improvement is to incorporate Dagster orchestration into each workflow for more streamlined orchestration.*

- *turso_databasing.py*: This script is only needed to save a version of Kaggle data into Turso cloud database.
Not required for the project to function properly.
- *vector_embedding.py*: This script is to perform vector embedding on text data and save the vectors into Pinecone.
The *_Run_* parameter in the *config.yaml* file is set to `False` to avoid incurring additional cost and runtime, but
if a rerun is required, the *_Run_* parameter will need to be set to `True`.


### Testing Embeddings
To test the embeddings in Pinecone, *vector_testing.ipynb* is created in a test folder. Two sentences have been done in the 
testing file `i am very angry` and `i am glad that I we have a happy conversation`.

A screenshot of the expected result is shown below.
![Testing Sample](/images/Embedding_Testing_Sample.png)

