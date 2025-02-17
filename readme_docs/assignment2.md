## Assignment 2

### Introduction

In this assignment, we will be focusing on prompt engineering (PE). We will be using a dataset consisting of a list of Grade 8th
math problems, along with their solutions. Through prompt engineering, we will measure how effective LLM is able to these problems.
We will be focusing on primarily four prompt engineering techniques listed below.
- **Basic LLM Prompt**: Only involve sending the actual math problem to LLM and only requesting a numerical answer from LLM
- **PE Best Practices**: Before sending the actual math problems, we give LLM more infomation about the role that it should take on
and more context about the problem itself
- **Chain of Thought & Few Short Learning**: Incorporate some examples of how to solve these math problems with step by step
instructions and chain of thought
- **OPRO**: Implementation of automated Optimized Prompt Engineering developed by Google

---

### Prerequisites

1. The *requirements.txt* file will have all the Python packages to be installed before running the code in this
repository.
2. It is recommended to use Python 3.11+ for running code in this repository.

Because all data is not saved locally in this repository, and the workflow / code do not create copies of data into any
local machines, these are the environment variables and API keys that you will need if you want to run the code in this repository.
If you send me a request for any of these keys, I am more than happy to generate a new one for you and have you run it on your
machine locally.

1. **PINECONE_API_KEY** - API Key to authenticate and login into Pinecone
2. **MOTHERDUCK_TOKEN** - MotherDuck database login and authentication token
3. **HUGGING_FACE_API** - Hugging Face authentication token
4. **OPENAI_API_KEY** - OpenAI API key (_Model: gpt-3.5-turbo_)

The link to the dataset that I am using in this assignment can be found here: https://huggingface.co/datasets/openai/gsm8k.

---

### Architecture

The architecture of this assignment can be summarized into four main parts:
1. Data Ingestion into MotherDuck
2. Running each PE method on a subset of data (300 math questions)
3. Evaluating the accuracy of each PE method by comparing the answer from LLM to the ground truth
4. Save results back to MotherDuck

The image below shows a high level overview of the entire workflow of this assignment.

![Assignment 1 Architecture](/images/Assignment2_Architecture.png)

---

### Quick Start

A Docker image has been built for this project.
1. **Image Name**: yeejunow-twy5319
2. **Server**: DeepDish2

> [!IMPORTANT]
> You will require the `Dockerfile` and `docker-compose.yaml` file to start up a Docker container for this image.
> Furthermore, all runs from this Docker image require API Keys for various cloud services.
> Docker has recommended that the best practice to handle secrets in Docker container is to mount temporary files into the container
> that contain API Keys, and will be destroyed when the Docker container is shut down. Read more here: https://docs.docker.com/compose/how-tos/use-secrets/.
> Moreover, using this method, Docker container will have no historical trace to identify these secrets.

To run this assignment, here are the steps.

1. Clone this repository onto your local machine/server (or DeepDish2).
2. In this working directory, you will need to create a `.secrets` folder, and in this folder you will need to create these four files listed below,
with each file containing the API keys for each cloud service. You can reach out to me to get these keys and I am happy to generate
new keys for you. Alternatively, you can also find these keys in my DeepDish2 directory in `assignment-2-YeeJunOw19`.
    - hugging_face_api_key.txt
    - motherduck_api_key.txt
    - openai_api_key.txt
    - pinecone_api_key.txt
3. Run the following command to start up an interactive Docker container.
```shell
docker compose run --rm genai
```

*To Run End-to-End Prompt Engineering Workflow*

As usual, all workflows have been incorporated as a part of Dagster orchestration, this is no different as well. In the interactive
shell in Docker container, run the following command to this end-to-end PE workflow

```shell
dagster job execute -m src -j run_prompt_engineering
```

Below is an example of what you will see in the terminal.

![Example Output](/images/Assignment2_WF_Output.png)

*To Get All Results*

To get the results of each of the PE methods and the date ran for each method, run the code below.

```shell
python -m test.assignment2
```

Below is an example of the output from running the code above.

![Example Output](/images/Assignment2_Results.png)

> [!NOTE]
> Reach out if you have questions about secrets or if you need API Keys generated!

---

### Project Structure

The main module pertaining this assignment is located in the `src` folder, under `prompt_engineering`. In this folder, there are
five sub-modules and each of them are listed below along with the functionality of each module.

1. `database_preprocessing`
   - Contains workflows to setup MotherDuck and make sure that the data warehouse is ready for PE workflows

2. `main_workflows`
   - Contains the code and workflows to run the first three PE methods

3. `opro_implementation`
   - Main module containing the algorithm for OPRO method

4. `data_modeling`
   - Contains the code to format data and send it back to MotherDuck for storage

5. `utils`
   - General utility module containing functions commonly used by many other modules

---

### Summary of Method and Results

All the test that I ran uses gpt-3.5-turbo from OpenAI.

1. Basic LLM Prompt
   - In this method, we do not provide any additional information to the LLM other than the actual math prblem itself
   - We also force the LLM to only return a numerical answer
   - This method, not surprising, consistently is the lowest scoring method amongst the four

2. PE Best Practices
   - Here, we provide more context to the LLM and we asked the LLM to assume a persona before answering the question
   - For example, the LLM is asked to assume he role of a Math professor and its job is to answer 8th Grade math problems
   - We also ask the LLM to reason through his thoughts before providing an answer
   - By implementing PE best practices, we see a drastic improvement in the LLM performance

3. Chain of Thought & Few Short Learning
   - To take it one step further, we give some examples to the LLM of what the questions look like and what the answers are expected to be
   - Then, for each of these examples, we also provide a step-by-step instruction and calculation to show how we solve these problems
   - Through this method, we are able to improve the LLM performance just slightly, but nevertheless, it's still an improvement

4. OPRO
   - This is the method developed by Google's DeepMind
   - This method involves continuously prompting the LLM by providing its previous prompts with an associated accuracy scores
   - The goal here is to have the LLM find another prompt that can get a higher score than the previous ones
   - For this assignment, we limit the OPRO loop to 5
   - This method too can produce high accuracy but not significantly higher than the the other two
   - Below is an example of the prompt given to LLM for OPRO
   
```markdown
Your task is to generate the answer starting sentence. Below are some sentences that have been generated in the past with their corresponding scores.
The score ranges from 0 to 100 with 0 being the lowest score indicating lowest accuracy and 100 being the highest score indicating highest accuracy.
The sentences are arranged in ascending order based on their scores.

Sentence: To solve this problem, we first need to examine
Score: 90

Sentence: Examining this problem reveals
Score: 85.89

(...other prompt-score pairs...)

The following examples show how to apply your generated sentence. You will need to replace <INS> in each input with your generated sentence.
Then, read the input and give an output. We say your output is wrong if your output is different from the given output, and we say your output is correct if they are the same.

Inputs:
Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
A: <INS>
Ground truth answer:
140

(...other question-answer pairs...)

Generate 3 starting sentences that are different from all the old ones above and have as high of a score as possible.
The starting sentences should be concise, effective, and generally applicable to all Q&A pairs above.
An example that you can take as a reference is [Calculating the answer, we get].
Write the sentences in square brackets.
```

---

### Additional Readings

https://arxiv.org/abs/2309.03409

https://aipapersacademy.com/large-language-models-as-optimizers/

https://jrodthoughts.medium.com/meet-opro-google-deepminds-new-method-that-optimizes-prompts-better-than-humans-4b840655b995

https://docs.docker.com/engine/swarm/secrets/

https://medium.com/@Aman-tech/how-make-async-calls-to-openais-api-cfc35f252ebd
