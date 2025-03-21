# Final Project

### Introduction

In this assignment, we will be focusing on Agentic AI. With the advancement in Large Language Models (LLMs), the ability to generate code is increasingly important.
The primary concern regarding code generation using cloud LLMs from major providers is the sharing of sensitive data and
information with third parties that may be untrusted. This lead to an explosion of many pre-trained LLMs that can be
downloaded locally and used for code generation.

This in turns sprout a different problem, where the LLMs are used to generate code that is not as high quality. This
lead to the need for fine-tuning these open-source LLMs to improve their performance. This project is similar to Assignment 4
but instead of comparing two different open-source models, we are comparing different techniques applied to the same open-source
model and evaluate the Python code quality of each method.

The model that we will be using is SmolLM model and below are the 4 different techniques that we will be applying.

1. Basic SmolLM Model
2. Basic SmolLM Model with basic RAG
3. Advanced Agentic RAG with basic SmolLM Model
4. Advanced Agentic RAG with fine-tuned SmolLM Model

---

### Prerequisite

1. The *requirements.txt* file will have all the Python packages to be installed before running the code in this
repository.
2. It is recommended to use Python 3.11+ for running code in this repository.
3. API keys to all the relevant services. These will be shared through Slack. Alternatively, you can acquire these secrets from
a folder called .secrets in my workspace in DeepDish, within the `stitching-project-YeeJunOw19` folder.

> [!NOTE]
> The fine-tuned model and same samples output can be found here in this shared drive.
> https://drive.google.com/drive/folders/1AZU-9-ZwoTEU9UC1Q10yz9pADgNlmY28?usp=drive_link
> 
> The Docker Image is built in DeepDish 4 with the name `yeejunow-twy5319`.

---

### Quick Start

This assignment consists of two parts: 1.) Streamlit Web Application, and 2.) Unit Testing Run. To setup the project for both
functionalities to run, follow the steps below.

*Step 1*:</br>
Clone the remote repository into your local machine.
```shell
git clone git@github.com:NUMLDS/stitching-project-YeeJunOw19.git
cd stitching-project-YeeJunOw19
```

*Step 2*:</br>
Copy the `.secrets` folder into `stitching-project-YeeJunOw19`. You should have 5 files containing different API keys for different
cloud provider and services.

*Step 3*:</br>
To be able to use the fine-tuned model, you will need to setup the appropriate folder in the project folder.
```shell
mkdir data_dump
```
Save the folder `SmolLM2-360M-Instruct-Python` (shared through the Google Drive link above) into the newly created folder.
The structure should look like this `stitching-project-YeeJunOw19/data_dump/SmolLM2-360M-Instruct-Python`.

---

### Running Streamlit Application

The Streamlit application will serve as a entry point for you to test out the functionalities of this Agent AI service.
To be able to use the Streamlit application properly, you will need to follow the steps listed below.

*Step 1*:</br>
SSH into DeepDish 4, and make sure to include a parameter for SSH tunneling. This is needed because Streamlit application will
require a UI to serve the application, and SSH tunneling will allow you to access the application on your local computer's web browser.
Replace the `<your-net-id>` with your actual NetID.
```shell
ssh -L 3825:localhost:3825 <your-net-id>@mlds-deepdish4.ads.northwestern.edu
```

*Step 2*:</br>
Once you are logged into DeepDish4, follow the steps above to setup the project if you have not done so already.

*Step 3*:</br>
Start up a Docker container in detached mode.
```shell
docker compose up -d --force-recreate --remove-orphans
```

*Step 4*:</br>
Run the Streamlit application from outside the Docker container so that this will be available on your local PC as well.
```shell
docker exec -it genai streamlit run streamlit_main_page.py --server.port 3825 --server.address 0.0.0.0
```

Once all these are done, you should be able to access the Streamlit application from your favorite browser from this address.
`http://localhost:3825/`

---

### Streamlit Application

The Streamlit application will look exactly the same as the application from Assignment 4.

![EC2 Specs](/images/Streamlit_Main_Page.png)

Then, instead of having two distinct sections, I have split the page into multiple different sections, focusing on evaluating the
Python Code and Code Quality. Below is a screenshot of one of the section that you will see in the application. The application
takes about 1 minute to run through all 4 workflows for a given question.

![EC2 Specs](/images/Streamlit_Final_Example.png)

---

### Discussion

Five different test were ran on this project and each of the question sent to the LLM is listed below.

- Write a Python function to calculate the sum of all integers in a list.
- Write a Python function to fit a logistic regression model to a dataset.
- Write a Python function to calculate average of a list of numbers.
- Write a Python function to sort a list of integers in ascending order.
- Write a Python function to get the maximum value from a list of numbers.

_Test 1: Summing a List of Integers_</br>
- Basic SmolLM & Basic SmolLM with RAG: Simple and correct implementations using sum(), along with an explicit loop version.
- Advanced Agentic RAG SmolLM: Introduced redundant function definitions, violating the DRY principle.
- Advanced Agentic RAG Fine-tuned SmolLM: Had excessive repetition of identical functions and test cases.

_Test 2: Logistic Regression Implementation_</br>
- Basic SmolLM: Incorrectly attempted logistic regression using np.polyfit, which is meant for polynomial regression.
- Basic SmolLM with RAG: Used LogisticRegression() but incorrectly handled multi-class targets.
- Advanced Agentic RAG SmolLM & Fine-tuned SmolLM: Repeated similar mistakes but improved maintainability.

_Test 3: Calculating the Average of a List_</br>
- Basic SmolLM & Basic SmolLM with RAG: Correct implementation, but failed to handle empty lists (leading to ZeroDivisionError).
- Advanced Agentic RAG SmolLM: Maintained correctness but lacked error handling.
- Advanced Agentic RAG Fine-tuned SmolLM: Added redundant test cases instead of improving error handling.

_Test 4: Sorting a List_</br>
- Basic SmolLM: Used sorted(), which is correct but lacked input validation.
- Basic SmolLM with RAG: Introduced redundant and unnecessary function definitions.
- Advanced Agentic RAG SmolLM: Maintained correctness but didn’t introduce significant improvements.
- Advanced Agentic RAG Fine-tuned SmolLM: Simple and correct but still lacked error handling.

_Test 5: Finding the Maximum Value in a List_</br>
- Basic SmolLM: Correct and simple, using max().
- Basic SmolLM with RAG: Maintained correctness but lacked input validation.
- Advanced Agentic RAG SmolLM & Fine-tuned SmolLM: Introduced redundancy and inconsistency in function naming.

To summarize, let take a look at the best approach for each problem.

1. Test 1: Basic SmolLM or a refined Advanced Agentic RAG version with input validation.
2. Test 2: Advanced Agentic RAG with proper multi-class handling (multi_class='multinomial') and error handling.
3. Test 3: Add a check for empty lists before performing division.
4. Test 4: Use sorted() but add input validation to check for list type.
5. Test 5: Keep a single function using max(), but add error handling for empty lists.

---

### Unit Test

To run the unit test, you will need to execute the code below. The unit test will run through 5 examples of Python related
questions. These 5 questions will run through the Agentic workflow and the outputs of each question (both code and quality) will
be located in `stitching-project-YeeJunOw19/data_dump/advanced_agentic_tests`.

The unit test takes about 2-3 minutes to complete in DeepDish4.

```shell
python3 -m test.stitching
```