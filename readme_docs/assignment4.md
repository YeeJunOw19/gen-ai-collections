# Assignment 4

### Introduction

In this assignment, we will be focusing on Agentic AI. With the advancement in Large Language Models (LLMs), the ability to generate code is increasingly important.
The primary concern regarding code generation using cloud LLMs from major providers is the sharing of sensitive data and
information with third parties that may be untrusted. This lead to an explosion of many pre-trained LLMs that can be
downloaded locally and used for code generation.

This in turns sprout a different problem, where the LLMs are used to generate code that is not as high quality. This
lead to the need for fine-tuning these open-source LLMs to improve their performance. This application aims to compare
two open-source model: LLama Instruct (base) and SmolLM (fine-tuned), and evaluate the performance of both of these
models. The code generated from these two models may not be production ready or up to standard, but this serves as a
baseline for evaluating whether to continue the future work of fine-tuning these models.

---

### Prerequisite

1. The *requirements.txt* file will have all the Python packages to be installed before running the code in this
repository.
2. It is recommended to use Python 3.11+ for running code in this repository.
3. API keys to all the relevant services. These will be shared through Slack. Alternatively, you can acquire these secrets from
a folder called .secrets in my workspace in DeepDish, within the `assignment-4-YeeJunOw19` folder.

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
git clone git@github.com:NUMLDS/assignment-4-YeeJunOw19.git
cd assignment-4-YeeJunOw19
```

*Step 2*:</br>
Copy the `.secrets` folder into `assignment-4-YeeJunOw19`. You should have 5 files containing different API keys for different
cloud provider and services.

*Step 3*:</br>
To be able to use the fine-tuned model, you will need to setup the appropriate folder in the project folder.
```shell
mkdir data_dump
```
Save the folder `SmolLM2-360M-Instruct-Python` (shared through the Google Drive link above) into the newly created folder.
The structure should look like this `assignment-4-YeeJunOw19/data_dump/SmolLM2-360M-Instruct-Python`.

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

### Streamlit Application Examples

The main page of Streamlit application is shown below. The first thing that you will need to do is to type in a Python question
that you would like to send to the two local models to be evaluated.

![EC2 Specs](/images/Streamlit_Main_Page.png)

The process takes about 30 seconds to generate the Python code for both the models, and the evaluation from OpenAI of the quality
of the generated code from both models.

**Example 1**:

![EC2 Specs](/images/Streamlit_Example1.png)

**Example 2**:

![EC2 Specs](/images/Streamlit_Example2.png)

**Example 3**:

![EC2 Specs](/images/Streamlit_Example3.png)

**Example 4**:

![EC2 Specs](/images/Streamlit_Example4.png)

**Example 5**:

![EC2 Specs](/images/Streamlit_Example5.png)

---

### Streamlit Architecture and Discussion

The overall architecture of the Agentic AI is shown in the graph below. Relevancy evaluation and quality evaluation are done
through OpenAI's `gpt-3.5-turbo`. You can find the code for the Agentic AI workflow in `src/agentic`. The code for Streamlit
application is in `streamlit_main_page.py`.

![EC2 Specs](/images/Assignment4_Agentic_Architecture.png)

The evaluation of the code quality is done using the four criteria listed below.

1. Correctness
2. Readability & Maintainability
3. Code Style
4. Scalability
5. Error Handling & Robustness

The overall assessment result is categorized into the three groups listed below.

1. *Yes* - The code generated by the local LLM is production ready and is of good quality
2. *No* - The code generated by the local LLM is not production ready and is of extreme poor quality
3. *Needs Improvement* - The code generated by the local LLM is not perfect for production, but with small tweaks it will be 

---

### Unit Test

To run the unit test, you will need to execute the code below. The unit test will run through 5 examples of Python related
questions. These 5 questions will run through the Agentic workflow and the outputs of each question (both code and quality) will
be located in `assignment-4-YeeJunOw19/data_dump/agentic_tests`.

The unit test takes about 2-3 minutes to complete in DeepDish4.

```shell
python3 -m test.assignment4
```