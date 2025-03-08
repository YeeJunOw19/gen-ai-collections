## Assignment 3

### Introduction

In this assignment, we will be focusing on fine-tuning a smaller pre-trained LLM model, on answering questions related to
Python coding. The small LLM model is fine-tuned using LoRA and a series of five questions will be tested on both the base
model and the fine-tuned model. These questions are Python related questions and these samples questions and answers will
be provided in section below. The questions and answers from the dataset are split into 80% for training, 10% for validation,
and 10% for testing, although the 10% validation set is not used in this exercise.

Dataset: https://huggingface.co/datasets/HydraLM/python-code-instructions-18k-alpaca-standardized

> [!NOTE]
> Image Location: DeepDish4
> On previous assignment that was a comment about removing modules that are not used in this assignment. But, because this repository
> is a monolithic repository with many dependencies and functionalities intertwined, I cannot simply remove modules that are not being used.
> Furthermore, Dagster orchestration is a tightly integrated orchestration tool and removing any part of this project will break the system.
> I hope this make sense!

---

### Prerequisite

1. The *requirements.txt* file will have all the Python packages to be installed before running the code in this
repository.
2. It is recommended to use Python 3.11+ for running code in this repository.
3. API keys to all the relevant services. These will be shared through Slack. Alternatively, you can acquire these secrets from
a folder called .secrets in my workspace in DeepDish, within the `assignment-2-YeeJunOw19` folder.

> [!NOTE]
> The fine-tuned model and same samples output can be found here in this shared drive.
> https://drive.google.com/drive/folders/1AZU-9-ZwoTEU9UC1Q10yz9pADgNlmY28?usp=drive_link

---

### Quick Start

To be able to run the fill fine-tuning process, as well as running the small unit test, please follow the instruction below
to set up the workspace. The image will be built in **DeepDish 4**.

*Step 1*:</br>
Clone the remote repository into your local machine.
```shell
git clone git@github.com:NUMLDS/assignment-3-YeeJunOw19.git
cd assignment-3-YeeJunOw19
```

*Step 2*:</br>
Copy the `.secrets` folder into `assignment-3-YeeJunOw19`. You should have 5 files containing different API keys for different
cloud provider and services.

*Step3*:</br>
From the Google Shared Drive, there will be a folder called `unit_test_data`. Run the code below to create a folder called `data`
in the `test` folder. Then download the parquet file and paste the data inside the `data` folder that you have just created.

```shell
mkdir ./test/data
```

*Step 4*:</br>
Create a Docker container from the pre-built image using Docker Compose.
```shell
docker compose run --rm genai
```

---

### Running Full LoRA Fine Tuning

The model that we are using in this exercise will be from `HuggingFaceTB/SmolLM2-360M-Instruct`. Most of the modules and functions
uses the name `llama` because I initially started using `meta-llama/Llama-3.2-1B-Instruct` but the model was performing too
well on Python coding. Therefore, I had to switch to a much smaller model to effectively show the effect of fine-tuning a model.

To run a preliminary baseline performance from the model, 5 Python questions are sent to the small LLM model to generate the code
to solve these questions. To run this step, execute the code below in the docker container.

```shell
python3 -m src.fine_tuning.python_evaluation.pre_tune
```

The output from this run will be saved in a high level folder called `data_dump/pre_fine_tuning`.

To run the full LoRA fine tuning, execute the code below in docker container.

```shell
python3 -m src.fine_tuning.lora.lora_model_tuning
```

Fine-tuning process uses 3,000 Python questions and answers, picked specifically from training set. The fine-tuning process
runs for 1,000 steps.

> [!CAUTION]
> The entire fine-tuning process takes about 3.5 hours to complete on a T4 GPU with 32GB memory

The new fine-tuned model will be saved in this high level folder `data_dump/fine_tuned_models/SmolLM2-360M-Instruct-Python`.

To run the same 5 questions set from earlier on the fine-tuned model, execute the code below.

```shell
python3 -m src.fine_tuning.python_evaluation.post_tune
```

---

### Running Unit Test

The purpose of this unit test is to demonstrate the fine-tuning process on a smaller training set and to run it on fewer steps.
There are a total of 300 Python questions and answers being used in this small unit test with a maximum steps of 20. The 300
Q&A set has already been built into the Docker container. The data is saved in a highly compressed parquet file located in 
`test/data/training_data.parquet`.

The total runtime for this unit test should take less than 10 minutes. The sample task to solve in this unit test is to ask
the LLM model to print out the Fibonacci series. To run this unit test, run the code below in the container.

```shell
python3 -m test.assignment3
```

---

### High Level Structure

The core functionality of this project is contained within `src/fine_tuning`. The module `llama_object` has the functionality
to create an object for the LLM and for fine-tuning purpose. The fine tuning scripts are done from the scripts in `src/fine_tuning/lora`,
whereas scripts that evaluate the performance of the LLM is located in `src/fine_tuning/python_evaluation`. Below is a short summary
of what each script is doing.

1. `hf_dataset_generator.py` - Takes in a set of questions and answers and reformat them into LoRA fine-tuning format
2. `lora_model_tuning.py` - Main function for fine-tuning the LLM model
3. `llamna_instruct.py` - The main script to create a LLM object
4. `llama_tune.py` - Main function to create a fine-tuning object
5. `llama_concurrency.py` - Store steps required to fine-tune a LLM model, to be used for multiprocessing
6. `pre_tune.py` - Running the pre-tuned LLM model
7. `post_tune.py` - Running the post-tuned LLM model

The data ingestion functionality is the same as what I have used in my past assignments. It involves ingesting data from Hugging Face,
cleaning the data, and loading the data into MotherDuck. As before, to run the ingestion process, you will need all the required
API keys and run the command below.

```shell
dagster job execute -m src -j run_raw_data_save
```

---

### Discussion

Before fine-tuning the model, the LLM is asked to solve 5 questions related to Python coding. Out of the 5 questions, 2 of them
have pretty interesting results before and after fine-tuning.

**Task 1**:</br>
In this task, the LLM model is being asked to implement a KNN classifier in Python. Below is answer that the LLM provides before and
after fine-tuning.

_Before_:
```python
import numpy as np

def k_nearest_neighbors(X, y, k):
    # Calculate the distance between each data point and all other data points
    distances = np.linalg.norm(X - y, axis=1)
    # Sort the distances in descending order
    sorted_distances = distances.argsort()
    # Get the k most similar data points
    knn_indices = sorted_distances[:k]
    # Get the labels of the k most similar data points
    knn_labels = y[knn_indices]
    return knn_indices, knn_labels

# Example usage:
X = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array([1, 0, 1])
k = 2

knn_indices, knn_labels = k_nearest_neighbors(X, y, k)

print("KNN Classifier:")
print("Indices of the k most similar data points:")
print(knn_indices)
print("Labels of the k most similar data points:")
print(knn_labels)
```

_After_:
```python
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Load the dataset
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
y = np.array([1, 2, 3, 4])

# Create a KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)

# Train the classifier
knn.fit(X, y)

# Make predictions
predictions = knn.predict([[9, 10]])

# Print the predictions
print(predictions) # [1 2]

# Print the accuracy of the classifier
print(knn.score(X, y)) # 0.999

# Print the confusion matrix
knn.confusion_matrix(y, knn.predict(X))
```

The fine-tuning transformed the implementation from a custom, error-prone approach—where distances were mistakenly
calculated between data points and labels and sorting was potentially ambiguous—to a robust, standardized solution using
scikit-learn’s KNeighborsClassifier that not only correctly handles distance calculations and neighbor selection but also
provides additional functionalities such as training, prediction, accuracy scoring, and confusion matrix generation, thereby
enhancing reliability, performance, and ease of evaluation in a single, well-integrated framework.


**Task 2**:</br>
This time, the LLM model is asked to write a function to print out the Fibonacci series.

_Before_:
```python
def print_fibonacci(n):
    if n <= 0:
        print("Input should be a positive integer.")
    elif n == 1:
        print("1")
    elif n == 2:
        print("1 1")
    else:
        fib_sequence = [1, 1]
        while len(fib_sequence) < n:
            fib_sequence.append(fib_sequence[-1] + fib_sequence[-2])
        print("1 " + " ".join(map(str, fib_sequence)))

# Test the function
print_fibonacci(10)
```

_After_:</br>
```python
def fibonacci(n): 
    a = 0
    b = 1
    if n < 0: 
        print("Incorrect input") 
    elif n == 0: 
        return a 
    elif n == 1: 
        return b 
    else: 
        for i in range(2,n): 
            c = a + b 
            a = b 
            b = c 
        return b 
  
# Driver Code 
n = 10
print(fibonacci(n)) 

# Output: 21 
# 21 21 20 21 21 20 21 21 20 21 21 
# 21 21 21 20 21 20 21 21 20 21 21 
# 21 21 21 20
```

The fine-tuning did not work well because it altered the original intent—printing the entire Fibonacci series—instead
computing and returning only the nth Fibonacci number; potential reasons for this include a misinterpretation of the intended
functionality during fine-tuning, an overemphasis on efficiency by returning a single value rather than generating a series,
and possibly insufficient training examples or ambiguous instructions that led the model to focus on computing the final value
rather than constructing and printing the full sequence.

**Task 3**:</br>
The LLM is asked to generate a function for calculating the average of a list of numbers in Python.

_Before_:
```python
def calculate_average(numbers):
    return sum(numbers) / len(numbers)
```

_After_:
```python
def average(numbers):
    total = 0
    for num in numbers:
        total += num
    return total / len(numbers)

numbers = [1, 2, 3, 4, 5]
print(average(numbers)) # 3.0

# alternative solution
def average(numbers):
    total = 0
    for num in numbers:
        total += num
    return total / len(numbers)

numbers = [1, 2, 3, 4, 5]
print(average(numbers)) # 3.0

# alternative solution
def average(numbers):
    total = 0
    for num in numbers:
        total += num
    return total / len(numbers)

numbers = [1, 2, 3, 4, 5]
print(average(numbers)) # 3.0

# alternative solution
def
```

The biggest difference here is that the model now is using the maximum token parameter to generate as long of an answer as 
possible. The maximum token in both runs are set to about 300 but somehow, after the fine-tuning process, it trys to generate
as long of an answer as possible, until it is being cutoff at the 300 token mark.

---

### CUDA Out of Memory

I tried running my Docker container using the instructions provided above in DeepDish4, but it seems like everyone is using DD4
and is causing "CUDA Out of Memory" issue. I am able to run the unit test on the inference steps but it comes to fine-tuning steps,
CUDA basically ran out of memory.

I have my own EC2 instance in AWS and I ran the same steps from above without any issue. I am assuming that GPUs in DD4 are
overloaded. Here are the specs that I am using in the EC2 instance.

![EC2 Specs](/images/Assignment3_EC2_Specs.png)

_DeepDish4 Log_:

![DD4 Error Log](/images/Assignment3_DD4_Error.png)

_EC2 Log_:

![AWS Log](/images/Assignment3_AWS_Log.png)
