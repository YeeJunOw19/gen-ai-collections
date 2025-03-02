
import os

TOP_LEVEL_LOCATION = "/run/secrets/"

# Pinecone API Key environment variable
try:
    with open(TOP_LEVEL_LOCATION + "pinecone_key", mode="r") as file:
        PINECONE_API_KEY = file.read()
except FileNotFoundError:
    PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")


# MotherDuck API Key environment variable
try:
    with open(TOP_LEVEL_LOCATION + "motherduck_key", mode="r") as file:
        MOTHERDUCK_TOKEN = file.read()
except FileNotFoundError:
    MOTHERDUCK_TOKEN = os.environ.get("MOTHERDUCK_TOKEN")


# Hugging Face API Key environment variable
try:
    with open(TOP_LEVEL_LOCATION + "hugging_face_key", mode="r") as file:
        HUGGING_FACE_API = file.read()
except FileNotFoundError:
    HUGGING_FACE_API = os.environ.get("HUGGING_FACE_API")


# OpenAI API Key environment variable
try:
    with open(TOP_LEVEL_LOCATION + "openai_key", mode="r") as file:
        OPEN_AI_KEY = file.read()
except FileNotFoundError:
    OPEN_AI_KEY = os.environ.get("OPENAI_API_KEY")


# Hugging Face API used for LLama Model
try:
    with open(TOP_LEVEL_LOCATION + "hugging_face_llama_model", mode="r") as file:
        LLAMA_MODEL_API = file.read()
except FileNotFoundError:
    LLAMA_MODEL_API = os.environ.get("HUGGING_FACE_API_NU")