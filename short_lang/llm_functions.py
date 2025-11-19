# Copyright (c) Praneeth Vadlapati

import os
import time

from dotenv import load_dotenv
from IPython.display import display, Markdown
import numpy as np
import openai

def load_env():
    load_dotenv(override=True)  # bypass the cache and reload the variables
load_env()


# data_folder = "data_files"
# if not os.path.exists(data_folder):
#     os.makedirs(data_folder)

def display_md(md_text):
    display(Markdown(md_text))


model = os.getenv("OPENAI_MODEL")
if model:
    model = model.strip()
    print(f"Model: {model}")
else:
    raise Exception("OPENAI_MODEL is not set in the environment variables")


client = openai.OpenAI()


def ask_llm(messages, max_retries=3):
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]

    for _ in range(max_retries):
        response = None
        try:
            response = client.chat.completions.create(messages=messages, model=model)
            response = response.choices[0].message.content.strip()
            if not response:
                raise Exception("Empty response from the bot")
            return response
        except openai.RateLimitError as e:
            print("Rate Limit Wait ")
            raise e
        except Exception as e:
            e = str(e)
            if "503" in e:  # Service Unavailable
                print("Unavailable Wait ")
                time.sleep(15)
            elif e == "Connection error.":
                print("Server not online ")
            else:
                print(f"Error Retrying ")
    raise Exception(f"No response from the bot after {max_retries} retries")


embed_model_name = os.getenv("EMBED_MODEL_NAME", "")
if not embed_model_name:
    raise Exception("EMBED_MODEL_NAME is not set in the environment variables")

embed_client = openai.OpenAI(base_url = os.getenv("EMBEDDING_BASE_URL"))


def get_embedding(value):
    response = embed_client.embeddings.create(
        input=[value],
        model=embed_model_name
    )
    embedding = response.data[0].embedding
    return np.array(embedding)

def get_embeddings(values):
    response = embed_client.embeddings.create(
        input=values,
        model=embed_model_name
    )
    return [item.embedding for item in response.data]


if __name__ == "__main__":
    # Test the functions
    test_message = "Hello, how are you?"
    # print("Asking LLM...")
    # response = ask_llm(test_message)
    # print("Response from LLM:", response)

    print("Getting embedding...")
    embedding = get_embedding(test_message)
    print("Embedding length:", len(embedding))
