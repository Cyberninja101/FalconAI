import time
from transformers import pipeline

# This code is wrong
# generator = pipeline("text-generation", model="distilgpt2")
# generator(
#     "In this course, we will teach you how to",
#     max_length=30,
#     num_return_sequences=2,
# )


def gpt2(text):
    # Make code that will respond to text using gpt2
    pass


def budget_falcon(input):
    time.sleep(1.2) # to represent waiting for the model to finish running
    return f"I am a dum dum: {input}"