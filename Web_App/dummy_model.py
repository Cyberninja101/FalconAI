import time

# Use a pipeline as a high-level helper
# Use a pipeline as a high-level helper
from transformers import pipeline, Conversation



def gpt(text):
    # pipe = pipeline("conversational", model="kitbear444/DialoGPT-small-kit")

    pipe = pipeline("conversational", model="microsoft/DialoGPT-medium")
    conversation = Conversation(text)
    response = pipe(conversation)
    print("Response:", response.generated_responses[0])
    print(type(response.generated_responses[0]))
    return response.generated_responses[0]



def budget_falcon(input):
    
    time.sleep(1.2) # to represent waiting for the model to finish running
    return f"I am a dum dum: {input}"