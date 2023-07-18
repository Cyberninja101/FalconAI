import time

# Use a pipeline as a high-level helper
# Use a pipeline as a high-level helper
from transformers import pipeline, Conversation



def gpt(text):
    # pipe = pipeline("text-generation", model="CoffeeAddict93/gpt1-modest-proposal")
    # pipe = pipeline("question-answering", model="monologg/koelectra-small-v2-distilled-korquad-384")


    # conversational_pipe = pipeline("conversational", model="kitbear444/DialoGPT-small-kit")
    # conversation = Conversation(text)
    # response = conversational_pipe(conversation)
    # print("Response:", response.generated_text)
    # return(response)

    conversational_pipe = pipeline("conversational", model="kitbear444/DialoGPT-small-kit")
    conversation = Conversation(text)
    response = conversational_pipe(conversation)
    print("Response:", response.generated_responses[0])
    print(type(response.generated_responses[0]))
    return response.generated_responses[0]



def budget_falcon(input):
    
    time.sleep(1.2) # to represent waiting for the model to finish running
    return f"I am a dum dum: {input}"