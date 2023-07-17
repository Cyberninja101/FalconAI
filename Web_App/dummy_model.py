import time

# Use a pipeline as a high-level helper
# Use a pipeline as a high-level helper
from transformers import pipeline



def gpt(text):
   pipe = pipeline("text-generation", model="CoffeeAddict93/gpt1-modest-proposal")
   return(pipe([text]))



def budget_falcon(input):
    
    time.sleep(1.2) # to represent waiting for the model to finish running
    return f"I am a dum dum: {input}"