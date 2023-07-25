
# Attempting to finetune GPT-J model with random RADAR data
# Need to run this on the million dollar servers though

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
import os
import sys

# # GPT-J Model (Bigger, Better, Use Servers)
# tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6b")
# model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6b")

# GPT-2 Model (Smaller, Faster, Temporary)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")


def gen(shards):
    for shard in shards:
        with open(shard) as f:
            for line in f:
                # print("line", line)
                yield {"line": line}

# Create custom huggingface dataset from radar textbook txt
shards = [os.sep.join([os.getcwd(),"HandbookPlainText",f"RadarHandbook_CH0{i}.txt"]) for i in range(1, 27)]
df = Dataset.from_generator(gen, gen_kwargs={"shards": shards})

print(df)

# Tokenize the data
# TODO: FIX TOKENIZER,it doesn't work
def tokenize_function(examples):
    return tokenizer(examples["line"], padding="max_length", truncation=True)

print(df.map(tokenize_function, batched=True))

