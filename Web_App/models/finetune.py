
# Attempting to finetune GPT-J model with random RADAR data
# Need to run this on the million dollar servers though

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset

# # GPT-J Model (Bigger, Better, Use Servers)
# tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6b")
# model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6b")

# GPT-2 Model (Smaller, Faster, Temporary)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Create custom huggingface dataset
def gen(shards):
    for shard in shards:
        with open(shard) as f:
            for line in f:
                yield {"line": line}

shards = [f"../../../HandbookPlainText/RadarHandbook_CH0{i}.txt" for i in range(1, 27)]
ds = Dataset.from_generator(gen, gen_kwargs={"shards": shards})

print(ds)
# Tokenize text function
# def tokenize_function(examples):
#     return tokenizer(examples["text"], padding="max_length", truncation=True)

# print(text.map(tokenize_function, batched=True))

