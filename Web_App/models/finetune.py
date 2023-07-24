
# Attempting to finetune GPT-J model with random RADAR data
# Need to run this on the million dollar servers though

# Temporarily running it on GPT2


from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TrainingArguments, Trainer, GPT2Config, GPT2Model
from datasets import Dataset
import os
import sys
import numpy as np
import evaluate


# # GPT-J Model (Bigger, Better, Use Servers)
# tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6b")
# model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6b")

# GPT-2 Model (Smaller, Faster, Temporary)
tokenizer = AutoTokenizer.from_pretrained("gpt2", num_labels=1)

# tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained("gpt2", num_labels=1)


def gen(shards):
    for shard in shards:
        with open(shard) as f:
            for line in f:
                yield {"label" : 0,
                    "text": line}

# Create custom huggingface dataset from radar textbook txt
shards = [os.sep.join([os.getcwd(),"HandbookPlainText",f"RadarHandbook_CH0{i}.txt"]) for i in range(1, 27)]
df = Dataset.from_generator(gen, gen_kwargs={"shards": shards})

# train test split data
df = df.train_test_split(test_size=0.2)


# Tokenize the data
# TODO: FIX TOKENIZER,it doesn't work
def tokenize_function(examples):
    # Fix pad token
    tokenizer.pad_token = tokenizer.eos_token
    
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

        
    return tokenizer(examples["text"], padding="max_length", max_length=20,truncation=True)

# tokenizing dataframe
df = df.map(tokenize_function, batched=True)
# print(type(df), df)

print(df)


# Initializing a GPT2 configuration
configuration = GPT2Config()

# Initializing a model (with random weights) from the configuration
model = GPT2Model(configuration)

# Accessing the model configuration
configuration = model.config

training_args = TrainingArguments(remove_unused_columns=False, output_dir="test_trainer")

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# Trainer object
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=df["train"],
    eval_dataset=df["test"],

    # compute_metrics=compute_metrics,
)

trainer.train()