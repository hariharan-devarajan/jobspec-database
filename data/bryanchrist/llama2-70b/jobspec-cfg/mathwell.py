import os
import sys
import builtins
os.environ['HF_HOME'] = '/scratch/brc4cb/llama/cache'
#os.environ['TRANSFORMERS_CACHE'] = '/scratch/brc4cb/llama/cache'
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
from collections import defaultdict
import copy
import json
import os
from os.path import exists, join, isdir
from dataclasses import dataclass, field
import sys
from typing import Optional, Dict, Sequence
import numpy as np
from tqdm import tqdm
import logging
import triton
#import bitsandbytes as bnb
import pandas as pd
import random

import torch
import transformers
from torch.nn.utils.rnn import pad_sequence
import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
    Seq2SeqTrainer,
    BitsAndBytesConfig,
    LlamaTokenizer

)
from datasets import load_dataset, Dataset
import evaluate

from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    PeftModel
)
from peft.tuners.lora import LoraLayer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
#from adapter-transformers import AdapterType, AdapterConfig, load_adapter

# Set the environment variable
os.environ["HF_REMOTES_OFFLINE"] = "1"
#from dotenv import load_dotenv
# Load the environmental variables from the .env file
#load_dotenv()

#token = os.getenv('huggingface_token')
#if token:
   #print('token loaded')
    
#token = os.environ['huggingface_token']
#Uncomment if needed
#from huggingface_hub import login
#login(token=token)

# Redirect stdin to /dev/null
sys.stdin = open(os.devnull)

model_path = "meta-llama/Llama-2-70b-hf"   # Specify the path to the model
#adapter_path = "mathwell/llama_QA_adapter_no_embed/checkpoint-1250"  # Specify the path to the adapter weights
adapter_path = "mathwell/egsm_only/checkpoint-1000"
tokenizer = AutoTokenizer.from_pretrained(model_path, use_auth_token=True)

# Patch the built-in input function to return 'y' automatically
def mock_input(prompt=None):
    return 'y'

# Patch the input function to use the mock_input function
builtins.input = mock_input

try:
    # Attempt to load the model with trust_remote_code=True
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        load_in_8bit=True, 
       # max_memory=max_memory,
        torch_dtype=torch.float16,
        use_auth_token=True,
        config=AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    )
except EOFError:
    # If an EOFError occurs, provide the expected input ('y')
    pass

# Restore stdin
sys.stdin = sys.__stdin__


# Set special tokens
DEFAULT_PAD_TOKEN = "[PAD]"
def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg
        
if tokenizer._pad_token is None:
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
        tokenizer=tokenizer,
        model=model,
    )

print('Adding special tokens.')
tokenizer.add_special_tokens({
        "eos_token": tokenizer.convert_ids_to_tokens(model.config.eos_token_id),
        "bos_token": tokenizer.convert_ids_to_tokens(model.config.bos_token_id),
#                 "unk_token": tokenizer.convert_ids_to_tokens(
#                     model.config.pad_token_id if model.config.pad_token_id != -1 else tokenizer.pad_token_id
#                 ),
})


# Load the adapter weights
model = PeftModel.from_pretrained(model, adapter_path)

import pandas as pd
import random
df = pd.read_csv('data/mathwell.csv')
topics = ['Superman', "Batman", "Wonder Woman", "Barbie", "Power Rangers", "basketball", "soccer", "football", "volleyball", 'field hockey',\
'Fortnite', 'Spiderman', "Iron Man", "Captain America", "Captain Marvel", "Thor, the God of Thunder", "Ninja Turtles", "Black Panther", "Taylor Swift", "swimming",\
"Pokémon", "Super Mario", "Naruto", "unicorns", "Hello Kitty", "Minecraft", "lacrosse", "cheer leading", "LeBron James", "Steph Curry", "Patrick Mahomes",\
"Serena Williams", "dogs", "cats", "dinosaurs", "Harry Potter", "cars", "planes", "trains", "pizza", "cookies", "ice cream", 'candy']
for i in range(0,5000):
    topic = random.choice(topics)
    final_prompt = f"Write a grade school math word problem about {topic} and Python function with a commented out step-by-step solution to solve the word problem."
    prompt = "Write a grade school math word problem and Python function with a commented out step-by-step solution to solve the word problem."
    questions = []
    for i in range(0, 8):
        question = df['output'].iloc[random.randint(0,len(df)-1)]
        questions.append(question)
    formatted_prompt = []
    for i in range(0,8):
        formatted_prompt.append((f"Below is an instruction that describes a task. "
                f"Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{prompt}\n\n### Response: {questions[i]}"))
    formatted_prompt.append(f"Below is an instruction that describes a task. "
                f"Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{final_prompt}\n\n### Response: ")
    formatted_prompt = "\n".join(formatted_prompt)
#     formatted_prompt = (f"Below is an instruction that describes a task. "
#             f"Write a response that appropriately completes the request.\n\n"
#             f"### Instruction:\n{prompt}\n\n### Response: Question: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?"
#             f"\nSolution:\ndef solution():\n    #Roger started with 5 tennis balls\n    tennis_balls = 5\n    #2 cans of 3 tennis balls each is\n    bought_balls = 2 * 3    \n    #tennis balls. The answer is\n    result = tennis_balls + bought_balls\n    return result"
#             f"\nBelow is an instruction that describes a task. "
#             f"Write a response that appropriately completes the request.\n\n"
#             f"### Instruction:\n{prompt}\n\n### Response: Question: The bakers at the Beverly Hills Bakery baked 200 loaves of bread on Monday morning. "
#             f"They sold 93 loaves in the morning and 39 loaves in the afternoon. A grocery store returned 6 unsold loaves. How many loaves of bread did they have left?"
#             f"\nSolution:\ndef solution():\n    #The bakers started with 200 loaves\n    loaves_baked = 200\n    #They sold 93 in the morning and 39 in the afternoon\n    loaves_sold_morning=93\n    loaves_sold_afternoon=39\n    "
#             f"#The grocery store returned 6 loaves\n    loaves_returned = 6\n    #The answer is\n    result = loaves_baked - loaves_sold_morning - loaves_sold_afternoon + loaves_returned\n    return result"
#             f"\nBelow is an instruction that describes a task. "
#             f"Write a response that appropriately completes the request.\n\n"
#             f"### Instruction:\n{prompt}\n\n### Response: Question: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?"
#             f"\nSolution:\ndef solution():\n    #Olivia started with $23\n    money_initial = 23\n    #She bought 5 bagels\n    bagels = 5\n    #Each bagel cost $3\n    bagel_cost = 3\n    #5 bagels at $3 a bagel cost\n    money_spent = bagels * bagel cost\n"
#             f"    #The answer is\n    result = money_initial - money_spent\n    return result"
#             f"\nBelow is an instruction that describes a task. "
#             f"Write a response that appropriately completes the request.\n\n"
#             f"### Instruction:\n{prompt}\n\n### Response: Question: Michael had 58 golf balls. On Tuesday, he lost 23 golf balls. On Wednesday, he lost 2 more. How many golf balls did he have at the end of Wednesday?\nSolution:\ndef solution()\n"
#             f"    #Michael started with 58 golf balls\n    golf_balls_initial = 58\n    #He lost 23 on Tuesday\n    golf_balls_lost_tuesday = 23\n    #He lost two more on Wednesday\n    golf_balls_lost_wednesday = 2\n    #The answer is\n    "
#             f"zresult = golf_balls_initial - golf_balls_lost_tuesday - golf_balls_lost_wednesday\n    return result"
#             f"\nBelow is an instruction that describes a task. "
#             f"Write a response that appropriately completes the request.\n\n"
#             f"### Instruction:\n{prompt}\n\n### Response: Question: There were nine computers in the server room. Five more computers were installed each day, from Monday to Thursday. How many computers are now in the server room?\nSolution:\ndef solution():\n"
#             f"    #There were initially 9 computers\n    computers_initial = 9\n    #They installed 5 more each day\n    computers_per_day = 5\n    #There are 4 days between Monday and Thursday\n    num_days = 4\n    #There were\n    "
#             f"computers_added = computers_per_day * num_days\n    #computers added. The answer is\n    result = computers_initial + computers_added\n    return result"
#             f"\nBelow is an instruction that describes a task. "
#             f"Write a response that appropriately completes the request.\n\n"
#             f"### Instruction:\n{final_prompt}\n\n### Response:")
    inputs = tokenizer.encode(formatted_prompt, return_tensors="pt")
    attention_mask = torch.ones_like(inputs)
    inputs = inputs.to('cuda')
    output = model.generate(inputs=inputs, attention_mask=attention_mask, max_new_tokens = 250, do_sample = True)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Split the generated text by the prompt to extract the newly generated part
    generated_text_parts = generated_text.split(final_prompt)
    newly_generated_text = generated_text_parts[-1].strip()
    if "\nBel" in newly_generated_text:
        newly_generated_text = newly_generated_text.split("\nBel")[0]
    output_file = "mathwell_questions_egsm_only.txt"  # Specify the path and filename for the output file
    with open(output_file, "a") as f:  # Open the file in append mode ("a")
        f.write(f"Topic: {topic} " + newly_generated_text + "\n")  # Append the newly generated text to the file
        