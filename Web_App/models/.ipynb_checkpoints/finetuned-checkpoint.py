# Needs to be in same directory as configs, data folder


# Imports

import sys
sys.path.append('/home/jovyan/.local/lib/python3.8/site-packages')
import torch
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate

from langchain import OpenAI, LLMChain
from langchain.tools import DuckDuckGoSearchRun 
from langchain.llms import HuggingFacePipeline
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish
import re
import langchain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from pydantic import BaseModel
from langchain import PromptTemplate
from langchain.schema.output_parser import BaseLLMOutputParser
from transformers import GenerationConfig, pipeline
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import torch
import yaml
from langchain import PromptTemplate
from transformers import (AutoConfig, AutoModel, AutoModelForSeq2SeqLM,
                          AutoTokenizer, GenerationConfig, LlamaForCausalLM,
                          LlamaTokenizer, pipeline)
import os

"""
Ad-hoc sanity check to see if model outputs something coherent
Not a robust inference platform!
"""

def read_yaml_file(file_path):
    with open(file_path, 'r') as file:
        try:
            data = yaml.safe_load(file)
            return data
        except yaml.YAMLError as e:
            print(f"Error reading YAML file: {e}")

def get_prompt(human_prompt):
    prompt_template=f"### HUMAN:\n{human_prompt}\n\n### RESPONSE:\n"
    return prompt_template

def get_llm_response(prompt):
    raw_output = pipe(get_prompt(prompt))
    return raw_output

class MyOutputParser(BaseLLMOutputParser):
    def __init__(self):
        super().__init__()

    def parse_result(self, output):
        text = output[0].dict()["text"]
        cut_off = text.find("\n", 3)
        
        # delete everything after new line
        
        return text[:cut_off]

class radar_llama():
    def __init__(self):
        # Loading model
        self.config = read_yaml_file(os.sep.join([os.getcwd(), "Web_App", "models","configs", "radar_open_llama_7b_qlora.yaml"]))
        print("Load model")
        self.model_path = f"{self.config['model_output_dir']}/{self.config['model_name']}"
        if "model_family" in self.config and self.config["model_family"] == "llama":
            self.tokenizer = LlamaTokenizer.from_pretrained(self.model_path)
            self.model = LlamaForCausalLM.from_pretrained(self.model_path, device_map="auto", load_in_8bit=True)
        else:
            sekf.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path, device_map="auto", load_in_8bit=True)
        
        # Creating HF pipeline
        self.pipe = pipeline(
            "text-generation",
            model=self.model, 
            tokenizer=self.tokenizer, 
            max_length=2200,
            temperature=0.95,
            top_p=0.95,
            repetition_penalty=1.15
        )
        self.local_llm = HuggingFacePipeline(pipeline=self.pipe)  
        
        # Creating Prompt Template
        self.template = """You are a professional radar and documents specialist, acting as the human's AI assistant. 
        You will answer the following questions the best you can, being as informative and factual as possible.
        If You don't know, say you don't know. The following is a friendly conversation between the human and the AI.

        Examples of how you should respond to questions. The format is (question, answer):
        What are radars?, Radar is a radiolocation system that uses radio waves to determine the distance, angle, and radial velocity of objects relative to the site. It is used to detect and track aircraft, ships, spacecraft, guided missiles, and motor vehicles, and map weather formations, and terrain. The term RADAR was coined in 1940 by the United States Navy as an acronym for radio detection and ranging.
        What is radar clutter?, Radar clutter is defined as the unwanted back-scattered signals or echoes generated from physical objects in the natural environment like ground, sea, birds, etc. Due to the presence of clutter, the detection of target by the radar system in the environment becomes difficult. Clutter is a term used for unwanted echoes in electronic systems, particularly in reference to radars. Such echoes are typically returned from ground, sea, rain, animals/insects, chaff and atmospheric turbulences, and can cause serious performance issues with radar systems.
        What does Minimum Signal of Interest mean in radars?, Minimum Signal of Interest (MSI) is the minimum signal level that a radar system can detect and process. It is also known as the minimum detectable signal (MDS). The MSI is usually defined as the signal level that produces a specified signal-to-noise ratio (SNR) at the output of the receiver. The MSI is an important parameter in radar systems because it determines the range at which a target can be detected.
        What is radar clutter and how can I avoid detecting it?, Radar clutter is defined as the unwanted back-scattered signals or echoes generated from physical objects in the natural environment like ground, sea, birds, etc. Due to the presence of radar clutter, the detection of target by the radar system in the environment becomes difficult. To avoid detecting clutter in radar, you can use the following techniques: Pulse Doppler Radar, Moving Target Indicator (MTI), or Clutter Map.
        What are radars? Explain in detail., Radar is a radio location system that uses radio waves to determine the distance (ranging), angle (azimuth), and radial velocity of objects relative to the site. It is used to detect and track aircraft, ships, spacecraft, guided missiles, and motor vehicles, and map weather formations, and terrain. The term RADAR was coined in 1940 by the United States Navy as an acronym for radio detection and ranging. Radar operates by transmitting electromagnetic energy toward objects, commonly referred to as targets, and observing the echoes returned from them. The radar antenna transmits pulses of radio waves that bounce off objects in their path. The radar receiver listens for echoes of the transmitted signal. The time delay between transmission and reception of the echo is used to determine the distance of the object from the radar.
        What is the difference between a s band and a l band radar?, S band radar has a frequency range of 2 GHz to 4 GHz while L band radar has a frequency range of 1 GHz to 2 GHz. 
        What is the best bbq place?, The best bbq place is Kloby's.
        Tell me a joke, Why did the tomato turn red? Because it saw the salad dressing!

        Current conversation:
        {history}
        Human: {input}
        AI:"""

        self.the_output_parser=MyOutputParser()

        self.PROMPT = PromptTemplate(input_variables=["history", "input"], template=self.template)
        
        
        # Creating LangChain Conversation Chain
        self.conversation = ConversationChain(
            prompt=self.PROMPT,
            llm=self.local_llm,
            memory=ConversationBufferWindowMemory(k=5),
            return_final_only=True,
            verbose=False,
            output_parser=self.the_output_parser,
        )
    
    def run(self, query):
        # query is the user question, string
        return self.conversation.predict(input=query)

        