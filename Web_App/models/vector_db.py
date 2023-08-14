from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import VectorDBQA
from langchain.document_loaders import TextLoader, DirectoryLoader
from chromadb.utils import embedding_functions
from pdfReader import pdfToTxt
import torch
import os
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.chains import RetrievalQA
from _OpalLLM import OpalLLM
import textwrap
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Any, List, Mapping, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM

# class gpt2(LLM):
#     @property
#     def _llm_type(self) -> str:
#         return "custom"

#     def _call(
#         self,
#         prompt: str,
#         stop: Optional[List[str]] = None,
#         run_manager: Optional[CallbackManagerForLLMRun] = None,
#     ) -> str:
#         if stop is not None:
#             raise ValueError("stop kwargs are not permitted.")
        
#         tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
#         model = AutoModelForCausalLM.from_pretrained("distilgpt2")
#         inputs = tokenizer(prompt, return_tensors="pt").input_ids
#         outputs = model.generate(inputs, max_new_tokens=7, do_sample=True, top_k=50, top_p=0.95).tolist()[0]
#         return (tokenizer.decode(outputs))

#

class vectordb():
    def __init__(self):
        self.path = os.getcwd()
        # loader = TextLoader('Web_App/contexts/Radar_Basics1.txt')
        self.dir = os.listdir("Web_App/contexts")
        for fname in self.dir:
            if fname.endswith('.pdf'):
                pdfToTxt("Web_App/contexts/" + str(fname), "Web_App/contexts")
                os.remove("Web_App/contexts/" + str(fname))
        else:
            if len(self.dir)!=1:
                self.loader = DirectoryLoader("Web_App/contexts", glob="./*.txt", loader_cls=TextLoader)
            else:
                return("No uploaded files.")
            self.documents = self.loader.load()

            self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            self.texts = self.text_splitter.split_documents(self.documents)

            if os.path.exists("instructor-xl.pt"):
                self.embedding = torch.load("instructor-xl.pt")
                print("Loaded embeddings from disk.")
            else:
                # If the embeddings are not found on disk, download them from Hugging Face
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                self.embedding = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl", 
                                                                    model_kwargs={"device": self.device})

            self.vectordb = Chroma.from_documents(self.texts, self.embedding)
            self.retriever = vectordb.as_retriever(search_kwargs={"k": 5})
            #model_name="lmsys/vicuna-33b"
            #model_name="databricks/dolly-v2-12b"
            self.model_name="meta-llama/Llama-2-13b-chat-hf"
            #model_name = "tiiuae/falcon-40b-instruct"
            #model_name="CarperAI/stable-vicuna-13b-delta"
            
            self.local_llm=OpalLLM(model=self.model_name,
                            temperature=0.5,
                            top_k=60,
                            top_p=0.95,
                            max_tokens=200,
                            repetition_penalty=1.15)
            
            self.qa_chain = RetrievalQA.from_chain_type(llm=self.local_llm,
                                            chain_type="stuff",
                                            retriever=self.retriever,
                                            return_source_documents=True,
                                            verbose=False)
    def predict(self, query):
            self.llm_response = self.qa_chain(query)
            self.lines = self.llm_response['result'].split('\n')

            self.wrapped_lines = [textwrap.fill(line, width=110) for line in self.lines]
            self.temp_resp = '\n'.join(self.wrapped_lines)
            self.temp_resp = str(self.temp_resp)
            if "</s>" in self.temp_resp:
                self.ind = self.temp_resp.find("</s>")
                self.temp_resp = self.temp_resp[:self.ind]

            self.trim_index = self.temp_resp.find("### Human:")
            if self.trim_index != -1: 
                    self.temp_resp = self.temp_resp[:self.trim_index]

            self.answer = ""
            self.answer += self.temp_resp
            self.answer += '\n\nSources:'

            self.clean_source_ls = []
            for source in self.llm_response["source_documents"]:
                if source.metadata['source'] not in self.clean_source_ls:
                    self.clean_source_ls.append(source.metadata['source'])
            for sources in self.clean_source_ls:
                self.answer += sources
            
            return(self.answer)