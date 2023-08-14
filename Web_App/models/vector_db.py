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
    def predict(self, query):
        path = os.getcwd()
        # loader = TextLoader('Web_App/contexts/Radar_Basics1.txt')
        dir = os.listdir("Web_App/contexts")
        for fname in dir:
            if fname.endswith('.pdf'):
                pdfToTxt("Web_App/contexts/" + str(fname), "Web_App/contexts")
                os.remove("Web_App/contexts/" + str(fname))
        else:
            if len(dir)!=1:
                loader = DirectoryLoader("Web_App/contexts", glob="./*.txt", loader_cls=TextLoader)
            else:
                return("No uploaded files.")
            documents = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            texts = text_splitter.split_documents(documents)

            if os.path.exists("instructor-xl.pt"):
                embedding = torch.load("instructor-xl.pt")
                print("Loaded embeddings from disk.")
            else:
                # If the embeddings are not found on disk, download them from Hugging Face
                device = "cuda" if torch.cuda.is_available() else "cpu"
                embedding = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl", 
                                                                    model_kwargs={"device": device})

            vectordb = Chroma.from_documents(texts, embedding)
            retriever = vectordb.as_retriever(search_kwargs={"k": 5})
            #model_name="lmsys/vicuna-33b"
            #model_name="databricks/dolly-v2-12b"
            model_name="meta-llama/Llama-2-13b-chat-hf"
            #model_name = "tiiuae/falcon-40b-instruct"
            #model_name="CarperAI/stable-vicuna-13b-delta"
            
            local_llm=OpalLLM(model=model_name,
                            temperature=0.5,
                            top_k=60,
                            top_p=0.95,
                            max_tokens=200,
                            repetition_penalty=1.15)
            
            qa_chain = RetrievalQA.from_chain_type(llm=local_llm,
                                            chain_type="stuff",
                                            retriever=retriever,
                                            return_source_documents=True,
                                            verbose=False)

            llm_response = qa_chain(query)
            lines = llm_response['result'].split('\n')

            wrapped_lines = [textwrap.fill(line, width=110) for line in lines]
            temp_resp = '\n'.join(wrapped_lines)
            temp_resp = str(temp_resp)
            if "</s>" in temp_resp:
                ind = temp_resp.find("</s>")
                temp_resp = temp_resp[:ind]

            trim_index = temp_resp.find("### Human:")
            if trim_index != -1: 
                    temp_resp = temp_resp[:trim_index]

            answer = ""
            answer += temp_resp
            answer += '\n\nSources:'

            clean_source_ls = []
            for source in llm_response["source_documents"]:
                if source.metadata['source'] not in clean_source_ls:
                    clean_source_ls.append(source.metadata['source'])
            for sources in clean_source_ls:
                answer += sources
            
            return(answer)