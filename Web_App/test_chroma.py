from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import VectorDBQA
from langchain.document_loaders import TextLoader, DirectoryLoader
import os
from langchain.embeddings import HuggingFaceInstructEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain import PromptTemplate, LLMChain
from langchain.chains.question_answering import load_qa_chain
from typing import Any, List, Mapping, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.chains import RetrievalQA
import textwrap
import torch.cuda

class gpt2(LLM):
    def retrieve(self, pdf):
        @property
        def _llm_type(self) -> str:
            return "custom"

        def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
        ) -> str:
            if stop is not None:
                raise ValueError("stop kwargs are not permitted.")
            
            tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
            model = AutoModelForCausalLM.from_pretrained("distilgpt2")
            inputs = tokenizer(prompt, return_tensors="pt").input_ids
            outputs = model.generate(inputs, max_new_tokens=7, do_sample=True, top_k=50, top_p=0.95).tolist()[0]
            return (tokenizer.decode(outputs))
        
class vectordb:
    loader = TextLoader('/Users/wangkl1/Documents/GitHub/basic-web-app/RadarPlainText/Radar_Basics1.txt')
    # loader = DirectoryLoader('../RadarPlainText/', glob="./*.txt", loader_cls=TextLoader)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)



    # # Downloading from HF took forever, check if embedding is on disk and use that
    # if os.path.exists("instructor-base"):
    #     # If the embeddings are found on disk, load them
    #     embedding = HuggingFaceInstructEmbeddings.from_pretrained("instructor-base")
    #     print("Loaded embeddings from disk.")
    # else:
    #     # If the embeddings are not found on disk, download them from Hugging Face
    #     print("Embeddings not found on disk. Downloading from Hugging Face...")
    #     embedding = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-base", 
    #                                                       model_kwargs={"device": "cuda"})
    #     embedding.save_pretrained("instructor-base")
    #     print("Saved embeddings to disk.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-base", 
                                                        model_kwargs={"device": device})

    vectordb = Chroma.from_documents(texts, embedding)

    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    qa_chain = RetrievalQA.from_chain_type(llm=gpt2(),
                                    chain_type="stuff",
                                    retriever=retriever,
                                    return_source_documents=True,
                                    verbose=False)

    def trim_string(input_string):
        input_string = str(input_string)
        trim_index = input_string.find("### Human:")
        if trim_index != -1:  # If the phrase is found
            return input_string[:trim_index]
        else:
            return input_string  # If the phrase isn't found, return the original string
    def wrap_text_preserve_newlines(text, width=110):
        # Split the input text into lines based on newline characters
        lines = text.split('\n')

        # Wrap each line individually
        wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

        # Join the wrapped lines back together using newline characters
        wrapped_text = '\n'.join(wrapped_lines)

        return wrapped_text

    def process_llm_response(llm_response):
        temp_resp = wrap_text_preserve_newlines(llm_response['result'])
        temp_resp = trim_string(temp_resp)
        print(temp_resp)
        print('\n\nSources:')
        for source in llm_response["source_documents"]:
            print(source.metadata['source'])

    query = "What is Radar"
    llm_response = qa_chain(query)
    process_llm_response(llm_response)