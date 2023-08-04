from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, DirectoryLoader
import os
from langchain.embeddings import HuggingFaceInstructEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Any, List, Mapping, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.chains import RetrievalQA
import textwrap
import torch.cuda
from pdfReader import pdfToTxt

class gpt2(LLM):
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
    def predict(self, query):
        path = os.getcwd()
        # loader = TextLoader('Web_App/contexts/Radar_Basics1.txt')
        dir = os.listdir("Web_App/contexts")
        for fname in dir:
            if fname.endswith('.pdf'):
                pdfToTxt("Web_App/contexts/" + str(fname), "Web_App/contexts")
                os.remove("Web_App/contexts/" + str(fname))
                print("removed")
        else:
            print("elsed")
            if len(dir)!=1:
                loader = DirectoryLoader(os.sep.join([path, "Web_App",'contexts']), glob="./*.txt", loader_cls=TextLoader)
                print("docs loaded")
            else:
                return("No uploaded files.")
            documents = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            texts = text_splitter.split_documents(documents)

            print("text splitted")
            print(len(texts))

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
            print(device)
            embedding = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-base", 
                                                                model_kwargs={"device": device})
            print("embedding")
            vectordb = Chroma.from_documents(texts, embedding)

            retriever = vectordb.as_retriever(search_kwargs={"k": 3})

            qa_chain = RetrievalQA.from_chain_type(llm=gpt2(),
                                            chain_type="stuff",
                                            retriever=retriever,
                                            return_source_documents=True,
                                            verbose=False)
            print("chain initialized")
            llm_response = qa_chain(query)
            lines = llm_response['result'].split('\n')

            wrapped_lines = [textwrap.fill(line, width=110) for line in lines]
            temp_resp = '\n'.join(wrapped_lines)
            temp_resp = str(temp_resp)
            trim_index = temp_resp.find("### Human:")
            if trim_index != -1: 
                    temp_resp = temp_resp[:trim_index]

            print("formatted")

            answer = ""
            answer += temp_resp
            answer += '\n\nSources:'
            for source in llm_response["source_documents"]:
                answer += source.metadata['source']

            return(answer)