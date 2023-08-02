from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import VectorDBQA
from langchain.document_loaders import TextLoader, DirectoryLoader
from chromadb.utils import embedding_functions
import os
from langchain.embeddings import HuggingFaceInstructEmbeddings

#loader = TextLoader('../RadarPlainText/Radar Basics1.txt')
loader = DirectoryLoader('../RadarPlainText/', glob="./*.txt", loader_cls=TextLoader)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_documents(documents)



# # Downloading from HF took forever, check if embedding is on disk and use that
if os.path.exists("instructor-base"):
    # If the embeddings are found on disk, load them
    embedding = HuggingFaceInstructEmbeddings.from_pretrained("instructor-base")
    print("Loaded embeddings from disk.")
else:
    # If the embeddings are not found on disk, download them from Hugging Face
    print("Embeddings not found on disk. Downloading from Hugging Face...")
    embedding = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-base", 
                                                      model_kwargs={"device": "cuda"})
    embedding.save_pretrained("instructor-base")
    print("Saved embeddings to disk.")

vectordb = Chroma.from_documents(texts, embedding)

retriever = vectordb.as_retriever(search_kwargs={"k": 2})

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