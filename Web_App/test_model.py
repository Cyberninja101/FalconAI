from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain import PromptTemplate, LLMChain
from langchain.chains.question_answering import load_qa_chain
from typing import Any, List, Mapping, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.memory import ConversationBufferMemory

#Create custom LangChain LLM, when we have the API for the actual model, just put the api call inside the _call function; pretty much what the _call function
class bloom(LLM):

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
        
        tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")
        model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-560m")
        inputs = tokenizer(prompt, return_tensors="pt").input_ids
        outputs = model.generate(inputs, max_new_tokens=15, do_sample=True, top_k=50, top_p=0.95).tolist()[0]
        return (tokenizer.decode(outputs))


template = """You are a chatbot having a conversation with a human.

Given the following long document and a question, create a final answer.

{context}

{chat_history} 
Human: {question}
Chatbot: """

llm = bloom()

prompt = PromptTemplate(template=template, input_variables=["context", "chat_history", "question"])
memory = ConversationBufferMemory(memory_key="chat_history", input_key="question")

llm_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate.from_template(template), 
    memory=memory
)

query = "What's my favorite color?"
file = "Human's favorite color is blue."

print(llm_chain.predict(context= file, question = query, return_only_outputs=True))