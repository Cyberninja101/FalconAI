from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain import PromptTemplate, LLMChain
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


template = """
You are an AI chatbot having a conversation with a human. You are talkative and remember details from the conversation's chat history. You answer with a single sentence.
You like to include data from the context section in your responses.

Context section: {context}

Chat history:
{chat_history}

Human: {question}
AI: """

llm = bloom()

prompt = PromptTemplate(template=template, input_variables=["question", "chat_history", "context"])
memory = ConversationBufferMemory(memory_key="chat_history")

llm_chain = LLMChain(
    prompt=prompt, 
    llm=llm,
    memory=memory
)

input = "What's my favorite color?"
file = "Human's favorite color is blue."

print(llm_chain.predict(question = input, context = file))