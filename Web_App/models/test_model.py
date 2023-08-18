from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain import PromptTemplate, LLMChain
from langchain.chains.question_answering import load_qa_chain
from typing import Any, List, Mapping, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.memory import ConversationBufferMemory
import random

#Create custom LangChain LLM, when we have the API for the actual model, just put the api call inside the _call function; pretty much what the _call function
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

class gpt2_model:
    def run(self, query):
        return "hi boi" + str(random.randint(0, 10))
        # template = """You are a chatbot having a conversation with a human.

        # Human: {question}
        # Chatbot: """

        # llm = gpt2()

        # prompt = PromptTemplate(template=template, input_variables=(["question"]))

        # llm_chain = LLMChain(
        #     llm=llm,
        #     prompt=prompt 
        # )

        # return(llm_chain.predict(question = query, return_only_outputs=True))