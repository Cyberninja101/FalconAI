import os
from langchain import PromptTemplate, HuggingFaceHub
from langchain.chains import LLMChain, ConversationChain
from langchain.chains.conversation.memory import (ConversationBufferMemory, 
                                                  ConversationSummaryMemory, 
                                                  ConversationBufferWindowMemory,
                                                  ConversationKGMemory)
class google_flan():
    def __init__(self):
        os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'

        # build prompt template for simple question-answering
        self.template = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. 
            If the AI does not know the answer to a question, it truthfully says it does not know. 

            Current conversation:
            {history}

            Human: {input}
            AI: """

        self.PROMPT = PromptTemplate(input_variables=["history", "input"], template=self.template)
        # initialize HF LLM
        self.flan_t5 = HuggingFaceHub(
            repo_id="google/flan-t5-xxl",
            model_kwargs={"temperature":0.01}
        )

        self.conversation = ConversationChain(
            prompt=self.PROMPT,
            llm=self.flan_t5,
            memory=ConversationBufferWindowMemory(k=4)
        )
    def run(self, text):
        return([self.conversation(text)["response"], self.conversation.memory.buffer])