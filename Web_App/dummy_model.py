import os
from langchain import PromptTemplate, HuggingFaceHub, FewShotPromptTemplate
from langchain.chains import LLMChain, ConversationChain
from langchain.chains.conversation.memory import (ConversationBufferMemory, 
                                                  ConversationSummaryMemory, 
                                                  ConversationBufferWindowMemory,
                                                  ConversationKGMemory)
from pdfReader import read_pdf

class model():
    def __init__(self):
        os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'

        # build prompt template for simple question-answering
        self.template = """
            Human: {input}
            AI: {answer}"""

        self.PROMPT = PromptTemplate(input_variables=["input", "answer"], template=self.template)


        # now break our previous prompt into a prefix and suffix
        # the prefix is our instructions
        # self.prefix = """
        # The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from the Current conversation. 
        # If the AI does not know the answer to a question, it truthfully says it does not know. 

        # Current conversation:
        # {history}

        # The following is a document containing vital context and knowledge. 
        # The AI will respond to new questions based on the content of the file. Here is the context:
        # """
        # # and the suffix our user input and output indicator
        # self.suffix = """
        # Document: {query}

        # User: {input}
        # Ai: 
        # """

        # # now create the few shot prompt template
        # self.few_shot_prompt_template = FewShotPromptTemplate(
        #     examples=[{"Document": read_pdf("RadarInfo/RadarHandbook_CH001.pdf")}],
        #     example_prompt=self.PROMPT,
        #     prefix=self.prefix,
        #     suffix=self.suffix,
        #     input_variables=["history","query","input"],
        #     example_separator="\n\n"
        # )

        
        # initialize HF LLM
        self.flan_t5 = HuggingFaceHub(
            repo_id="google/flan-t5-xxl",
            model_kwargs={"temperature":0.1}
        )

        self.conversation = ConversationChain(
            prompt=self.PROMPT,
            llm=self.flan_t5,
            memory=ConversationBufferWindowMemory(k=4)
        )
    def run(self, text):
        return([self.conversation(text)["response"], self.conversation.memory.buffer])

chatbot = model()
print(chatbot.run("How are you?"))