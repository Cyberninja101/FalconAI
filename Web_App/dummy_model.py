import os
from langchain import PromptTemplate, HuggingFaceHub, FewShotPromptTemplate
from langchain.chains import LLMChain, ConversationChain
from langchain.chains.conversation.memory import (ConversationBufferMemory, 
                                                  ConversationSummaryMemory, 
                                                  ConversationBufferWindowMemory,
                                                  ConversationKGMemory)
class model():
    def __init__(self):
        os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'

        # build prompt template for simple question-answering
        self.template = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from the Current conversation. 
            If the AI does not know the answer to a question, it truthfully says it does not know. 

            Current conversation:
            {history}

            Human: {input}
            AI: """

        self.PROMPT = PromptTemplate(input_variables=["history", "input"], template=self.template)


        # # now break our previous prompt into a prefix and suffix
        # # the prefix is our instructions
        # prefix = """The following are exerpts from conversations with an AI
        # assistant. The assistant is typically sarcastic and witty, producing
        # creative  and funny responses to the users questions. Here are some
        # examples: 
        # """
        # # and the suffix our user input and output indicator
        # suffix = """
        # User: {query}
        # AI: """

        # # now create the few shot prompt template
        # few_shot_prompt_template = FewShotPromptTemplate(
        #     examples=examples,
        #     example_prompt=self.template,
        #     prefix=prefix,
        #     suffix=suffix,
        #     input_variables=["query"],
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
            memory=ConversationBufferMemory(),
        )
    def run(self, text):
        return([self.conversation(text)["response"], self.conversation.memory.buffer])