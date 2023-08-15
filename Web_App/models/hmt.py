#imports

import torch
import transformers
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate

from langchain import OpenAI, LLMChain, LLMMathChain
from langchain.tools import DuckDuckGoSearchRun
from langchain.tools import WikipediaQueryRun
from langchain.utilities import WikipediaAPIWrapper

from typing import List, Union
from langchain.schema import AgentAction, AgentFinish
import re
import langchain
from transformers import GenerationConfig, pipeline
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.tools import ShellTool
from langchain.tools import HumanInputRun
from langchain.agents import create_pandas_dataframe_agent
# import chess
# import chess.engine
from stockfish import Stockfish
import re
from _OpalLLM import OpalLLM

class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        if "As an AI language model, I cannot" in llm_output:
            raise SyntaxError(f"Output does not follow prompt: {llm_output}")
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)
    
class HMT():
    def __init__(self):
        
        # Use Opal Server
        self.model_name="tiiuae/falcon-40b-instruct"
        self.local_llm=OpalLLM(model=self.model_name,
                        temperature=0.1,
                        top_k=60,
                        top_p=0.95,
                        max_tokens=500,
                        repetition_penalty=1.15)

        # Define which tools the agent can use to answer user queries
        # search = DuckDuckGoSearchRun()
        # wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
        # shell_tool = ShellTool()
        # shell_tool.description = shell_tool.description + f"args {shell_tool.args}".replace(
        #     "{", "{{"
        # ).replace("}", "}}")
        # self_ask_with_search = initialize_agent(
        #     [shell_tool], self.local_llm, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True
        # )

        # def google_search(input_text):
        #     search_results = search.run(f"site:google.com {input_text}")
        #     return search_results

        # def chess_guide(input_text):
        #     search_results2 = search.run(f"site:chess.com {input_text}")
        #     return search_results2


        # def shell(input_text):
        #     search_results4 = self_ask_with_search.run(input_text)
        #     return search_results4


        # def chess_moves(input_text):
        #     stockfish = Stockfish('/workspace/pv-data/InternFolders/Niko/stockfish-ubuntu-x86-64-modern')

        #     moves = input_text.split()
        #     moves_clean = []
        #     for i in moves:
        #         x = i.find(".")
        #         if x != -1:
        #             moves_clean.append(i[x+1:])
        #         else:
        #             moves_clean.append(i)
            
        #     board = chess.Board()
        #     for move in moves_clean:
        #         board.push_san(move)
        #     fen = board.fen()

        #     stockfish.set_fen_position(fen)
        
        #     best_move = stockfish.get_best_move()

        #     return best_move
        

        # llm_math_chain = LLMMathChain.from_llm(llm=self.local_llm, verbose=True)
        self.tools = []
        #     Tool(
        #         name = "Search Google",
        #         func=google_search,
        #         description="useful for getting answers to general questions"
        #     ),
        #     Tool(
        #         name="Calculator",
        #         func=llm_math_chain.run,
        #         description="useful for when you need to answer arithmetic questions"
        #     ),
        #     Tool(
        #         name="Chess Search",
        #         func=chess_guide,
        #         description="useful for general chess questions related to openings or pieces"
        #     ),
        #     Tool(
        #         name="Shell Tool",
        #         func=shell,
        #         description = "useful for interacting with the local file system and using shell commands"
        #     ),
        #     Tool(
        #         name="Chess Move Predict",
        #         func=chess_moves,
        #         description="useful for predicting the next chess move when given a set of moves in standard algebraic notation"
        #     )
        
        # Set up the base template
        self.template = """Answer the following questions as best you can. You have access to the following tools:

            {tools}

            Use the following format:

            Question: the input question you must answer
            Thought: you should always think about what to do
            Action: the action to take, should be one of [{tool_names}]
            Action Input: the input to the action
            Observation: the result of the action
            ... (this Thought/Action/Action Input/Observation can repeat N times)
            Thought: I now know the final answer
            Final Answer: the final answer to the original input question

            Here are some examples for this format:

            Question: What is 5 * 1234?
            Thought: This is an arithmetic question. I should use the Calculator
            Action: Calculator
            Action Input: 5 * 1234
            Observation: 5 * 1234 is 6170
            Thought: I now know the final answer
            Final Answer: 5 * 1234 is 6170

            Question: I am playing a chess game with the following moves in algebraic notation: e4 e5 f4 . I am black, what should be my next move? 
            Thought: I should use the Sheel tool to download stockfish to find the next move
            Action: Shell Tool
            Action Input: !pip install stockfish
            Observation: (successful installation message)
            Thought: I have now installed Stockfish and can use it to predict the next move
            Final Answer: Move your pawn from e5 to f4

            Question: What happens to the pawn when it reaches the end of the chess board?
            Thought: I should find out what happens when a pawn gets to the other side of the chess board
            Action: Chess Search
            Action Input: What happens to a pawn when it reaches the other side of the chess board
            Observation: If the Pawn reaches the opposite side of the chessboard, it has the unique ability to promote to another piece. The pawn can become a Queen, Bishop, Rook, or Knight. There are no restrictions to how many pieces of a given type you can have via promotion.
            Thought: I now know the final answer
            Final Answer: Pawn Promotion is one of the special moves in chess. It happens when a Pawn reaches the opponent's back rank (first row of the opponent) and then it is replaced by any piece a player decides to, except The King

            Begin! Remember to answer as a professional chess player and use the template when giving your final answer.

            Question: {input}
            {agent_scratchpad}"""
        

        # Set up a prompt template
        class CustomPromptTemplate(StringPromptTemplate):
            # The template to use
            template: str
            # The list of tools available
            tools: List[Tool]
            
            def format(self, **kwargs) -> str:
                # Get the intermediate steps (AgentAction, Observation tuples)
                # Format them in a particular way
                intermediate_steps = kwargs.pop("intermediate_steps")
                thoughts = ""
                for action, observation in intermediate_steps:
                    thoughts += action.log
                    thoughts += f"\nObservation: {observation}\nThought: "
                # Set the agent_scratchpad variable to that value
                kwargs["agent_scratchpad"] = thoughts
                # Create a tools variable from the list of tools provided
                kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
                # Create a list of tool names for the tools provided
                kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
                return self.template.format(**kwargs)
            
        self.prompt = CustomPromptTemplate(
            template=self.template,
            tools=self.tools,
            # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
            # This includes the `intermediate_steps` variable because that is needed
            input_variables=["input", "intermediate_steps"]
        )

        self.llm_chain = LLMChain(llm=self.local_llm, prompt=self.prompt)
        self.tool_names = [tool.name for tool in self.tools]

        self.agent = LLMSingleActionAgent(
            llm_chain=self.llm_chain, 
            output_parser=CustomOutputParser(),
            stop=["\nObservation:"], 
            allowed_tools=self.tool_names
        )
        
        self.agent_executor = AgentExecutor.from_agent_and_tools(agent=self.agent, 
                                                    tools=self.tools,
                                                    verbose=True
                                                    )
    def predict(self, query):
        return self.agent_executor.run(query)

query = HMT()
print(query.predict("I am playing a chess game with the following moves in algebraic notation: b4 b5 c4 . I am black, what should be my next move?"))


    
# class my_class():
#     def __init__(self, mode):
#         self.flag = True
#         if mode == "add":
#             self.flag = True
#         elif mode == "multiply":
#             self.flag = False
#     def use(self, x, y):
#         if self.flag:
#             return x+y
#         elif not self.flag:
#             return x*y

# my_obj = my_class("multiply")
# print(my_obj.use(4, 5))
