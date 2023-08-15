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
# from stockfish import Stockfish
import re
from _OpalLLM import OpalLLM
    
class HMT():
    def __init__(self):
        
        # Use Opal Server
        self.model_name="tiiuae/falcon-40b-instruct"
        self.local_llm=OpalLLM(model=self.model_name,
                        temperature=0.1,
                        top_k=10,
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
        self.template = """Answer the following questions as best you can.

                Use the following format:
        
                Question: the input question you must answer
                Thought: list out every step you need to take to solve this question
                Step: go through one step in the list
                ... (Keep using Step to go through the list until you've gone through all the steps)
                Thought: I now know the final answer
                Final Answer: the final answer to the original input question
        
                Here are some examples for this format:
        
                Question: How do I bake a cake?
                Thought: 1. Find out what cooking tools needed
                         2. Find out what ingredients are needed
                         3. Find out how to make a cake batter
                         4. Find out the temperature needed to bake the cake
                         5. Put the cake in the oven
                Step: 1. To bake a cake you will need a mixer, a rubber spatulas, Measuring cups and spoons, Mixing bowls, Cake pans, Cooling racks, a whisk, a spatula, and an oven thermometer
                Step: 2. The ingredients you will need to bake a cake is Flour, Sugar, Eggs, Butter or oil, Baking powder or baking soda, and Milk or water
                Step: 3. Before making the ake batter grease and flour your cake pans, then mix together your dry ingredients in one bowl and your wet ingredients in another bowl, combine the dry and wet ingredients, and pour the batter into the prepared cake pans.
                Step: 4. The temperature needed to bake a cake is between 325°F and 450°F. For basic cake recipes, 350°F is the most common temperature.
                Step: 5. Now put the cake in the oven at the aforementioned temperature for an hour or until golden brown
                Thought: I now know the final answer
                Final Answer: First you would need to get the proper cooking tools to bake a cake, like a mixer, a rubber spatulas, Measuring cups and spoons, Mixing bowls, Cake pans, Cooling racks, a whisk, a spatula, and an oven thermometer
                    Then you would need to get the ingredients that are needed to bake the cake like Flour, Sugar, Eggs, Butter or oil, Baking powder or baking soda, and Milk or water
                    After you have all the cooking tools and ingredients needed, grease and flour your cake pans to make your cake batter, after that mix together your dry ingredients in one bowl and your wet ingredients in another bowl, combine the dry and wet ingredients, and pour the batter into the prepared cake pans.
                    The temperature needed to preheat the oven will be between 325°F and 450°F
        
                Question: What happens to the pawn when it reaches the end of the chess board?
                Thought: 1.Find out what happens when a pawn gets to the other side of the chess board, 
                         2.Find out what pawn promotion is.
                Step: 1. When a pawn reaches the end of the chess board pawn promotion happens. 
                Step: 2. Pawn Promotion is one of the special moves in chess. It happens when a Pawn reaches the opponent's back rank (first row of the opponent) and then it is replaced by any piece a player decides to, except The King
                Thought: I now know the final answer
                Final Answer: Pawn Promotion is one of the special moves in chess. It happens when a Pawn reaches the opponent's back rank (first row of the opponent) and then it is replaced by any piece a player decides to, except The King
                
                Begin! Remember to use the template when giving your final answer.
                
                Question: {input}
                {agent_scratchpad}"""
                # Question: What is the difference between a ninja and a samurai?
                # Thought: 1. Find out what ninjas did historically
                #          2. Find out what samurais did histtorically
                #          3. Compare the two, to find differences
                # Step: 1. Ninjas were trained as assassins and mercenaries and usually belonged to the lower classes of Japanese society. They used ninjutsu to disrupt enemies and gather information. The functions of a ninja included espionage, sabotage, infiltration, assassination and guerrilla warfare.
                # Step: 2. Samurai were part of an elite class of Japanese warriors who fought to defend their medieval lords. They were considered to be the best warriors in medieval Japan as they were well-trained in the art of battle.
                # Step: 3. The main difference between a ninja and a samurai is who they were and what they did. Samurai were part of an elite class of Japanese warriors who fought to defend their medieval lords. They were considered to be the best warriors in medieval Japan as they were well-trained in the art of battle. Ninjas, on the other hand, were trained as assassins and mercenaries and usually belonged to the lower classes of Japanese society. They used ninjutsu to disrupt enemies and gather information.
                # Thought: I now know the final answer
                # Final Answer: The main difference between a ninja and a samurai is who they were and what they did. Samurai were part of an elite class of Japanese warriors who fought to defend their medieval lords. They were considered to be the best warriors in medieval Japan as they were well-trained in the art of battle. Ninjas, on the other hand, were trained as assassins and mercenaries and usually belonged to the lower classes of Japanese society. They used ninjutsu to disrupt enemies and gather information.

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

        self.output_parser = CustomOutputParser()
        
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
            output_parser=self.output_parser,
            stop=["\nObservation:"], 
            allowed_tools=self.tool_names
        )
        
        self.agent_executor = AgentExecutor.from_agent_and_tools(agent=self.agent, 
                                                    tools=self.tools,
                                                    verbose=True
                                                    )
    def predict(self, query):
        return self.agent_executor.run(query)



