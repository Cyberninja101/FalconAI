# FALCON Chatbot

<img src="/ReadMe_images/falcon_logo.png" height="90" align="right" margin-right="10px">

FALCON is an open source chatbot fine-tuned on Open-Llama 7B v2 for RADAR applications. This chatbot was created by four high schoolers during the summer of 2023 as part of the JHU Applied Physics Laboratory's (APL) [ASPIRE](https://secwww.jhuapl.edu/stem/aspire/) program. FALCON chatbot assists APL users in the AMDS (Air & Missle Defence Sector) with day-to-day tasks in secure/confidential working environments with no internet access. With FALCON, APL users can have more confidence giving the AI more complicated tasks, freeing their time for more creative endeavors. 


Website: [http://falconai.outer.jhuapl.edu](http://falconai.outer.jhuapl.edu/)

*Disclaimer: This is a secure website from APL, and is only available in specific whitelisted IP address areas.*

- Original model: Open-Llama v2 (7 billion parameters)
- Fine-tuned on: radar textbooks and journals, threats briefs
- Protected with: LangChain gaurdrails (NSFW, jailbreaking)
- Built with: ReAct framework Human Machine Teaming capabilities
- Constructed with: Flask (Python web framework)

**Put in a photo of the Web App UI, maybe with some box shadows when it looks nice, remind Zenchang pls**

## Features
- Normal Mode
    - The normal mode uses the fine-tuned open-llama model, and is best suited for answering general questions relating to radars. With conversational memory, users can ask questions and converse with the model. Users can also ask questions outside of radars, but the responses might not be as accurate.
- Document Mode
    - The document ingest mode allows users to upload multiple documents to FALCON. Using the llama 13B **Kenny, confirm the final model** model from OPAL, users can ask specific questions about the text, such as "What are the top 3 threats I need to know about from North Korea as a radar specialist?", or more general questions, such as "summarize and list the main points of this document".
- Explanation Mode
    - The explanation mode activates HMT (Human Machine Teaming), built with the ReAct framework, prompting the model to explain how it got to its answers. With more advanced AI reasoning, users can have more confidence giving the AI more complicated tasks, freeing their time for more creative endeavors. 
- Strong CounterAI Gaurdrails
    - To prevent the model from outputting illegal, profane, or otherwise NSFW content, we implemented AI guardrails. By using language model chaining via LangChain, we are able to determine if input prompts fall into the above categories without impacting model performance.

## How to use FALCON Chatbot


## ASPIRE Poster
<img src="/ReadMe_images/FALCON_POSTER.png">

## Credits

Contributors:
- Zenchang Sun
- Kenneth Wang
- Justin Sykes
- Niko Tabernero

APL Mentors
- Michael Tabernero
- Josh McCarter
- Ryan Allen
- Edwina Liu


