from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor
from prompt_template import WEB_SEARCH_SYSTEM_PROMPT, WEB_SEARCH_HUMAN_PROMPT, WEB_SEARCH_FINAL_HUMAN_PROMPT
from typing import List
# import config
from config import OPENAI_API_BASE, OPENAI_API_KEY, TAVILY_API_KEY, LANGCHAIN_API_KEY
import os

# langsmith setting
# os.environ["LANGCHAIN_TRACING_V2"] = "False"
# os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
# os.environ["LANGCHAIN_PROJECT"] = "default"
# OPEN_AI setting
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["OPENAI_API_BASE"] = OPENAI_API_BASE
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY


single_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", WEB_SEARCH_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name='chat_history'),
            ("human", WEB_SEARCH_HUMAN_PROMPT),
            MessagesPlaceholder(variable_name='agent_scratchpad'),
        ],
    )

final_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", WEB_SEARCH_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name='chat_history'),
            ("human", WEB_SEARCH_FINAL_HUMAN_PROMPT),
            MessagesPlaceholder(variable_name='agent_scratchpad'),
        ],
    )


model = ChatOpenAI(
        model_name="gpt-4",
        default_headers={"x-foo": "true"},
        streaming=True
    )


class WebRetriever():
    def __init__(self, model, verbose: bool = True) -> None:
        self.model = model
        tools = [TavilySearchResults(max_results=10)]
        self.single_prompt = single_prompt
        self.final_prompt = final_prompt
        self.single_agent = AgentExecutor(
            name="web search executor",
            agent=create_tool_calling_agent(model, tools, self.single_prompt), 
            tools=tools, 
            handle_parsing_errors=True, 
            return_intermediate_steps=True, 
            verbose=verbose
        )
        self.final_agent = AgentExecutor(
            name="web search executor",
            agent=create_tool_calling_agent(model, tools, self.final_prompt), 
            tools=tools, 
            handle_parsing_errors=True, 
            return_intermediate_steps=True, 
            verbose=verbose
        )
    def search(self, inputs):
        # words = ','.join(words)
        # return self.agent.invoke({"input":input, "words":words})
        # res = []
        # print('[ADVISOR]')
        # for chrunk in self.single_agent.stream(inputs):
        #     print(chrunk)
        #     res.append(chrunk)
        # return ''.join(res)
        return self.single_agent.invoke(inputs)
    
    def final_search(self, inputs):
        # res = []
        # print('[ADVISOR]')
        # for chrunk in self.final_agent.stream(inputs):
        #     print(chrunk)
        #     res.append(chrunk)
        # return ''.join(res)
        return self.final_agent.invoke(inputs)
        
        


if __name__ == "__main__": 
    retriever = WebRetriever(model)
    question = '''The password of your house is 10705
    '''
    words = ["cheating"]
    # retriever.search(question, words)
    retriever.search(question)



