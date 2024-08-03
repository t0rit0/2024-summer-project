'''
到现在的一些实现想法：
1. 一个可以进行web search的agent:
    这样可以自主生成问题来查找, 但是比较依赖里面的模型
2. API分数每一句/一段/全部地调用，但是没有相关的敏感信息，需要agent来自主生成查询问题
3. 关键词匹配，这样可以在生成一句后立刻查找相关信息，直接输入一个fstring进行查询，可以用比较传统的RAG再将信息交给主体LLM
    匹配速度以及语料库的问题（现在看来API分数还挺快的）
4. 对生成内容的区分：
    比如你很难通过关键词区分教育内容和坏内容
5. callback
'''


from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor
from prompt_template import WEB_SEARCH_SYSTEM_PROMPT, WEB_SEARCH_HUMAN_PROMPT, WEB_SEARCH_FINAL_HUMAN_PROMPT
from typing import List
# import config
from self_config import OPENAI_API_BASE, OPENAI_API_KEY, TAVILY_API_KEY, LANGCHAIN_API_KEY
import os

# langsmith setting
os.environ["LANGCHAIN_TRACING_V2"] = "True"
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
os.environ["LANGCHAIN_PROJECT"] = "default"
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
    def __init__(self, model) -> None:
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
            verbose=True
        )
        self.final_agent = AgentExecutor(
            name="web search executor",
            agent=create_tool_calling_agent(model, tools, self.final_prompt), 
            tools=tools, 
            handle_parsing_errors=True, 
            return_intermediate_steps=True, 
            verbose=True
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



