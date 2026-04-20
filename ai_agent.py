#Step1: Setup API keys for Groq OPENAI and Tavily
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

#Step2: Set up LLM and tools
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

openai_llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=OPENAI_API_KEY,
    temperature=0.2
)

groq_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=GROQ_API_KEY,
    temperature=0.2
)

tavily_tool = TavilySearchResults(
    api_key=TAVILY_API_KEY,
    max_results=3
)

search_tools = TavilySearchResults(max_results=3)


#Step3: Setup AI agent with search tool functionality
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import AIMessage

system_prompt = "Act as a AI chatbot who is smart and friendly"

def get_response_from_ai_agent(llm_id, query, allow_search, system_prompt, provider):
    if provider == "groq":
        llm = ChatGroq(model =llm_id)
    elif provider == "openai":
        llm = ChatOpenAI(model =llm_id)
    else:
        raise ValueError("Invalid provider")    
    
    tools = [TavilySearchResults(max_results=3)] if allow_search else []

    agent = create_react_agent(
        model=llm_id,
        tools=tools,
        prompt=system_prompt
    )
    
    state = {"messages": query}
    response = agent.invoke(state)
    messages = response.get("messages")
    ai_message = [message.content for message in messages if isinstance(message, AIMessage)]
    return ai_message[-1]


# agent = create_react_agent(
#     model=groq_llm,
#     tools=[search_tools],
#     prompt=system_prompt
# )

# query = "Tell me about the latest news on AI and Machine Learning."
# state = {"messages": query}
# response = agent.invoke(state)
# messages = response.get("messages")
# ai_message = [message.content for message in messages if isinstance(message, AIMessage)]
# print("AI Message: ",ai_message[-1])

