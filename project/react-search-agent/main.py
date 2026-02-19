from dotenv import load_dotenv
import sys
sys.stdout.reconfigure(encoding="utf-8")
load_dotenv()
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch


llm = ChatOllama(model="llama3.2")
my_tools=[TavilySearch()]
agent = create_agent(model=llm,tools=my_tools) 
def main():
    print("Hello from Langchain-course")
    result = agent.invoke({"messages":HumanMessage(content="search for 3 job postings for an ai engineer using langchain in the bay area on linkedin anf list their details ")})
    print(result)


if __name__ == "__main__":
    main()