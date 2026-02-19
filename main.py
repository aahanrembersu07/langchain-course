import os
from dotenv import load_dotenv
from pathlib import Path
from langsmith import traceable

# --- Load .env ---
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

print("Tracing:", os.getenv("LANGSMITH_TRACING"))

from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama


# --- Force LangSmith tracing ---
@traceable
def run_chain(chain, information):
    return chain.invoke({"information": information})


def main():
    print("Hello from langchain-course!")

    information = """
    Elon Reeve Musk (born June 28, 1971) is a businessman and entrepreneur known for his leadership of Tesla, SpaceX, Twitter, and xAI.
    """

    summary_template = """
    Given the information {information} about a person I want you to create:
    1. A short summary
    2. Two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template
    )

    llm = ChatOllama(
        temperature=0,
        model="gemma3:270m"
    )

    chain = summary_prompt_template | llm

    # 🔥 IMPORTANT: Use traced function
    response = run_chain(chain, information)

    print(response.content)


if __name__ == "__main__":
    main()


