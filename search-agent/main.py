from dotenv import load_dotenv

load_dotenv()


from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from tavily import TavilyClient
from langchain_tavily import TavilySearch
# from schemas import AgentResponse


tavily = TavilyClient()


# @tool
# def search(query: str) -> str:
#     """
#     Search the internet using Tavily.
#     Args:
#         query: Search query
#     Returns:
#         Concise search results
#     """
#     print(f"\nSearching for: {query}")

#     response = tavily.search(
#         query=query,
#         max_results=1,
#         search_depth="basic",
#         include_answer=False
#     )

#     results = response.get("results", [])

#     # Limit content to reduce token usage
#     formatted_results = []

#     for item in results:
#         formatted_results.append(
#             f"""
#             Title: {item.get('title', '')}
#             URL: {item.get('url', '')}

#             """
#         )

#     return "\n".join(formatted_results)





llm = ChatGroq(temperature=0, model="openai/gpt-oss-20b" ,  max_tokens=500)
tools = [ TavilySearch() ]
agent = create_agent(model=llm,tools=tools )

# def main():
#     print("Hello from search-agent!")
#     result = agent.invoke({"messages":HumanMessage(content="search for 1 job postings for an ai engineer using langchain in the india on linkedin and list their details")})
#     print(result)


def main():
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "search for 1 job postings for an ai engineer in india on linkdin",
                }
            ]
        }
    )
    # Access structured response from the agent
    structured = result.get("structured_response", None)
    print(structured if structured is not None else result)


if __name__ == "__main__":
    main()
