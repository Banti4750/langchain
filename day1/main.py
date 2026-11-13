
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# Initialize Groq model
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.7,
    max_tokens=500
)

# Send a message to the LLM
response = llm.invoke("Explain LangChain in simple words.")

# Print the response
print(response.content)
