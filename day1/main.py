
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Load environment variable
load_dotenv()

# Initialize Groq model
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    max_tokens=500
)

# Send a message to the LLM
# response = llm.invoke("Explain LangChain in simple words.")

# Create prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI tutor. Explain concepts simply."),
    ("human", "{question}")
])


# Create a chain
chain = prompt | llm

# Chat loop
while True:
    question = input("\nYou: ")

    if question.lower() in ["exit", "quit"]:
        break

    response = chain.invoke({"question": question})

    print("\nAI:", response.content)

# Print the response
# print(response.content)
