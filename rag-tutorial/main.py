import os

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

print("Initializing components...")

# 1. Initialize embedding model
embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        encode_kwargs={"normalize_embeddings": True}
    )


# 2. Initialize Groq
llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0
    )

# 3. Connect to Pinecone
vectorstore = PineconeVectorStore(
        index_name=os.environ["INDEX_NAME"],
        embedding=embeddings
    )


prompt_template = ChatPromptTemplate.from_template(
        """Answer the question based only on the following context:

        {context}

        Question: {question}

        Provide a detailed answer:"""
)

def format_docs(docs):
    """Format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

def main():
    print("Retrieving...")
    print("Hello from rag-tutorial!")

     # Query
    query = "what is Pinecone in machine learning?"

    # ========================================================================
    # Option 0: Raw invocation without RAG
    # ========================================================================
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 0: Raw LLM Invocation (No RAG)")
    print("=" * 70)
    result_raw = llm.invoke([HumanMessage(content=query)])
    print("\nAnswer:")
    print(result_raw.content)



if __name__ == "__main__":
    main()
