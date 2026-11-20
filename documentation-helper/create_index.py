import os

from dotenv import load_dotenv
load_dotenv()

from pinecone import Pinecone, ServerlessSpec

def main():
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index_name = os.environ["INDEX_NAME"]

    if pc.has_index(index_name):
        print(f"index '{index_name}' found deleting ....")
        pc.delete_index(index_name)


    pc.create_index(
        name=index_name,
        vector_type="dense",
               dimension=384, # BAAI/bge-small-en-v1.5"

               metric="cosine",
               spec=ServerlessSpec(cloud="aws", region="us-east-1"),

    )

    print(f"Index '{index_name}' created")

if __name__ == "__main__":
    main()