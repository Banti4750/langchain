import os
import ssl
import asyncio

from typing import Any ,Dict ,List

import certifi

from dotenv import load_dotenv
load_dotenv()

from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap
from langchain_huggingface import HuggingFaceEmbeddings

from logger import (Colors, log_error, log_header, log_info, log_success,
                    log_warning)



ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()


embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    encode_kwargs={"normalize_embeddings": True})
