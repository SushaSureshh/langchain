# TODO
# We will load a web page
# Split the docs 
# Convert to vectors and save it to the vactor db 
# Use the LLM to ask it questions
import os
from dotenv import load_dotenv, dotenv_values 
from langchain_openai import ChatOpenAI
import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
# loading variables from .env file
load_dotenv() 



# accessing and printing value
# print(os.getenv("OPEN_AI_API_KEY"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-3.5-turbo-0125", openai_api_key=OPENAI_API_KEY)


# Load, chunk and index the contents of the blog.
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()
# print(docs)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1000, chunk_overlap=200
# )
all_splits = text_splitter.split_documents(docs)
# print("First split",all_splits[0])
# print("Second split", all_splits[1])


# TODO: we need to index the docs now - that is convert the text to vectors
# TODO : need to save the vectors to index
# TODO: Use the chroma vector store

vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings(), openai_api_key=OPENAI_API_KEY)
retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

rag_chain.invoke("What is Task Decomposition?")
