import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

if __name__ == "__main__":
    print("Ingesting data...")
    loader = TextLoader("mediumblog.txt", encoding="utf-8")
    document = loader.load()
    print("Splitting document...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"Trozos de texto: {len(texts)}")

    embbedings = OpenAIEmbeddings(model="text-embedding-3-small")
    PineconeVectorStore.from_documents(texts, embbedings, index_name=os.environ["INDEX_NAME"])
    print("Ingestion complete.")