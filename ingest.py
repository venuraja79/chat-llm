"""Load html from files, clean up, split, ingest into Weaviate."""
import pickle

from langchain.document_loaders import ReadTheDocsLoader, PyPDFLoader, DirectoryLoader
from langchain.embeddings import OpenAIEmbeddings, CohereEmbeddings, SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS


def ingest_docs():
    """Get documents from web pages."""
    #loader = ReadTheDocsLoader("langchain.readthedocs.io/en/latest/")
    print("Loading documents and preprocessing them...")
    loader = DirectoryLoader("./docs")
    raw_documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    documents = text_splitter.split_documents(raw_documents)
    embeddings = SentenceTransformerEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)

    # Save vectorstore
    print("Writing to vector store")
    with open("vectorstore.pkl", "wb") as f:
        pickle.dump(vectorstore, f)


if __name__ == "__main__":
    ingest_docs()
