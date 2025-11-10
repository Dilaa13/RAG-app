import os
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI  # <- new

# ------------ settings ------------
DATA_DIR = Path("data")
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "my_rag_collection"
# ----------------------------------


def load_documents(data_dir: Path):
    docs = []
    for path in data_dir.glob("*.txt"):
        loader = TextLoader(str(path), encoding="utf-8")
        docs.extend(loader.load())
    return docs


def build_vectorstore(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )
    chunks = splitter.split_documents(documents)

    # we can keep Hugging Face embeddings for now, they are already installed
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
    )
    vectordb.persist()
    return vectordb


def get_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectordb = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )
    return vectordb


def build_llm():
    # model name can be changed to what your account supports
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )


def build_qa_chain(vectordb):
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    llm = build_llm()
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )
    return qa_chain


def main():
    if not Path(CHROMA_DIR).exists():
        if not DATA_DIR.exists():
            raise FileNotFoundError("data/ folder not found. Create it and add .txt files.")
        raw_docs = load_documents(DATA_DIR)
        vectordb = build_vectorstore(raw_docs)
    else:
        vectordb = get_vectorstore()

    qa_chain = build_qa_chain(vectordb)
    print("RAG app ready with OpenAI. Ask a question.")

    while True:
        query = input("\nYour question (or 'exit'): ").strip()
        if query.lower() in ("exit", "quit"):
            break

        result = qa_chain.invoke(query)
        print("\nAnswer:\n", result["result"])

        print("\nSources:")
        for doc in result["source_documents"]:
            print("-", doc.metadata.get("source", "unknown"))


if __name__ == "__main__":
    main()
