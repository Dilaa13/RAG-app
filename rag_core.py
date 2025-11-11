# rag_core.py
from pathlib import Path
from typing import List
import shutil

from langchain_core.documents import Document
# if you installed langchain-chroma, you can switch this import to:
# from langchain_chroma import Chroma
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

DATA_DIR = Path("data")
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "my_rag_collection"


def load_documents(data_dir: Path) -> List[Document]:
    docs = []
    for path in data_dir.glob("*.txt"):
        text = path.read_text(encoding="utf-8")
        docs.append(Document(page_content=text, metadata={"source": path.name}))
    return docs


def simple_chunk(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    parts = []
    start = 0
    n = len(text)
    while start < n:
        end = start + chunk_size
        parts.append(text[start:end])
        start = end - overlap
    return parts


def build_vectorstore(documents: List[Document]) -> Chroma:
    chunks = []
    for doc in documents:
        for part in simple_chunk(doc.page_content, 500, 100):
            chunks.append(Document(page_content=part, metadata=doc.metadata))

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
    )
    return vectordb


def get_vectorstore() -> Chroma:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectordb = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )
    return vectordb


def get_or_create_vectorstore() -> Chroma:
    if not Path(CHROMA_DIR).exists():
        if not DATA_DIR.exists():
            raise FileNotFoundError("data/ folder not found.")
        docs = load_documents(DATA_DIR)
        return build_vectorstore(docs)
    return get_vectorstore()


def rebuild_vectorstore() -> Chroma:
    # delete the whole chroma folder safely
    if Path(CHROMA_DIR).exists():
        shutil.rmtree(CHROMA_DIR, ignore_errors=True)

    docs = load_documents(DATA_DIR)
    return build_vectorstore(docs)


def answer_question(question: str):
    vectordb = get_or_create_vectorstore()
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    docs = retriever.invoke(question)
    context = "\n\n".join(d.page_content for d in docs)
    prompt = (
        "You are an assistant. Use ONLY the context to answer.\n"
        "If the answer is not in the context, say so.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\nAnswer:"
    )

    resp = llm.invoke(prompt)
    sources = [d.metadata.get("source", "unknown") for d in docs]
    return resp.content, sources
