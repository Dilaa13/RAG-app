from pathlib import Path

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

DATA_DIR = Path("data")
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "my_rag_collection"


def load_documents(data_dir: Path):
    docs = []
    for path in data_dir.glob("*.txt"):
        text = path.read_text(encoding="utf-8")
        docs.append(Document(page_content=text, metadata={"source": path.name}))
    return docs


def simple_chunk(text: str, chunk_size: int = 500, overlap: int = 100):
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


def build_vectorstore(documents):
    # manual chunking
    chunked_docs = []
    for doc in documents:
        parts = simple_chunk(doc.page_content, 500, 100)
        for part in parts:
            chunked_docs.append(
                Document(page_content=part, metadata=doc.metadata)
            )

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vectordb = Chroma.from_documents(
        documents=chunked_docs,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
    )
    vectordb.persist()
    return vectordb


def get_vectorstore():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectordb = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )
    return vectordb


def make_prompt(question: str, docs: list[Document]) -> str:
    context = "\n\n".join(d.page_content for d in docs)
    prompt = (
        "You are an assistant that answers using the provided context only.\n"
        "If the answer is not in the context, say you cannot find it.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer in clear English."
    )
    return prompt


def main():
    # build db if missing
    if not Path(CHROMA_DIR).exists():
        if not DATA_DIR.exists():
            raise FileNotFoundError("data/ folder not found. Create it and add .txt files.")
        docs = load_documents(DATA_DIR)
        vectordb = build_vectorstore(docs)
    else:
        vectordb = get_vectorstore()

    retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    print("RAG app ready with OpenAI. Ask a question.")
    while True:
        question = input("\nYour question (or 'exit'): ").strip()
        if question.lower() in ("exit", "quit"):
            break

        # 1. retrieve
        retrieved_docs = retriever.invoke(question)

        # 2. build prompt
        prompt = make_prompt(question, retrieved_docs)

        # 3. call OpenAI
        answer = llm.invoke(prompt)

        # 4. show answer
        print("\nAnswer:\n", answer.content)

        # 5. show sources
        print("\nSources:")
        for d in retrieved_docs:
            print("-", d.metadata.get("source", "unknown"))


if __name__ == "__main__":
    main()
