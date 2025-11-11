# streamlit_app.py
import streamlit as st
from pathlib import Path

from rag_core import (
    DATA_DIR,
    get_or_create_vectorstore,
    rebuild_vectorstore,
    answer_question,
)

st.set_page_config(page_title="RAG App", page_icon="📄")

st.title("RAG App (upload → index → ask)")
st.write("Upload .txt files. They will be indexed into Chroma, then you can ask questions.")

# make sure data/ exists
DATA_DIR.mkdir(exist_ok=True)

st.sidebar.header("Upload documents")
uploaded_files = st.sidebar.file_uploader(
    "Upload text files", type=["txt"], accept_multiple_files=True
)

if uploaded_files:
    for f in uploaded_files:
        content = f.read().decode("utf-8")
        target_path = DATA_DIR / f.name
        target_path.write_text(content, encoding="utf-8")
    with st.spinner("Indexing uploaded files..."):
        rebuild_vectorstore()
    st.sidebar.success("Files uploaded and indexed.")

# make sure we have a vector store
with st.spinner("Loading vector store..."):
    _ = get_or_create_vectorstore()

question = st.text_input("Your question")
ask = st.button("Ask")

if ask and question.strip():
    with st.spinner("Thinking..."):
        answer, sources = answer_question(question.strip())
    st.subheader("Answer")
    st.write(answer)

    st.subheader("Sources")
    for s in sources:
        st.write(f"- {s}")
elif ask:
    st.warning("Please enter a question.")
