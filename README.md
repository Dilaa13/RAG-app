# Retrieval-Augmented Generation (RAG) Application

### Overview
This project implements a **Retrieval-Augmented Generation (RAG)** system using the **LangChain** framework and the **Chroma** vector database.

The application allows a user to upload or store text documents (e.g., company policies, meeting notes) and then ask natural-language questions.  
The system retrieves the most relevant document chunks from the vector database and uses an OpenAI model to generate accurate, context-aware answers.

---

### Features Implemented

✅ **LangChain Framework**
- Used `langchain_core`, `langchain_community`, and `langchain_openai` for document and model integration.  

✅ **Chroma Vector Database**
- Stores document embeddings locally in a persistent folder (`chroma_db/` and `chroma.sqlite3`).  
- Supports semantic retrieval through similarity search.

✅ **Document Loader**
- Loads all `.txt` files from the `data/` folder automatically.  
- Supports simple chunking for better embedding and retrieval.

✅ **Retrieval-Augmented Generation Flow**
- Retrieves top-k most relevant text chunks for each user question.  
- Builds a context-aware prompt and generates an answer using an OpenAI model.  
- Displays both the answer and the document sources used.

✅ **Environment Management**
- Uses an environment variable `OPENAI_API_KEY` for API access.  
- Fully functional in local development environments.

---

### Folder Structure

RAG-app/
│
├── data/ # Folder containing text files (policy.txt, notes.txt, etc.)
├── chroma_db/ # Auto-created Chroma vector database
├── rag_app.py # Main application script
└── README.md # Project documentation


---

### How to Run

1. **Install dependencies**
   ```bash
   pip install langchain langchain-community langchain-openai chromadb openai

2. **Set your OpenAI API key**
   setx OPENAI_API_KEY "your_api_key_here"

3. **Prepare your data**
   Place your text files in the data/ folder (e.g., policy.txt, notes.txt).

4. **Run the application**
   python rag_app.py

5. **Ask questions**
   Your question (or 'exit'): What are the remote work policies?


**Example Interaction**

RAG app ready with OpenAI. Ask a question.
Your question (or 'exit'): What is Project Orion?

Answer:
Project Orion is the upcoming AI-based productivity system currently under early-stage development.

Sources:
- notes.txt


**Improvements to be Added**

🔸 1. Web or GUI Interface
      Build a simple Streamlit or Flask interface for easier interaction.

🔸 2. Model Configuration
      Allow the user to choose different OpenAI models or adjust retrieval parameters (k, chunk size).

🔸 3. Reporting / Logging
      Record questions, answers, and sources for analytics or debugging.


**Author**
Developed by Charith Dhananjaya

