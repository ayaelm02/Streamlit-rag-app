import os
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
import streamlit as st

# Load environment variables
load_dotenv()

# Persistent directory for Chroma DB
persist_directory = "./rag_env/pdf_chroma_db"

# Streamlit UI
st.title("Streamlit RAG Application (Ollama & Open-Source Models)")
st.sidebar.title("Settings")

# Define available models
models = {
    "Ollama Llama 3.1": {
        "type": "ollama",
        "model": "llama3.1",
        "base_url": os.getenv("OLLAMA_BASE_URL")
    },
    "Flan-T5 (Small)": {
        "type": "seq2seq",
        "name": "google/flan-t5-small"
    },
    "MiniLM (Embeddings)": {
        "type": "embedding",
        "name": "sentence-transformers/all-MiniLM-L6-v2"
    }
}

# Sidebar to select LLM
selected_model = st.sidebar.selectbox("Select Model", options=models.keys())
model_config = models[selected_model]

# Initialize the selected model
if model_config["type"] == "ollama":
    llm = Ollama(model=model_config["model"], base_url=model_config["base_url"])
    embed_model = OllamaEmbeddings(model=model_config["model"], base_url=model_config["base_url"])
elif model_config["type"] == "seq2seq":
    tokenizer = AutoTokenizer.from_pretrained(model_config["name"])
    model = AutoModelForSeq2SeqLM.from_pretrained(model_config["name"])
elif model_config["type"] == "causal":
    tokenizer = AutoTokenizer.from_pretrained(model_config["name"])
    model = AutoModelForCausalLM.from_pretrained(model_config["name"])
elif model_config["type"] == "embedding":
    embedding_model = SentenceTransformer(model_config["name"])

# Initialize or load Chroma vector store
if model_config["type"] in ["ollama", "embedding"]:
    vector_store = Chroma(
        persist_directory=persist_directory,
        embedding_function=embed_model.encode if model_config["type"] == "ollama" else embedding_model.encode
    )
else:
    vector_store = Chroma(persist_directory=persist_directory)

retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# Document upload
st.sidebar.subheader("Add Your Documents")
uploaded_files = st.sidebar.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        loader = PyPDFLoader(uploaded_file)
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(documents)
        vector_store.add_documents(chunks)
        vector_store.persist()
    st.sidebar.success("Documents added successfully!")

# Chat interface
st.subheader("Chat with RAG")
question = st.text_input("Enter your question here")

if st.button("Get Answer"):
    if question:
        with st.spinner("Generating response..."):
            # Generate response using the selected model
            if model_config["type"] == "ollama":
                response = llm.invoke({"prompt": question})
                response_text = response.get("answer", "No response generated.")
            elif model_config["type"] == "seq2seq":
                inputs = tokenizer(question, return_tensors="pt")
                outputs = model.generate(inputs["input_ids"], max_length=100)
                response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            elif model_config["type"] == "causal":
                inputs = tokenizer(question, return_tensors="pt")
                outputs = model.generate(inputs["input_ids"], max_length=100)
                response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            elif model_config["type"] == "embedding":
                retrieval_chain = RetrievalQA.from_chain_type(llm=None, retriever=retriever)
                response_text = retrieval_chain.run(question)

            st.write("**Response:**", response_text)
    else:
        st.error("Please enter a question.")
