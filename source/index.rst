.. Streamlit-rag-app documentation master file, created by
   sphinx-quickstart on Mon Dec  2 04:52:11 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Streamlit RAG Application
==========================

Welcome to the **Streamlit RAG Application** documentation! This project provides an easy-to-use web interface that integrates Retrieval-Augmented Generation (RAG) with powerful language models. It is designed to allow users to upload documents (PDFs), interact with them through a chatbot, and retrieve information with the help of vector databases.

Features
--------
- **Document Upload**: Upload multiple PDF documents to be processed.
- **Retrieval-Augmented Generation (RAG)**: The application integrates language models to retrieve relevant information from the uploaded documents based on user queries.
- **Language Support**: Answers can be provided in multiple languages, including Arabic.
- **User-Friendly Interface**: Built using Streamlit for seamless interaction.
- **Vector Database**: Uses Chroma as the vector store for efficient document retrieval.

Getting Started
---------------

### Prerequisites

Before running the Streamlit RAG Application, make sure you have the following installed:

- **Python 3.9+** (or your preferred version)
- **Conda** (for environment management)
- **Streamlit** (for running the app)
- **Ollama** (for LLM integration)
- **Chroma** (for vector database storage)
- **Sphinx** (for building documentation)
  
### Installation

To get started with the Streamlit RAG Application, follow these steps:

#### 1. Clone the Repository

Clone the repository to your local machine:

```bash
git clone https://github.com/yourusername/Streamlit-rag-app.git
cd Streamlit-rag-app

#### 2. Set Up Your Environment

We recommend using Conda to manage your environment. If you don't have Conda installed, download and install Miniconda or Anaconda.

Create a new Conda environment:

```bash
conda create --name Rag_env python=3.9
conda activate Rag_env

#### 3. Install Dependencies

Install the required dependencies from the requirements.txt file:

```bash
pip install -r requirements.txt

#### 4. Run the Application

To run the application locally, use the following command:

```bash
streamlit run app.py

After this, the app will be available at http://localhost:8501 in your browser.

Usage
-----

### 1. Upload Documents

- On the sidebar, click **"Upload PDFs"** and select the PDF files you want to use in the application.
- The documents will be processed and stored in the **Chroma** vector store. After processing, you can query them through the app.

### 2. Ask Questions

- Enter your question in the text input field under **"Chat with RAG"**.
- Click the **"Get Answer"** button to submit your query.
- The application will use the RAG model to retrieve relevant information from the uploaded documents and generate an answer. It utilizes document retrieval and language generation to formulate an accurate response.

### 3. Language Support

The application can handle multi-language queries. If the query includes the keyword "Arabic", the response will be provided in Arabic. Similarly, you can extend the application to other languages as needed by adjusting the language prompt accordingly.

---

Technical Architecture
----------------------

The **Streamlit RAG Application** utilizes a combination of several components to provide a smooth experience for document upload, processing, and answering queries. The key components of the architecture are as follows:

### 1. Document Loader

The **`PyPDFLoader`** is used to extract text from PDF documents. It processes the content of uploaded PDFs, converting them into text that can be indexed and searched. The loader ensures that each page of the PDF is loaded correctly for further processing.

- **Libraries used**: `PyPDFLoader` (for PDF extraction)
- **Function**: Converts PDF content into documents that can be split and processed.

### 2. Text Splitting

To make the documents suitable for efficient retrieval, the **`RecursiveCharacterTextSplitter`** is used. This component splits the documents into smaller, manageable chunks (e.g., 1000 characters each). This ensures better performance when retrieving information based on queries.

- **Libraries used**: `RecursiveCharacterTextSplitter` (for splitting text into chunks)
- **Function**: Divides large documents into smaller parts for efficient indexing.

### 3. Vector Database

For fast and efficient retrieval of document chunks, the application uses **Chroma** as the vector store. The vector store holds document embeddings (numerical representations of document content) and allows for similarity-based search.

- **Libraries used**: `Chroma` (for storing document embeddings)
- **Function**: Stores text chunks as vector embeddings for efficient retrieval.

#### How it works:
- **Embedding Generation**: Embeddings are generated using the **`OllamaEmbeddings`** model, which transforms text into numerical vectors. This process helps the model understand the semantic meaning of text.
- **Search**: When a query is entered, the application retrieves the most relevant document chunks using Chroma’s similarity-based search algorithm.

### 4. Retriever

The **Retriever** uses Chroma's search capabilities to find the most relevant document chunks based on the user’s query. It allows the application to perform an efficient search and retrieve the top-k relevant chunks.

- **Libraries used**: `Chroma.as_retriever` (for retrieval)
- **Function**: Fetches relevant document chunks based on the user's input.

### 5. Retrieval-Augmented Generation (RAG)

RAG is a hybrid approach that combines information retrieval with language generation. The **`Ollama`** model is used to generate responses based on the retrieved information. It first retrieves relevant chunks from the vector store and then uses the language model to generate a fluent and relevant answer.

- **Libraries used**: `Ollama` (for generating responses based on retrieved information)
- **Function**: Combines retrieved information with generative language models to formulate an answer.

#### How it works:
- **Retrieval**: Relevant document chunks are retrieved based on the user's query.
- **Generation**: The language model (Ollama) generates a coherent and accurate response by combining the retrieved document chunks with its own knowledge.


Add your content using ``reStructuredText`` syntax. See the
`reStructuredText <https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html>`_
documentation for details.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

