# medibot

-Developed an intelligent medical chatbot (“MediBot”) trained on The Gale Encyclopedia of Medicine (2nd Edition) to provide context-based medical information retrieval.
-Implemented document preprocessing and chunking pipeline using PyPDFLoader and RecursiveCharacterTextSplitter for efficient text segmentation.
-Created vector embeddings with sentence-transformers/all-MiniLM-L6-v2 and stored them in a FAISS vector database for fast semantic search.
-Integrated Zephyr-7B LLM (HuggingFaceH4/zephyr-7b-beta) for natural language generation via LangChain’s HuggingFace Pipeline.
-Designed a Streamlit-based user interface for interactive Q&A, enabling users to query medical topics with precise, context-driven answers.
-Ensured factual accuracy through custom prompt engineering, instructing the model to rely solely on provided context and avoid hallucination.
