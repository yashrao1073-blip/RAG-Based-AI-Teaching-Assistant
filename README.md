# RAG-Based AI Teaching Assistant

An AI-powered Teaching Assistant built using **Retrieval-Augmented Generation (RAG)** that answers user questions from educational video content. 
The system converts videos into a searchable knowledge base using semantic embeddings and generates context-aware responses with a Large Language Model.

## Features

- Convert educational videos into audio
- Transcribe audio using OpenAI Whisper
- Chunk transcripts into smaller semantic sections
- Generate embeddings using Ollama's **bge-m3** embedding model
- Store embeddings for efficient semantic retrieval
- Retrieve the most relevant transcript chunks based on user queries
- Generate context-aware answers using **Llama 3.2**

## Project Workflow

```
Educational Videos
        │
        ▼
Audio Extraction
        │
        ▼
OpenAI Whisper Transcription
        │
        ▼
Transcript Chunking
        │
        ▼
Embedding Generation (bge-m3)
        │
        ▼
Embedding Storage
        │
        ▼
User Question
        │
        ▼
Question Embedding
        │
        ▼
Semantic Similarity Search
        │
        ▼
Relevant Context Retrieval
        │
        ▼
Prompt Construction
        │
        ▼
Llama 3.2
        │
        ▼
Generated Answer
```

## Technologies Used

- Python
- OpenAI Whisper
- Ollama
- Llama 3.2
- bge-m3 Embedding Model
- Pandas
- Joblib
- NumPy
- Scikit-learn

## How It Works

1. Educational videos are converted into audio.
2. Audio is transcribed using OpenAI Whisper.
3. Transcripts are divided into smaller chunks.
4. Each chunk is converted into a vector embedding using **bge-m3**.
5. Embeddings are stored for semantic retrieval.
6. When a user asks a question:
   - The question is embedded.
   - The most relevant transcript chunks are retrieved.
   - A prompt is constructed using the retrieved context.
   - Llama 3.2 generates a context-aware answer.

## Future Improvements

- Replace DataFrame storage with a dedicated vector database (e.g., FAISS)
- Support PDF and document ingestion
- Multi-document semantic search
- Streaming responses
- Conversation history
- Source citations with confidence scores

