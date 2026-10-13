# RAG-Based AI Teaching Assistant

An AI-powered Teaching Assistant built using **Retrieval-Augmented Generation (RAG)** that answers questions from educational video content. The system converts videos into a searchable knowledge base using semantic embeddings and generates context-aware responses with a large language model.

## Features

- Convert educational videos into audio
- Transcribe audio using OpenAI Whisper
- Chunk transcripts into smaller semantic sections
- Generate embeddings using Ollama's **bge-m3** embedding model
- Store embeddings for efficient semantic retrieval
- Retrieve relevant transcript chunks based on user queries
- Generate answers using **Llama 3.2**
- Ask questions in a Streamlit chat UI and jump to relevant lesson timestamps

## Project workflow

```text
Educational videos → audio extraction → Whisper transcription → transcript chunks
→ bge-m3 embeddings → semantic retrieval → context prompt → Llama 3.2 answer
```

## Run the Streamlit app

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Install and start [Ollama](https://ollama.com/), then pull the models:

   ```bash
   ollama pull bge-m3
   ollama pull llama3.2
   ```

3. Run the app from the project folder:

   ```bash
   streamlit run app.py
   ```

The app expects `embeddings.joblib` in the project folder and local `.mp4` course videos in `videos/`. It uses transcript JSON files in `jsons/` to rebuild embeddings if the existing index was serialized with an incompatible pandas version. `process_incoming.py` remains available as the original command-line RAG flow.

## Technologies

Python, Streamlit, OpenAI Whisper, Ollama, Llama 3.2, bge-m3, Pandas, Joblib, NumPy, and scikit-learn.
