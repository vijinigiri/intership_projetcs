# Shazam_Clone
AI-powered semantic search for video subtitles.

# Enhancing Search Engine Relevance for Video Subtitles

## Background
In the fast-evolving landscape of digital content, effective search engines play a pivotal role in connecting users with relevant information. This project focuses on improving search relevance for video subtitles, enhancing the accessibility of video content through a semantic search engine that supports both text and audio queries.

## Objective
Develop an advanced subtitle search engine that efficiently retrieves subtitles based on user queries. The primary goal is to leverage Natural Language Processing (NLP), machine learning, and GPU-accelerated indexing to enhance the relevance and accuracy of search results.

## Keyword-based vs. Semantic Search
### Keyword-Based Search Engine
- Relies on exact keyword matches between user queries and indexed subtitles.
- Efficient for simple queries but lacks contextual understanding.

### Semantic Search Engine
- Goes beyond keyword matching to understand the meaning and context of queries and subtitle content.
- Uses embeddings and similarity metrics for more accurate search results.

### Comparison
While keyword-based search engines focus on word matches, semantic search engines aim to understand deeper meanings and context, leading to more intelligent retrieval.

## Core Logic
To compare a user query against video subtitles, the core logic involves three key steps:

### 1. Data Preprocessing
- Sample a subset of the data if computational resources are limited.
- Clean subtitles by removing timestamps and metadata.
- Convert cleaned subtitle text into numerical representations using embeddings.
- Encode user queries into similar vector representations.

### 2. Cosine Similarity Calculation
- Compute cosine similarity between subtitle document vectors and user query vectors.
- Determine relevance based on similarity scores.
- Retrieve and return the top-matching subtitle documents.

## Data
- The dataset consists of video subtitles stored in a `.db` file format.
- Extract and preprocess subtitle data.
- Store subtitle embeddings in a ChromaDB database for efficient retrieval.

## Step-by-Step Process
### Part 1: Ingesting and Indexing Subtitles
1. Load and analyze subtitle data.
2. Extract and clean subtitle text.
3. Perform text vectorization using:
   - **TF-IDF** (for keyword-based search).
   - **BERT-based SentenceTransformers** (for semantic search).
4. **Document Chunking:**
   - Break large subtitle documents into smaller chunks.
   - Use overlapping windows to preserve context.
5. Store embeddings efficiently in ChromaDB.

### Part 2: Query Processing and Retrieval
1. Accept user search queries in **text** or **audio** format.
2. Convert audio queries to text using **Whisper**.
3. Preprocess the transcribed text.
4. Generate query embeddings.
5. Compute similarity scores with stored subtitle embeddings.
6. Retrieve and display the most relevant subtitle segments.

## Optimization Strategies
- **GPU-Accelerated BERT Embeddings**: Optimize embedding generation with fp16 precision.
- **Efficient ChromaDB Storage**: Implement batch insertion and ensure GPU acceleration.
- **Fast FAISS Indexing**: Enable quick nearest-neighbor retrieval for embeddings.
- **Memory-Efficient Processing**: Implement chunking and batching to handle large datasets.

## Expected Outcomes
By implementing this approach, the project aims to:
- Improve the accuracy and relevance of subtitle search results.
- Enable seamless retrieval of subtitles using text and audio queries.
- Optimize processing speed using GPU acceleration.

This project enhances video content accessibility, making subtitle search more efficient and user-friendly.



