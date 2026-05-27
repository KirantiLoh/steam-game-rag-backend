# Steamrec - Backend

![Steamrec showcase](./docs/images/steamrec.gif)

A high-performance FastAPI backend powering an intelligent Steam game discovery and recommendation system. Built with modern retrieval-augmented generation (RAG) architecture, combining hybrid search, neural reranking, and conversational AI.

## Features

### 🔍 Advanced Search Capabilities

- **Hybrid Search Engine**: Combines BM25 text matching with CLIP vector similarity for optimal relevance
- **Image-Based Search**: Upload game screenshots or artwork to find visually similar games using CLIP embeddings
- **Neural Reranking**: Cross-encoder models (MS MARCO MiniLM) for precision reranking of search results
- **Smart Filtering**: Filter by genre, platform (Windows/Mac/Linux), and price range
- **Similarity Search**: Find games similar to any title in the catalog
- **Trending Games**: Discover popular titles based on review metrics

### 💬 Conversational AI Assistant

- **RAG-Powered Chat**: Context-aware game recommendations using Retrieval-Augmented Generation
- **Real-Time Streaming**: Server-Sent Events (SSE) for token-by-token response streaming
- **Multi-Turn Conversations**: Session management with conversation history (30-minute TTL)
- **Smart Citations**: Automatic game references with `[Game Name](game_id)` format
- **Multi-Modal Input**: Support for text + image queries in chat
- **Free Tier Integration**: Powered by Google Gemini 2.5 Flash

### ⚡ Performance & Scalability

- **Search Caching**: In-memory LRU cache for frequently requested queries
- **Batch Processing**: Optimized batch lookups to eliminate N+1 query problems
- **Async-First Design**: Full async/await support with FastAPI + asyncio
- **Rate Limiting**: Configurable middleware for API protection
- **GPU Acceleration**: CUDA support for CLIP and cross-encoder inference

### 🏗️ Architecture

- **Elasticsearch Backend**: Document storage and hybrid search (BM25 + kNN)
- **Sentence Transformers**: CLIP-ViT-B-32 for image/text embeddings
- **Cross-Encoder Reranking**: MS MARCO MiniLM-L-6-v2 for relevance scoring
- **Session Management**: In-memory session store with automatic TTL expiration
- **Modular Design**: Clear separation of concerns (search, chat, storage, API)

## Tech Stack

- **Framework**: FastAPI 0.115+
- **Search**: Elasticsearch 8.x
- **ML Models**: 
  - sentence-transformers (CLIP, cross-encoders)
  - PyTorch 2.0+
- **LLM Integration**: Google Gemini 2.5 Flash (google-genai SDK)
- **Data Processing**: pandas, numpy, pyarrow
- **Server**: Uvicorn (ASGI)

## Installation

### Prerequisites

- Python 3.10+
- Elasticsearch 8.x (running on localhost:9200 or configured via `ES_URL`)
- CUDA-compatible GPU (optional, for faster inference)

### Setup

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Configure environment variables**:
Create a `.env` file in the backend directory:
```env
# Elasticsearch
ES_URL=http://localhost:9200

# LLM Configuration (Gemini)
LLM_GEMINI_API_KEY=your_api_key_here  # Get from https://aistudio.google.com/app/apikey
LLM_MODEL_NAME=gemini-2.5-flash
LLM_MAX_TOKENS=8192
LLM_TEMPERATURE=0.7

# RAG Settings
LLM_RAG_TOP_K=25
LLM_RAG_RERANK=true

# Rate Limiting
LLM_MAX_REQUESTS_PER_MINUTE=15

# Session Management
LLM_SESSION_TTL_SECONDS=1800  # 30 minutes
LLM_MAX_CONVERSATION_TURNS=5
```

3. **Index Steam game data** (if not already done):
```bash
python scripts/elastic_index.py
```

4. **Start the server**:
```bash
python main.py
```

The API will be available at `http://localhost:8000`.

## API Endpoints

### Search Endpoints

#### `GET /api/search`
Hybrid search with text and vector similarity.

**Query Parameters**:
- `q` (string): Search query
- `limit` (int): Number of results (1-100, default: 20)
- `rerank` (bool): Enable cross-encoder reranking (default: true)
- `genre` (string): Filter by genre
- `platform` (string): Filter by platform (Windows, Mac, Linux)
- `price_min` (float): Minimum price
- `price_max` (float): Maximum price

**Example**:
```bash
curl "http://localhost:8000/api/search?q=sci-fi+roguelike&limit=10&rerank=true"
```

#### `POST /api/search/image`
Search games by uploading an image.

**Form Data**:
- `file` (file): Image file (JPEG, PNG)
- `limit` (int): Number of results (1-100, default: 20)

**Example**:
```bash
curl -X POST -F "file=@screenshot.jpg" -F "limit=10" http://localhost:8000/api/search/image
```

#### `GET /api/games/{game_id}`
Get detailed information about a specific game.

#### `GET /api/games/{game_id}/similar`
Find similar games based on vector similarity.

**Query Parameters**:
- `limit` (int): Number of results (1-50, default: 8)
- `rerank` (bool): Enable reranking (default: false)

#### `GET /api/trending`
Get trending games based on popularity metrics.

**Query Parameters**:
- `limit` (int): Number of results (1-50, default: 12)

### Chat Endpoints

#### `POST /api/chat`
Conversational game recommendation assistant with RAG.

**Form Data**:
- `query` (string, required): User message (1-500 chars)
- `session_id` (string, optional): Session ID for multi-turn conversations
- `stream` (bool): Enable SSE streaming (default: true)
- `image` (file, optional): Image file for multi-modal queries
- `genre` (string, optional): Filter recommendations by genre
- `platform` (string, optional): Filter by platform
- `price_min` (float, optional): Minimum price filter
- `price_max` (float, optional): Maximum price filter

**Streaming Response** (SSE):
```
event: session
data: <session_id>

event: metadata
data: <games_retrieved_count>

event: token
data: <token_text>

event: done
data: success
```

**Non-Streaming Response** (JSON):
```json
{
  "response": "Here are some great sci-fi roguelikes...",
  "session_id": "abc-123",
  "games_retrieved": 25
}
```

**Example**:
```bash
# Streaming mode
curl -X POST -F "query=I love challenging roguelikes" -F "stream=true" \
  http://localhost:8000/api/chat

# Non-streaming mode with filters
curl -X POST -F "query=Recommend multiplayer games" -F "stream=false" \
  -F "platform=Windows" -F "price_max=30" \
  http://localhost:8000/api/chat
```

#### `DELETE /api/chat/session/{session_id}`
Clear conversation history for a session.

#### `GET /api/chat/sessions`
Get count of active chat sessions.

#### `GET /api/chat/health`
Health check for chat service with configuration details.

### Health Endpoint

#### `GET /api/health`
System health check with Elasticsearch connection status.

## Configuration

### Rate Limiting

The API includes built-in rate limiting middleware to protect against abuse and stay within free tier limits (Gemini: 15 RPM).

Configured via `LLM_MAX_REQUESTS_PER_MINUTE` in `.env`.

### Search Caching

The search cache uses an LRU strategy with a default capacity of 1000 queries. Cache keys include all query parameters (query text, filters, rerank settings) for accurate cache hits.

### Session Management

Chat sessions automatically expire after the configured TTL (default: 30 minutes). The session manager maintains conversation history with a maximum number of turns (default: 5) to prevent context overflow.

### Model Configuration

All models are loaded on startup:
- **CLIP Model**: `clip-ViT-B-32` (image/text embeddings)
- **Reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **LLM**: Google Gemini 2.5 Flash (free tier)

GPU acceleration is automatically enabled when CUDA is available.

## Development

### Project Structure

```
backend/
├── api/                    # API layer
│   ├── main.py            # FastAPI app initialization
│   ├── models.py          # Pydantic models
│   └── routes/            # API endpoints
│       ├── search.py      # Search endpoints
│       ├── chat.py        # Chat endpoints
│       └── health.py      # Health check
├── search/                 # Search infrastructure
│   ├── es_retriever.py    # Elasticsearch hybrid search
│   ├── game_store.py      # Game metadata storage
│   ├── reranker.py        # Cross-encoder reranking
│   └── cache.py           # Search cache
├── chat/                   # Chat/RAG services
│   ├── llm_client.py      # LLM client (Gemini)
│   ├── rag_service.py     # RAG orchestration
│   ├── session_manager.py # Session management
│   └── rate_limiter.py    # Rate limiting middleware
├── core/                   # Core configuration
│   └── config.py          # Settings management
├── scripts/                # Utility scripts
│   ├── elastic_index.py   # Index games to Elasticsearch
│   ├── download_indices.py # Download pre-built indices
│   └── check_search.py    # Search testing
├── tests/                  # Test suite
├── legacy/                 # Legacy retrieval implementations
├── main.py                # Entry point
└── requirements.txt       # Python dependencies
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_metadata.py

# Manual chat testing
python tests/test_chat_manual.py
```

### Logging

All components use Python's standard logging with INFO level by default:
- Request/response logging
- Model loading status
- Cache hit/miss metrics
- Error stack traces

### Optimization Tips

1. **Enable Reranking Selectively**: For exploratory queries, skip reranking to save 100-200ms
2. **Batch Requests**: Frontend should implement request batching/debouncing
3. **Cache Warming**: Pre-populate cache with common queries
4. **GPU Acceleration**: Use CUDA-enabled GPU for 3-5x inference speedup

## API Documentation

Interactive API documentation is available at:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## License

This project is part of the Steam Game Recommender system.

## Support

For issues, questions, or contributions, please refer to the main project repository.

---

**Built with FastAPI, Elasticsearch, Sentence Transformers, and Google Gemini**
