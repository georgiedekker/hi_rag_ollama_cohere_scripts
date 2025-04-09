# HI_RAG Implementation

This is an implementation of the Hierarchical Retrieval Augmented Generation (HiRAG) system based on the [HiRAG repository](https://github.com/hhy-huang/HiRAG) by hhy-huang, with enhancements by [georgiedekker](https://github.com/georgiedekker/HiRAG).

## Overview

HI_RAG is a new approach to create a retrieval augmented generation function. It uses hierarchy in the construction of a multi-layer knowledge graph to improve the quality of generated responses.

Key features:
- Hierarchical knowledge organization with global, bridge, and local knowledge layers
- Multi-layer graph construction for better representation of knowledge relationships
- Dynamic retrieval process that integrates information across layers
- Significantly better results compared to traditional RAG approaches
- Multi-provider support (OpenAI, Ollama, DeepSeek, Azure OpenAI, and Cohere)
- Robust text sanitization for handling special characters and JSON parsing challenges

## Publishing to Git

If you're planning to publish this code to a Git repository, please follow these steps:

1. Check for sensitive information first:
   ```bash
   ./check_sensitive_info.sh
   ```

2. Ensure your `.env` file is not included in Git:
   ```bash
   # Verify .gitignore includes .env
   cat .gitignore | grep .env
   ```

3. Use `REPOSITORY.md` as your README.md in the Git repository:
   ```bash
   cp REPOSITORY.md README.md
   ```

See [REPOSITORY.md](REPOSITORY.md) for detailed information about the repository structure.

## Setup Instructions

### Prerequisites

- Python 3.8+ with pip
- Docker and Docker Compose (for containerized usage)
- Ollama (running locally or accessible via API) for open source models
- Neo4j (optional, for graph database storage)
- API keys for Cohere, OpenAI, DeepSeek, or Azure OpenAI (if using those providers)

### Installation

#### Option 1: Local Installation

1. Run the setup verification script which will check and install necessary components:

```bash
cd hi_rag
python verify_setup.py
```

This script will:
- Install required packages from requirements.txt
- Check if HiRAG is installed, and install it if not
- Create the data directory if it doesn't exist

2. After successful verification, you can use the hi_rag_demo.py script.

#### Option 2: Docker Installation

1. Clone this repository
2. Make sure Ollama is installed and running
3. Use Docker Compose to build and run the container:

```bash
cd hi_rag
docker-compose up -d
```

### Neo4j Setup (Optional)

To use the Neo4j integration:

1. Install Neo4j (Community or Enterprise edition)
2. Start the Neo4j service
3. Create a database and set a username and password
4. When running the pipeline, use the `--use-neo4j` flag along with connection details

## Configuration

The implementation supports multiple model providers:

### Ollama Models Configuration

Ollama provides a way to run open-source models locally. The system is configured to work with:

1. **GLM4** - A powerful open-source model from Tsinghua University
2. **rjmalagon/gte-qwen2-7b-instruct:f16** - A fine-tuned embedding model (3584 dimensions)

To use these models:

1. **Install and Start Ollama**:
   ```bash
   # Install Ollama (if not already installed)
   curl -fsSL https://ollama.com/install.sh | sh
   
   # Start the Ollama service
   ollama serve
   ```

2. **Pull the Required Models**:
   ```bash
   # Pull the GLM4 model
   ollama pull glm4
   
   # Pull the embedding model
   ollama pull rjmalagon/gte-qwen2-7b-instruct:f16
   ```

3. **Configure in .env file**:
   ```bash
   # Set Ollama as provider
   PROVIDER=ollama
   
   # Configure Ollama endpoint (default is localhost)
   OLLAMA_BASE_URL=http://localhost:11434
   
   # Set default LLM model
   OPENAI_MODEL_NAME=glm4
   
   # Set embedding model (for vector embeddings)
   OLLAMA_EMBEDDING_MODEL=rjmalagon/gte-qwen2-7b-instruct:f16
   ```

### Cohere Configuration

This implementation includes integration with Cohere's API for entity extraction:

1. **Set up Cohere API Key**:
   ```bash
   # In your .env file
   COHERE_API_KEY=your_api_key
   COHERE_CHAT_MODEL=command
   COHERE_EMBEDDING_MODEL=embed-english-v3.0
   COHERE_EMBEDDING_DIM=1024
   ```

2. **Run the Cohere pipeline**:
   ```bash
   ./run_cohere_pipeline.sh ingest_dir ner_dir chunker_dir
   ```

### Other Providers Configuration

You can also configure:
- DeepSeek API for "best" model functions
- OpenAI API for GPT models and embeddings
- Azure OpenAI for hosted OpenAI models

## Usage

### Basic Usage

The implementation provides two Python scripts for working with HiRAG:

#### 1. run_hirag.py (Recommended)

This script automatically handles both indexing and querying in one step:

```bash
# Basic usage with default sample document
python run_hirag.py --query "What are the key features of HiRAG?"

# Specify a different document
python run_hirag.py --query "What is HiRAG?" --document path/to/your/document.txt

# Force reindexing even if vector store exists
python run_hirag.py --query "What is HiRAG?" --force-reindex

# Clean vector database (useful for fixing dimension mismatches)
python run_hirag.py --query "What is HiRAG?" --clean

# Change the query mode
python run_hirag.py --query "What is HiRAG?" --mode naive
```

#### 2. hi_rag_demo.py

For more control over the indexing and querying steps:

```bash
# Index a document
python hi_rag_demo.py --index sample_document.txt

# Run a query using the hierarchical mode
python hi_rag_demo.py --query "What are the key features of HiRAG?" --mode hi

# Interactive mode
python hi_rag_demo.py
```

### Using the Convenience Shell Script

A shell script `run.sh` is provided for easier usage:

```bash
# Setup the environment
./run.sh --setup

# Run a query
./run.sh -q "What is HiRAG?"

# Run with different modes and options
./run.sh -q "What is HiRAG?" -m naive     # Use naive RAG mode
./run.sh -q "What is HiRAG?" -f           # Force reindexing
./run.sh -q "What is HiRAG?" -c           # Clean vector database
./run.sh -q "What is HiRAG?" -d my_doc.txt  # Use a different document

# Show help
./run.sh -h
```

### Pipeline Integration

A pipeline integration script `pipeline_integration.py` and a convenience shell script `run_pipeline.sh` are provided to integrate HI_RAG with the existing pipeline components (ingest, graph_ner, and rag_chunker).

#### Using the Pipeline Integration Shell Script

```bash
# Show help with all available options
./run_pipeline.sh -h

# Basic usage (indexing only)
./run_pipeline.sh -i ../ingest/outputs -n ../graph_ner/output -c ../rag_chunker/output

# Index and run a query
./run_pipeline.sh -i ../ingest/outputs -n ../graph_ner/output -c ../rag_chunker/output -q "What is the main topic?"

# Using Neo4j integration
./run_pipeline.sh -i ../ingest/outputs -n ../graph_ner/output -c ../rag_chunker/output --use-neo4j

# Full Neo4j configuration
./run_pipeline.sh -i ../ingest/outputs -n ../graph_ner/output -c ../rag_chunker/output \
    --use-neo4j --neo4j-url "neo4j://localhost:7687" --neo4j-user "neo4j" --neo4j-pass "password"

# Advanced chunking configuration
./run_pipeline.sh -i ../ingest/outputs -n ../graph_ner/output -c ../rag_chunker/output \
    --chunk-size 1500 --chunk-overlap 200

# Using HNSWLib for vector database
./run_pipeline.sh -i ../ingest/outputs -n ../graph_ner/output -c ../rag_chunker/output --use-hnswlib

# Complete configuration with all features
./run_pipeline.sh -i ../ingest/outputs -n ../graph_ner/output -c ../rag_chunker/output \
    -q "What is the main topic?" -m hi --use-neo4j --neo4j-url neo4j://localhost:7687 \
    --chunk-size 1500 --chunk-overlap 200 --max-cluster-size 15 --use-hnswlib \
    --embedding-batch 64 --embedding-async 16 --naive-rag
```

### Cohere Pipeline Integration

To use the Cohere API for entity extraction and text processing:

```bash
# Run the Cohere pipeline with your data directories
./run_cohere_pipeline.sh ingest_dir ner_dir chunker_dir
```

The script includes robust text sanitization to ensure all chunks are properly processed, handling special characters, JSON delimiters, and other potential issues.

### Working with Ollama Models

#### Using GLM4 for Text Generation

GLM4 is a powerful open source model that provides high-quality generation capabilities:

```bash
# First ensure GLM4 is pulled into Ollama
ollama pull glm4

# Configure environment variables
export PROVIDER=ollama
export OPENAI_MODEL_NAME=glm4
export OLLAMA_BASE_URL=http://localhost:11434

# Run HiRAG with GLM4
python run_hirag.py --query "What are the key concepts in this document?"
```

#### Using rjmalagon/gte-qwen2-7b-instruct:f16 for Embeddings

This model provides high-quality 3584-dimensional embeddings:

```bash
# Pull the embedding model
ollama pull rjmalagon/gte-qwen2-7b-instruct:f16

# Configure environment variables
export OLLAMA_EMBEDDING_MODEL=rjmalagon/gte-qwen2-7b-instruct:f16

# When running HiRAG, it will automatically use this model for embeddings
python run_hirag.py --query "What are the main themes?"
```

#### Advanced Ollama Configuration

You can fine-tune Ollama's behavior in your `.env` file:

```
# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_TIMEOUT=60  # Seconds before timeout
OLLAMA_EMBEDDING_MODEL=rjmalagon/gte-qwen2-7b-instruct:f16
OLLAMA_EMBEDDING_DIM=3584
OLLAMA_CONCURRENCY=4  # Maximum concurrent requests
```

## Advanced Features

#### Vector Database Options

HiRAG supports two vector database backends:

1. **NanoVectorDB** (default): Simpler and lightweight
2. **HNSWLib**: More optimized for larger datasets

Use the `--use-hnswlib` flag to switch to HNSWLib.

#### Graph Storage Options

HiRAG supports two graph storage backends:

1. **NetworkX** (default): Stores graph data in local files
2. **Neo4j**: Stores graph data in a Neo4j database

Use the `--use-neo4j` flag to switch to Neo4j storage.

#### Chunking Options

You can customize how documents are chunked:

- `--chunk-size`: Size of each chunk in tokens (default: 1200)
- `--chunk-overlap`: Overlap between consecutive chunks in tokens (default: 100)

#### Embedding Options

You can customize embedding generation:

- `--embedding-batch`: Number of texts to embed in a single batch (default: 32)
- `--embedding-async`: Maximum concurrent embedding function calls (default: 8)

#### Clustering Options

You can customize graph clustering:

- `--max-cluster-size`: Maximum number of clusters to create (default: 10)

#### RAG Modes

You can choose between different RAG modes:

- `--naive-rag`: Enable naive RAG mode (no hierarchical features)
- `--no-hierarchical`: Disable hierarchical mode

### Text Sanitization Features

The implementation includes a robust text sanitization module to handle special characters and JSON parsing challenges:

1. **Character Escaping**: Automatically escapes backslashes, quotes, newlines, and other special characters
2. **JSON Safety**: Ensures all text is safe for inclusion in JSON structures
3. **Error Recovery**: Handles common JSON parsing errors like missing commas
4. **Recursive Sanitization**: Sanitizes all text fields in nested data structures

This is particularly important when working with the Cohere API, which may encounter issues with malformed JSON.

### Docker Usage

If you're using the Docker setup, run commands inside the container:

```bash
# Copy your document to the data directory first
cp sample_document.txt data/

# Run inside the container
docker exec -it hirag_hirag_1 python run_hirag.py --query "What is HiRAG?"
```

### Query Modes

The system supports several query modes:
- `hi`: Full hierarchical retrieval (default)
- `naive`: Traditional RAG approach
- `hi_nobridge`: Hierarchical retrieval without the bridge layer
- `hi_local`: Using only local knowledge
- `hi_global`: Using only global knowledge
- `hi_bridge`: Using only bridge knowledge

## Files Included

- `Dockerfile`: Container definition for running HiRAG
- `docker-compose.yml`: Orchestration for HiRAG and Ollama services
- `config.yaml`: Configuration for the various models and parameters
- `.env.example`: Example environment variables file
- `hi_rag_demo.py`: Main implementation file demonstrating HiRAG usage
- `run_hirag.py`: Combined script for indexing and querying in one step
- `run.sh`: Convenient shell script for common operations
- `test_hirag.py`: Unit tests for the HiRAG implementation
- `sample_document.txt`: Example document for indexing and querying
- `verify_setup.py`: Script to verify and set up the environment
- `pipeline_integration.py`: Script to integrate HI_RAG with the existing pipeline
- `run_pipeline.sh`: Convenient shell script for pipeline integration
- `run_cohere_pipeline.sh`: Script for running the Cohere entity extraction pipeline
- `text_sanitizer.py`: Module for ensuring text is properly escaped and safe for JSON
- `mini_entity_extract.py`: Extracts entities using Cohere API
- `test_sanitizer.py`: Tests for the text sanitization functionality
- `test_pipeline.py`: Test script for the pipeline integration
- `check_sensitive_info.sh`: Script to check for sensitive information before Git publishing
- `REPOSITORY.md`: Documentation specifically for the Git repository

## Implementation Details

The implementation leverages the original HiRAG codebase with custom configurations:

1. **Model Providers**:
   - **Ollama**: Local models like GLM4 and rjmalagon/gte-qwen2-7b-instruct:f16
   - **Cohere**: Entity extraction and embeddings
   - **DeepSeek**: Chat and advanced LLM operations
   - **OpenAI/Azure**: Optional providers for GPT models

2. **Features**:
   - Hierarchical knowledge organization
   - Entity-based retrieval
   - Text sanitization and error handling
   - Multiple storage options (NanoVectorDB, HNSWLib, Neo4j)
   - Configurable chunking, embedding, and clustering

3. **Pipeline Integration**:
   - Seamless connection with ingest, NER, and chunker components
   - Comprehensive output consolidation
   - Multi-provider workflow support

## Troubleshooting

### Read-only filesystem error
If you encounter an error like `OSError: [Errno 30] Read-only file system: '/app'`, it means you're trying to use Docker paths in your local environment. The script has been updated to automatically detect and use local paths when needed.

### Module not found error
If you see `ModuleNotFoundError: No module named 'hirag'`, run the verification script:

```bash
python verify_setup.py
```

This will install the HiRAG package and its dependencies.

### Dimension mismatch error
If you see an error like `AssertionError: Embedding dim mismatch, expected: 3584, but loaded: 1536`, it means there's a mismatch between the configured embedding dimensions and the existing vector database. To fix this, use the `--clean` option:

```bash
# Using run_hirag.py directly
python run_hirag.py --query "What is HiRAG?" --clean

# Using the shell script
./run.sh -q "What is HiRAG?" -c

# Using the pipeline integration
./run_pipeline.sh -i ../ingest/outputs -n ../graph_ner/output -c ../rag_chunker/output --clean
```

This will delete the existing vector database files and create a new one with the correct dimensions.

### JSON parsing errors with Cohere
If you encounter JSON parsing errors with the Cohere API, the system now includes robust text sanitization:

1. The text sanitizer automatically escapes special characters
2. The JSON parser has recovery mechanisms for common errors
3. Automatic retry logic is implemented for problematic chunks

### Neo4j connection errors
If you encounter errors connecting to Neo4j, check:
1. Neo4j service is running
2. Credentials are correct
3. Connection URL is correct (`neo4j://localhost:7687` is the default)
4. Neo4j APOC plugin is installed (required for some graph algorithms)

### Empty vector store error
If queries fail with various errors even after fixing other issues, make sure a document has been indexed first. The `run_hirag.py` script will automatically index a document if needed.

### Ollama model errors
If you encounter errors with Ollama models:

1. **Check if the model is pulled**:
   ```bash
   ollama list
   ```

2. **Verify Ollama is running**:
   ```bash
   curl http://localhost:11434/api/tags
   ```

3. **Check model dimensions**:
   For embedding models, ensure the `OLLAMA_EMBEDDING_DIM` matches the model's dimensions (e.g., 3584 for rjmalagon/gte-qwen2-7b-instruct:f16)

## Testing

Run the included tests to verify the implementation:

```bash
# Test HiRAG core functionality
python test_hirag.py

# Test pipeline integration
python test_pipeline.py

# Test text sanitization
python test_sanitizer.py
```

## Further Resources

- [HiRAG GitHub Repository](https://github.com/hhy-huang/HiRAG)
- [HiRAG Research Paper](https://arxiv.org/abs/2503.10150)
- [Neo4j Documentation](https://neo4j.com/docs/)
- [HNSWLib Documentation](https://github.com/nmslib/hnswlib)
- [Ollama Documentation](https://ollama.ai/docs)
- [Cohere API Reference](https://docs.cohere.com/reference/about)
- [DeepSeek API Documentation](https://platform.deepseek.com/docs) 