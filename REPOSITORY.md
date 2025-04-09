# HiRAG Pipeline Implementation

This repository contains an implementation of the Hierarchical Retrieval Augmented Generation (HiRAG) system, with pipelines for both Ollama and Cohere.

## Quick Start

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/hirag-pipeline.git
   cd hirag-pipeline
   ```

2. Set up your environment:
   ```bash
   # Copy the example env file and edit it with your API keys and settings
   cp .env.example .env
   nano .env  # or use any text editor
   ```

3. Run with Docker:
   ```bash
   # Start the containers
   docker-compose up -d
   
   # Run the Ollama pipeline
   docker exec -it hirag ./run_pipeline.sh -i /app/data/input -n /app/data/ner -c /app/data/chunks -q "Your query here"
   
   # Run the Cohere pipeline
   docker exec -it hirag ./run_cohere_pipeline.sh /app/data/input /app/data/ner /app/data/chunks
   ```

4. Or run locally:
   ```bash
   # Install dependencies
   pip install -r requirements.txt
   
   # Run the Ollama pipeline
   ./run_pipeline.sh -i input_dir -n ner_dir -c chunks_dir -q "Your query here"
   
   # Run the Cohere pipeline
   ./run_cohere_pipeline.sh input_dir ner_dir chunks_dir
   ```

## Features

- Hierarchical knowledge organization
- Multiple model provider support
- Text sanitization for robust JSON handling
- Docker integration with Ollama
- Neo4j integration (optional)

## Supported Model Providers

### Ollama

Uses open-source models like GLM4 and embedding models locally.

```bash
# In .env
PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OPENAI_MODEL_NAME=glm4
OLLAMA_EMBEDDING_MODEL=rjmalagon/gte-qwen2-7b-instruct:f16
```

### Cohere

Uses Cohere's API for entity extraction and embeddings.

```bash
# In .env
PROVIDER=cohere
COHERE_API_KEY=your-api-key
COHERE_EMBEDDING_MODEL=embed-english-v3.0
COHERE_CHAT_MODEL=command
```

### Other Providers

OpenAI, DeepSeek, and Azure OpenAI are also supported. See `.env.example` for configuration options.

## Docker Setup

The docker-compose.yml includes:

1. **HiRAG**: The main service running the pipelines
2. **Ollama**: For local model inference 
3. **Neo4j** (optional): For graph storage

```bash
# Start all services
docker-compose up -d

# Pull models in Ollama
docker exec -it ollama ollama pull glm4
docker exec -it ollama ollama pull rjmalagon/gte-qwen2-7b-instruct:f16

# Run a query
docker exec -it hirag ./run_pipeline.sh -q "What is HiRAG?"
```

## Pipeline Scripts

### Ollama Pipeline

```bash
./run_pipeline.sh --ingest-dir input_dir --ner-dir ner_dir --chunker-dir chunks_dir --query "What is HiRAG?"
```

### Cohere Pipeline

```bash
./run_cohere_pipeline.sh input_dir ner_dir chunks_dir
```

## Advanced Configuration

See `config.yaml` for configuration options, including:

- Vector database settings
- Clustering algorithms
- Embedding dimensions
- Token limits

## Troubleshooting

- **API Key Issues**: Ensure your API keys are correctly set in `.env`
- **Docker GPU Support**: For GPU support, ensure NVIDIA Container Toolkit is installed
- **JSON Parsing Errors**: The text sanitizer module should handle most issues with special characters

## Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

This project is licensed under the MIT License. 