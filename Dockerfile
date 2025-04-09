FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    jq \
    && rm -rf /var/lib/apt/lists/*

# Clone the forked HiRAG repository
RUN git clone https://github.com/georgiedekker/HiRAG.git

# Install Python dependencies
WORKDIR /app/HiRAG
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -e .

# Return to app directory
WORKDIR /app

# Copy application files
COPY *.py *.sh /app/
COPY config.yaml /app/
COPY text_sanitizer.py /app/
COPY mini_entity_extract.py /app/
COPY cohere_embedding.py /app/

# Make shell scripts executable
RUN chmod +x /app/*.sh

# Create directories for working files
RUN mkdir -p /app/data
RUN mkdir -p /app/output

# Set environment variables
ENV PYTHONPATH=/app:/app/HiRAG

# Create sample .env file if none exists (will be overridden by volume mount)
RUN echo "# Sample .env file - override with your own values\n\
PROVIDER=ollama\n\
OLLAMA_BASE_URL=http://ollama:11434\n\
COHERE_EMBEDDING_MODEL=embed-english-v3.0\n\
COHERE_EMBEDDING_DIM=1024\n\
" > /app/.env.example

# Default command - show available commands
CMD ["bash", "-c", "echo 'HiRAG container is ready! Available commands:'; echo '- ./run_pipeline.sh - Run HiRAG pipeline'; echo '- ./run_cohere_pipeline.sh - Run Cohere pipeline'; echo '- python hi_rag_demo.py - Run HiRAG demo'"] 