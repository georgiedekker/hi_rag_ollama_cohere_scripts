# Publishing HiRAG to Git

This document outlines the steps taken to prepare the HiRAG implementation for publishing to a Git repository.

## Files Prepared

1. **Docker Support**
   - Updated `Dockerfile` to use the forked HiRAG repository: https://github.com/georgiedekker/HiRAG
   - Created `docker-compose.yml` with support for Ollama models and optional Neo4j
   - Configured container to run both HiRAG and Cohere pipelines

2. **Environment Variables**
   - Created `.env.example` as a template for environment variables
   - Added `.gitignore` to exclude `.env` files from the repository
   - Updated `config.yaml` to use environment variables with defaults

3. **Documentation**
   - Created `REPOSITORY.md` as the main README for the Git repository
   - Updated the local `README.md` with Git publishing instructions
   - Added clear instructions for all model providers (Ollama, Cohere, etc.)

4. **Security**
   - Created `check_sensitive_info.sh` to scan for potential API keys or credentials
   - Verified all code uses environment variables or placeholders
   - Added instructions for checking files before publishing

## Steps to Publish

1. **Run the security check**:
   ```bash
   ./check_sensitive_info.sh
   ```
   Review any findings and ensure no actual API keys are included.

2. **Test the Docker setup**:
   ```bash
   docker-compose build
   docker-compose up -d
   ```
   Verify that the container starts correctly and can access the forked HiRAG repository.

3. **Prepare the repository**:
   ```bash
   # Create the Git repository
   git init
   
   # Copy REPOSITORY.md to README.md for the Git repo
   cp REPOSITORY.md README.md
   
   # Add files
   git add .gitignore Dockerfile docker-compose.yml *.py *.sh config.yaml .env.example README.md
   
   # Initial commit
   git commit -m "Initial commit of HiRAG implementation with Ollama and Cohere support"
   ```

4. **Push to GitHub**:
   ```bash
   # Add your remote repository
   git remote add origin https://github.com/your-username/hirag-pipeline.git
   
   # Push to main branch
   git push -u origin main
   ```

## Verification After Publishing

1. Clone the repository to a fresh directory
2. Copy `.env.example` to `.env` and add your API keys
3. Run with Docker:
   ```bash
   docker-compose up -d
   ```
4. Or run locally:
   ```bash
   pip install -r requirements.txt
   ./run_pipeline.sh -i input_dir -n ner_dir -c chunks_dir -q "Test query"
   ```

## Files Excluded from Git

- `.env` - Contains actual API keys
- `data/` - Contains data files and may include sensitive information
- `output/` - Contains processing results
- Vector database files - Large binary files not needed for distribution
- Cached responses - May contain sensitive information

## Maintenance

1. Regularly run the sensitive information check when adding new code
2. Keep the `.env.example` file updated with new environment variables
3. Document any changes to configuration or dependencies in README.md 