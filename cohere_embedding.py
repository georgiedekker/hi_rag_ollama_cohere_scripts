#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import logging
import numpy as np
import yaml
import asyncio
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

try:
    from dotenv import load_dotenv
    # Load environment variables from .env file
    load_dotenv()
except ImportError:
    # If dotenv is not installed, just continue
    pass

# from openai import AsyncOpenAI
from dataclasses import dataclass
import cohere

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load configuration from YAML file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Extract Cohere configurations
COHERE_API_KEY = os.environ.get("COHERE_API_KEY", config['cohere']['api_key'])
COHERE_EMBEDDING_MODEL = os.environ.get("COHERE_EMBEDDING_MODEL", config['cohere']['embedding_model'])
# COHERE_BASE_URL = os.environ.get("COHERE_BASE_URL", config['cohere']['base_url'])
COHERE_INPUT_TYPE = os.environ.get("COHERE_INPUT_TYPE", config['cohere']['input_type'])
# Make sure embedding dimension is an integer
COHERE_EMBEDDING_DIM = int(os.environ.get("COHERE_EMBEDDING_DIM", config['model_params']['cohere_embedding_dim']))
MAX_TOKEN_SIZE = int(os.environ.get("COHERE_MAX_TOKEN_SIZE", config['model_params']['max_token_size']))

logger.info(f"Using COHERE_API_KEY: {COHERE_API_KEY[:3]}...{COHERE_API_KEY[-3:]}")
logger.info(f"Using COHERE_EMBEDDING_DIM: {COHERE_EMBEDDING_DIM}")
logger.info(f"Using COHERE_EMBEDDING_MODEL: {COHERE_EMBEDDING_MODEL}")
logger.info(f"Using COHERE_INPUT_TYPE: {COHERE_INPUT_TYPE}")

@dataclass
class EmbeddingFunc:
    embedding_dim: int
    max_token_size: int
    func: callable

    async def __call__(self, *args, **kwargs) -> np.ndarray:
        return await self.func(*args, **kwargs)

def wrap_embedding_func_with_attrs(**kwargs):
    """Wrap a function with attributes"""
    def final_decro(func) -> EmbeddingFunc:
        new_func = EmbeddingFunc(**kwargs, func=func)
        return new_func
    return final_decro

@wrap_embedding_func_with_attrs(embedding_dim=COHERE_EMBEDDING_DIM, max_token_size=MAX_TOKEN_SIZE)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
)
async def cohere_embedding_direct(texts: list[str]) -> np.ndarray:
    """Use the Cohere embedding API directly with AsyncClient"""
    logger.info(f"Generating embeddings for {len(texts)} texts using Cohere AsyncClient")
    
    # Initialize Cohere AsyncClient
    co = cohere.AsyncClient(api_key=COHERE_API_KEY)
    
    # Process in smaller batches to avoid token limits
    batch_size = 10
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        
        try:
            response = await co.embed(
                texts=batch_texts,
                model=COHERE_EMBEDDING_MODEL,
                input_type=COHERE_INPUT_TYPE,
                embedding_types=["float"]
            )
            
            # Access embeddings via the correct object attribute structure
            # Response is an object, not a dictionary
            batch_embeddings = response.embeddings.float
            all_embeddings.extend(batch_embeddings)
            logger.info(f"Processed batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
            
        except Exception as e:
            logger.error(f"Error generating embeddings with Cohere AsyncClient: {e}")
            # Return zero embeddings as fallback
            for _ in batch_texts:
                all_embeddings.append([0] * COHERE_EMBEDDING_DIM)
    
    return np.array(all_embeddings)

@wrap_embedding_func_with_attrs(embedding_dim=COHERE_EMBEDDING_DIM, max_token_size=MAX_TOKEN_SIZE)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
)
async def cohere_embedding_compatible(texts: list[str]) -> np.ndarray:
    """Use the Cohere embedding API through OpenAI compatibility layer"""
    logger.info(f"Generating embeddings for {len(texts)} texts using Cohere Compatibility API")
    
    # # Initialize OpenAI client with Cohere compatibility API
    # client = AsyncOpenAI(
    #     api_key=COHERE_API_KEY,
    #     base_url="https://api.cohere.ai/v1"
    # )
    client = cohere.AsyncClient(api_key=COHERE_API_KEY)
    
    # Process in smaller batches to avoid token limits
    batch_size = 10
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        
        try:
            response = await client.embed(
                texts=batch_texts,
                model=COHERE_EMBEDDING_MODEL,
                input_type=COHERE_INPUT_TYPE,
                embedding_types=["float"]
            )
            
            # Access embeddings via the correct object attribute structure
            batch_embeddings = response.embeddings.float
            all_embeddings.extend(batch_embeddings)
            logger.info(f"Processed batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
            
        except Exception as e:
            logger.error(f"Error generating embeddings with Cohere compatibility API: {e}")
            # Return zero embeddings as fallback
            for _ in batch_texts:
                all_embeddings.append([0] * COHERE_EMBEDDING_DIM)
    
    return np.array(all_embeddings)

# Default to using the direct API with AsyncClient
cohere_embedding = cohere_embedding_direct

# Function to switch between implementations
def use_direct_api():
    global cohere_embedding
    cohere_embedding = cohere_embedding_direct
    logger.info("Switched to using Cohere API directly with AsyncClient")

def use_compatibility_api():
    global cohere_embedding
    cohere_embedding = cohere_embedding_compatible
    logger.info("Switched to using Cohere Compatibility API")

# Simple test function
async def test_embedding():
    texts = ["This is a test document", "Another test document for embedding"]
    result = await cohere_embedding(texts)
    logger.info(f"Test embedding shape: {result.shape}")
    return result

if __name__ == "__main__":
    logger.info("Running test embedding")
    asyncio.run(test_embedding()) 