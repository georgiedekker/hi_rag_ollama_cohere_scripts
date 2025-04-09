#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import logging
import numpy as np
import yaml
import asyncio
from hirag import HiRAG, QueryParam
from openai import AsyncOpenAI, OpenAI
from dataclasses import dataclass
from hirag.base import BaseKVStorage
from hirag._utils import compute_args_hash
from pathlib import Path
from tqdm import tqdm
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load configuration from YAML file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Extract configurations
GLM_API_KEY = config['glm']['api_key']
GLM_MODEL = config['glm']['model']
GLM_URL = config['glm']['base_url']
DEEPSEEK_API_KEY = config['deepseek']['api_key']
DEEPSEEK_MODEL = config['deepseek']['model']
DEEPSEEK_URL = config['deepseek']['base_url']
EMBEDDING_MODEL = config['glm']['embedding_model']

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

@wrap_embedding_func_with_attrs(
    embedding_dim=config['model_params']['glm_embedding_dim'], 
    max_token_size=config['model_params']['max_token_size']
)
async def embedding_function(texts: list[str]) -> np.ndarray:
    """Use the Ollama embedding model specified in config"""
    logger.info(f"Generating embeddings for {len(texts)} texts using {EMBEDDING_MODEL}")
    
    client = OpenAI(
        api_key=GLM_API_KEY,
        base_url=GLM_URL
    )
    
    embeddings = []
    # Process in smaller batches to avoid token limits
    batch_size = 10
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        try:
            response = client.embeddings.create(
                input=batch_texts,
                model=EMBEDDING_MODEL,
            )
            batch_embeddings = [d.embedding for d in response.data]
            embeddings.extend(batch_embeddings)
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            # Return zero embeddings as fallback
            for _ in batch_texts:
                embeddings.append([0] * config['model_params']['glm_embedding_dim'])
    
    return np.array(embeddings)

async def glm_llm_function(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    """Function to call the GLM model via Ollama"""
    logger.info(f"Calling GLM model with prompt: {prompt[:50]}...")
    
    openai_client = OpenAI(
        api_key=GLM_API_KEY,
        base_url=GLM_URL
    )
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Get the cached response if having
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    
    if hashing_kv is not None:
        args_hash = compute_args_hash(GLM_MODEL, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]

    try:
        response = openai_client.chat.completions.create(
            model=GLM_MODEL, 
            messages=messages,
            **kwargs
        )
        result = response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error calling GLM model: {e}")
        result = "Sorry, I encountered an error processing your request."
    
    # Cache the response if having
    if hashing_kv is not None:
        await hashing_kv.upsert(
            {args_hash: {"return": result, "model": GLM_MODEL}}
        )
    
    return result

async def deepseek_llm_function(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    """Function to call the Deepseek API"""
    logger.info(f"Calling Deepseek model with prompt: {prompt[:50]}...")
    
    openai_async_client = AsyncOpenAI(
        api_key=DEEPSEEK_API_KEY, 
        base_url=DEEPSEEK_URL
    )
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Get the cached response if having
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    
    if hashing_kv is not None:
        args_hash = compute_args_hash(DEEPSEEK_MODEL, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]

    try:
        response = await openai_async_client.chat.completions.create(
            model=DEEPSEEK_MODEL, 
            messages=messages,
            **kwargs
        )
        result = response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error calling Deepseek model: {e}")
        result = "Sorry, I encountered an error processing your request."
    
    # Cache the response if having
    if hashing_kv is not None:
        await hashing_kv.upsert(
            {args_hash: {"return": result, "model": DEEPSEEK_MODEL}}
        )
    
    return result

def initialize_hirag():
    """Initialize the HiRAG system with our configuration"""
    logger.info("Initializing HiRAG system...")
    
    # Create the working directory if it doesn't exist
    # Use relative path instead of absolute Docker path
    working_dir = config['hirag']['working_dir']
    
    # If the working directory is absolute and starts with /app, use a relative path instead
    if working_dir.startswith('/app'):
        working_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
        logger.info(f"Using local working directory: {working_dir}")
    
    os.makedirs(working_dir, exist_ok=True)
    
    # Initialize HiRAG
    graph_func = HiRAG(
        working_dir=working_dir,
        enable_llm_cache=config['hirag']['enable_llm_cache'],
        embedding_func=embedding_function,
        best_model_func=deepseek_llm_function,  # Using Deepseek for best model
        cheap_model_func=glm_llm_function,      # Using GLM for cheaper operations
        enable_hierachical_mode=config['hirag']['enable_hierachical_mode'], 
        embedding_batch_num=config['hirag']['embedding_batch_num'],
        embedding_func_max_async=config['hirag']['embedding_func_max_async'],
        enable_naive_rag=config['hirag']['enable_naive_rag']
    )
    
    return graph_func

def index_document(graph_func, document_path):
    """Index a document in the HiRAG system"""
    logger.info(f"Indexing document: {document_path}")
    
    document_path = Path(document_path)
    if not document_path.exists():
        logger.error(f"Document does not exist: {document_path}")
        return False
    
    try:
        with open(document_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        graph_func.insert(content)
        logger.info(f"Successfully indexed document: {document_path}")
        return True
    except Exception as e:
        logger.error(f"Error indexing document: {e}")
        return False

def process_query(graph_func, query, mode="hi"):
    """Process a query using HiRAG"""
    logger.info(f"Processing query: {query} (mode: {mode})")
    
    result = graph_func.query(query, param=QueryParam(mode=mode))
    return result

def main():
    """Main function to run HiRAG"""
    parser = argparse.ArgumentParser(description='HiRAG Demo')
    parser.add_argument('--index', type=str, help='Path to the document to index')
    parser.add_argument('--query', type=str, help='Query to process')
    parser.add_argument('--mode', type=str, default='hi', 
                        choices=['hi', 'naive', 'hi_nobridge', 'hi_local', 'hi_global', 'hi_bridge'],
                        help='Query mode')
    
    args = parser.parse_args()
    
    # Initialize HiRAG
    graph_func = initialize_hirag()
    
    # Index document if provided
    if args.index:
        success = index_document(graph_func, args.index)
        if not success:
            return
    
    # Process query if provided
    if args.query:
        result = process_query(graph_func, args.query, args.mode)
        print("\nQuery Result:")
        print("-" * 80)
        print(result)
        print("-" * 80)
    else:
        # Interactive mode
        print("\nHiRAG Interactive Mode")
        print("Type 'exit' to quit")
        print("-" * 80)
        
        while True:
            query = input("\nEnter your query: ")
            if query.lower() == 'exit':
                break
            
            result = process_query(graph_func, query, args.mode)
            print("\nResult:")
            print("-" * 80)
            print(result)
            print("-" * 80)

if __name__ == "__main__":
    main() 