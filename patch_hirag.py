#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import importlib.util
import logging
import numpy as np
import asyncio
import json
from dotenv import load_dotenv
import yaml
import cohere

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def patch_hirag():
    """Patch the HiRAG library to include support for Cohere embeddings"""
    logger.info("Patching HiRAG library to add Cohere embedding support")
    
    # Import hirag modules
    try:
        from hirag import _llm
        from hirag._utils import wrap_embedding_func_with_attrs
        from hirag import hirag
        import cohere
        from openai import AsyncOpenAI
        from tenacity import (
            retry,
            stop_after_attempt,
            wait_exponential,
            retry_if_exception_type,
        )
        
        # Import our cohere embedding implementation
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from cohere_embedding import cohere_embedding_direct
    except ImportError as e:
        logger.error(f"Failed to import required modules: {e}")
        logger.error("Make sure hirag and cohere packages are installed")
        return False
    
    # We'll use the cohere_embedding_direct function from our module
    # instead of defining it here
    
    # Define a new cohere completion function
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    async def cohere_completion(prompt: str, **kwargs) -> str:
        """Use Cohere's chat completion API directly"""
        logger.info(f"Using Cohere chat completion API directly")
        
        # Get API key from environment
        api_key = os.environ.get("COHERE_API_KEY", "")
        if not api_key:
            logger.error("COHERE_API_KEY environment variable not set")
            return "Error: COHERE_API_KEY not set"
        
        # Use the AsyncClient for better performance
        co = cohere.AsyncClient(api_key=api_key)
        
        try:
            # Default model
            model = os.environ.get("COHERE_CHAT_MODEL", "command")
            max_tokens = kwargs.get("max_tokens", 500)
            temperature = kwargs.get("temperature", 0.7)
            
            # Call the API with await
            response = await co.chat(
                message=prompt,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            return response.text
            
        except Exception as e:
            logger.error(f"Error with Cohere chat completion: {e}")
            return f"Error with Cohere API: {str(e)}"
    
    # Replace the get_openai_async_client_instance function completely
    def new_get_openai_async_client_instance():
        """Use our patched version that never uses Ollama when PROVIDER=cohere"""
        provider = os.environ.get("PROVIDER", "").lower()
        
        if provider == "cohere":
            logger.info("Overriding OpenAI client to use Cohere API directly")
            # This will be caught by our patched functions and not actually used
            return AsyncClient(
                api_key=os.environ.get("COHERE_API_KEY", "fake-key"),
                base_url="https://api.cohere.ai/v1"
            )
        
        # Standard OpenAI client
        base_url = os.environ.get("OPENAI_API_BASE", os.environ.get("OPENAI_BASE_URL", None))
        api_key = os.environ.get("OPENAI_API_KEY", 'ollama')  # Default to ollama if not set
        
        # Create the client with the environment variables if they exist
        if base_url:
            return AsyncOpenAI(base_url=base_url, api_key=api_key)
        else:
            return AsyncOpenAI()
    
    # Create a completely new openai_complete_if_cache function
    async def new_openai_complete_if_cache(
        prompt, model, llm_response_cache, temperature=0.7, max_tokens=700,
        top_p=0.95, frequency_penalty=0, presence_penalty=0, **kwargs
    ):
        """Completely overridden version that uses Cohere for the cohere provider"""
        provider = os.environ.get("PROVIDER", "").lower()
        
        if provider == "cohere":
            logger.info(f"Using Cohere chat completion instead of OpenAI")
            return await cohere_completion(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
        
        # Use the original function for non-Cohere providers
        openai_async_client = _llm.get_openai_async_client_instance()
        
        # Try to get from cache first
        if llm_response_cache is not None:
            key = (prompt, model, temperature, max_tokens, top_p, frequency_penalty, presence_penalty)
            key_hash = await hirag._utils.compute_args_hash(key)
            # Try to get from cache
            from_cache = await llm_response_cache.get(key_hash)
            if from_cache is not None:
                logger.debug("LLM request from cache")
                return from_cache
        
        # Not in cache, must query OpenAI
        try:
            openai_kwargs = {k: v for k, v in kwargs.items() if k not in ["prompt", "model", "temperature", "max_tokens", "top_p", "frequency_penalty", "presence_penalty"]}
            
            response = await openai_async_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                **openai_kwargs
            )
        
            result = response.choices[0].message.content.strip()
            
            # Cache the result
            if llm_response_cache is not None:
                await llm_response_cache.set(key_hash, result)
            
            return result
        except Exception as e:
            logger.error(f"Error in openai_complete_if_cache: {e}")
            return f"Error with API: {str(e)}"
    
    # Replace the original function with our patched versions
    _llm.get_openai_async_client_instance = new_get_openai_async_client_instance
    _llm.openai_complete_if_cache = new_openai_complete_if_cache
    
    # Patch the module to include cohere embedding - use our improved version
    _llm.cohere_embedding = cohere_embedding_direct
    
    # Create a new determine_default_provider function that includes cohere
    def new_determine_default_provider():
        """New version that checks for Cohere API key first"""
        if 'COHERE_API_KEY' in os.environ and os.environ.get('COHERE_API_KEY'):
            return "cohere"
        # Original logic for other providers
        if 'GLM_MODEL' in os.environ:
            return "glm"
        elif 'DEEPSEEK_API_KEY' in os.environ and os.environ.get('DEEPSEEK_API_KEY'):
            return "deepseek"
        elif os.environ.get('OPENAI_API_BASE', '').find('ollama') >= 0 or os.environ.get('OPENAI_API_KEY') == 'ollama':
            return "ollama"
        elif os.environ.get('AZURE_OPENAI_API_KEY'):
            return "azure"
        else:
            return "openai"
    
    # Replace the function directly in the module
    hirag.determine_default_provider = new_determine_default_provider
    
    # Create completely new completion functions for Cohere
    async def cohere_best_model_complete(*args, **kwargs):
        """Cohere version of the best model complete function"""
        logger.info("Using Cohere command model for best model completion")
        # Make sure we have the COHERE_CHAT_MODEL set to a sensible default
        if "COHERE_CHAT_MODEL" not in os.environ:
            os.environ["COHERE_CHAT_MODEL"] = "command"
        return await cohere_completion(*args, **kwargs)
    
    async def cohere_cheap_model_complete(*args, **kwargs):
        """Cohere version of the cheap model complete function"""
        logger.info("Using Cohere command-light model for cheap model completion")
        # Set to the cheaper model
        temp_model = os.environ.get("COHERE_CHAT_MODEL")
        os.environ["COHERE_CHAT_MODEL"] = "command-light"
        result = await cohere_completion(*args, **kwargs)
        # Restore original model
        if temp_model:
            os.environ["COHERE_CHAT_MODEL"] = temp_model
        return result
    
    # Patch HiRAG class to handle Cohere
    original_post_init = hirag.HiRAG.__post_init__
    
    def patched_post_init(self):
        """Patched version that adds Cohere support"""
        # Call the original __post_init__
        original_post_init(self)
        
        # Add cohere embedding if provider is cohere
        provider = os.environ.get("PROVIDER", hirag.determine_default_provider()).lower()
        
        if provider == "cohere":
            logger.info("Using Cohere for embeddings")
            self.embedding_func = _llm.cohere_embedding
            
            # Update the embedding dimension to match Cohere
            self.node2vec_params["dimensions"] = 1024
            
            # Override LLM functions with Cohere versions
            logger.info("Overriding LLM functions with Cohere versions")
            self.best_model_func = cohere_best_model_complete
            self.cheap_model_func = cohere_cheap_model_complete
    
    # Replace the original method with the patched one
    hirag.HiRAG.__post_init__ = patched_post_init
    
    logger.info("Successfully patched HiRAG library with Cohere embedding and completion support")
    return True

if __name__ == "__main__":
    patch_hirag() 