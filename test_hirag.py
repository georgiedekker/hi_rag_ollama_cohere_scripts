#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import os
import tempfile
import shutil
from pathlib import Path
import yaml
import asyncio
from hirag import HiRAG, QueryParam

class TestHiRAG(unittest.TestCase):
    def setUp(self):
        """Set up a temporary directory for testing"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / "config.yaml"
        self.test_doc_path = Path(self.temp_dir) / "test_document.txt"
        
        # Create a test document
        with open(self.test_doc_path, "w") as f:
            f.write("""
HiRAG: Hierarchical Retrieval Augmented Generation

HiRAG is an advanced retrieval augmented generation system that uses hierarchy in the construction of a multi-layer knowledge graph.

Key features:
1. Hierarchical knowledge representation
2. Improved retrieval accuracy
3. Better context understanding
4. More comprehensive responses

Compared to traditional RAG systems, HiRAG shows significant improvements in:
- Comprehensiveness
- Empowerment
- Diversity
- Overall quality of responses

The architecture consists of:
- Global knowledge layer
- Bridge knowledge layer
- Local knowledge layer

HiRAG outperforms other RAG approaches including NaiveRAG, GraphRAG, LightRAG, FastGraphRAG, and KAG.
            """)
        
        # Create a minimal config for testing
        with open(self.config_path, "w") as f:
            f.write("""
hirag:
  working_dir: "{}"
  enable_llm_cache: false
  enable_hierachical_mode: true
  embedding_batch_num: 1
  embedding_func_max_async: 2
  enable_naive_rag: true
""".format(self.temp_dir))
        
        # Set up a mock embedding function and LLM function
        self.embedding_dim = 3584  # Updated to match the actual embedding dimension
        
        # Mock functions are defined in the test_hirag_system method
        
    def tearDown(self):
        """Clean up temporary directory"""
        shutil.rmtree(self.temp_dir)
    
    def test_hirag_system(self):
        """Test the HiRAG system with mock functions"""
        # Load config
        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)
        
        # Define a mock embedding function that returns consistent vectors
        async def mock_embedding_func(texts):
            # Return simple embeddings for testing with correct dimensions
            import numpy as np
            return np.array([[0.1] * self.embedding_dim] * len(texts))
        
        # Wrap the embedding function with metadata
        from hirag_demo_source import EmbeddingFunc
        wrapped_embedding_func = EmbeddingFunc(
            embedding_dim=self.embedding_dim,
            max_token_size=1000,
            func=mock_embedding_func
        )
        
        # Define a mock LLM function that returns a predefined response
        async def mock_llm_func(prompt, system_prompt=None, history_messages=[], **kwargs):
            # Simple response that includes some of the prompt to verify it's being used
            return f"Response to: {prompt[:20]}..."
        
        # Initialize HiRAG with mock functions
        hirag = HiRAG(
            working_dir=config['hirag']['working_dir'],
            enable_llm_cache=config['hirag']['enable_llm_cache'],
            embedding_func=wrapped_embedding_func,
            best_model_func=mock_llm_func,
            cheap_model_func=mock_llm_func,
            enable_hierachical_mode=config['hirag']['enable_hierachical_mode'],
            embedding_batch_num=config['hirag']['embedding_batch_num'],
            embedding_func_max_async=config['hirag']['embedding_func_max_async'],
            enable_naive_rag=config['hirag']['enable_naive_rag']
        )
        
        # Index the test document
        with open(self.test_doc_path, "r") as f:
            content = f.read()
        hirag.insert(content)
        
        # Test queries with different modes
        test_query = "What is HiRAG?"
        for mode in ["hi", "naive", "hi_nobridge", "hi_local", "hi_global", "hi_bridge"]:
            result = hirag.query(test_query, param=QueryParam(mode=mode))
            self.assertIsNotNone(result)
            self.assertIsInstance(result, str)
            self.assertTrue(len(result) > 0)
            
            # For more thorough testing, we could check that different modes return
            # different results, but with our mock LLM they'll all be similar

def load_module_from_file(file_path, module_name):
    """Load a module from a file path for testing purposes"""
    import importlib.util
    import sys
    
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

if __name__ == "__main__":
    # First, we need to load the hi_rag_demo.py as a module for testing
    hirag_file = Path(__file__).parent / "hi_rag_demo.py"
    if hirag_file.exists():
        try:
            # Import the demo file as a module for testing
            demo_module = load_module_from_file(str(hirag_file), "hirag_demo_source")
            print(f"Successfully loaded {hirag_file}")
        except Exception as e:
            print(f"Error loading {hirag_file}: {e}")
            print("Tests may fail if the module can't be loaded")
    else:
        print(f"Warning: {hirag_file} not found. Make sure it exists before running tests.")
        
    unittest.main() 