#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import unittest
import tempfile
import json
import shutil
from pathlib import Path
import yaml
from unittest.mock import patch, MagicMock, Mock

# Import the functions from the pipeline integration script
from pipeline_integration import (
    load_chunked_data,
    load_ner_data,
    load_ingested_data,
    combine_data,
    save_combined_data,
    initialize_hirag,
    load_config
)

class TestPipelineIntegration(unittest.TestCase):
    """Test cases for the pipeline integration script"""
    
    def setUp(self):
        """Set up test environment"""
        # Create temporary directories for test data
        self.temp_dir = tempfile.mkdtemp()
        self.ingest_dir = os.path.join(self.temp_dir, "ingest")
        self.ner_dir = os.path.join(self.temp_dir, "ner")
        self.chunker_dir = os.path.join(self.temp_dir, "chunker")
        self.output_dir = os.path.join(self.temp_dir, "output")
        
        # Create the directories
        os.makedirs(self.ingest_dir, exist_ok=True)
        os.makedirs(self.ner_dir, exist_ok=True)
        os.makedirs(self.chunker_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create some test data files
        
        # Ingest data (text file)
        with open(os.path.join(self.ingest_dir, "document.txt"), "w") as f:
            f.write("This is a test document for ingestion.\n")
            f.write("It contains some sample text to test the pipeline integration.\n")
        
        # NER data (JSON file with entities)
        ner_data = {
            "entities": [
                {"text": "test document", "type": "DOCUMENT", "start": 10, "end": 23},
                {"text": "pipeline integration", "type": "CONCEPT", "start": 65, "end": 85}
            ]
        }
        with open(os.path.join(self.ner_dir, "entities.json"), "w") as f:
            json.dump(ner_data, f)
        
        # Chunker data (JSON file with chunks)
        chunker_data = {
            "chunks": [
                {"id": 1, "text": "This is a test document for ingestion."},
                {"id": 2, "text": "It contains some sample text to test the pipeline integration."}
            ]
        }
        with open(os.path.join(self.chunker_dir, "graph_output.json"), "w") as f:
            json.dump(chunker_data, f)
            
        # Create a test config file
        self.config_file = os.path.join(self.temp_dir, "config.yaml")
        config_data = {
            "openai": {
                "embedding_model": "test-model",
                "model": "test-model",
                "api_key": "test-key",
                "base_url": "http://localhost:11434/v1"
            },
            "glm": {
                "model": "test-model",
                "api_key": "test-key",
                "base_url": "http://localhost:11434/v1",
                "embedding_model": "test-model"
            },
            "model_params": {
                "glm_embedding_dim": 768,
                "max_token_size": 4096
            },
            "neo4j": {
                "url": "neo4j://localhost:7687",
                "user": "neo4j",
                "password": "testpassword"
            }
        }
        with open(self.config_file, 'w') as f:
            yaml.dump(config_data, f)
    
    def tearDown(self):
        """Clean up test environment"""
        # Remove the temporary directory and all its contents
        shutil.rmtree(self.temp_dir)
    
    def test_load_chunked_data(self):
        """Test loading chunked data"""
        chunked_data = load_chunked_data(self.chunker_dir)
        self.assertIsNotNone(chunked_data)
        self.assertIn("chunks", chunked_data)
        self.assertEqual(len(chunked_data["chunks"]), 2)
    
    def test_load_ner_data(self):
        """Test loading NER data"""
        ner_data = load_ner_data(self.ner_dir)
        self.assertIsNotNone(ner_data)
        self.assertIn("entities", ner_data.lower())
        self.assertIn("test document", ner_data)
        self.assertIn("pipeline integration", ner_data)
    
    def test_load_ingested_data(self):
        """Test loading ingested data"""
        ingested_data = load_ingested_data(self.ingest_dir)
        self.assertIsNotNone(ingested_data)
        self.assertIn("test document", ingested_data)
        self.assertIn("pipeline integration", ingested_data)
    
    def test_combine_data(self):
        """Test combining data from different sources"""
        chunked_data = load_chunked_data(self.chunker_dir)
        ner_data = load_ner_data(self.ner_dir)
        ingested_data = load_ingested_data(self.ingest_dir)
        
        combined_data = combine_data(chunked_data, ner_data, ingested_data)
        self.assertIsNotNone(combined_data)
        self.assertIn("Document Metadata", combined_data)
        self.assertIn("Chunked Data", combined_data)
        self.assertIn("Named Entity Data", combined_data)
        self.assertIn("Original Document Content", combined_data)
    
    def test_save_combined_data(self):
        """Test saving combined data to a file"""
        chunked_data = load_chunked_data(self.chunker_dir)
        ner_data = load_ner_data(self.ner_dir)
        ingested_data = load_ingested_data(self.ingest_dir)
        
        combined_data = combine_data(chunked_data, ner_data, ingested_data)
        output_file = os.path.join(self.output_dir, "combined_data.txt")
        
        result = save_combined_data(combined_data, output_file)
        self.assertTrue(result)
        self.assertTrue(os.path.exists(output_file))
        
        # Check that the content is correct
        with open(output_file, "r") as f:
            content = f.read()
        self.assertEqual(content, combined_data)
        
    def test_load_config(self):
        """Test loading configuration from YAML file"""
        config = load_config(self.config_file)
        self.assertIsNotNone(config)
        self.assertIn("openai", config)
        self.assertIn("glm", config)
        self.assertIn("neo4j", config)
        self.assertEqual(config["neo4j"]["user"], "neo4j")
        self.assertEqual(config["neo4j"]["password"], "testpassword")
    
    @patch('pipeline_integration.HiRAG')
    @patch('pipeline_integration.Neo4jStorage')
    @patch('pipeline_integration.NetworkXStorage')
    @patch('pipeline_integration.NanoVectorDBStorage')
    @patch('pipeline_integration.HNSWVectorStorage')
    def test_initialize_hirag_with_networkx(self, mock_hnswlib, mock_nano, mock_networkx, mock_neo4j, mock_hirag):
        """Test initializing HiRAG with NetworkX storage"""
        config = load_config(self.config_file)
        
        # Mock HiRAG implementation to avoid actual initialization
        mock_hirag_instance = MagicMock()
        mock_hirag.return_value = mock_hirag_instance
        
        # Call the function
        graph_func = initialize_hirag(
            config=config,
            working_dir=self.output_dir,
            use_neo4j=False
        )
        
        # Check that HiRAG was initialized with correct parameters
        mock_hirag.assert_called_once()
        # Check that NetworkX was used, not Neo4j
        self.assertEqual(mock_hirag.call_args[1]['graph_storage_cls'], mock_networkx)
        self.assertNotEqual(mock_hirag.call_args[1]['graph_storage_cls'], mock_neo4j)
        
    @patch('pipeline_integration.HiRAG')
    @patch('pipeline_integration.Neo4jStorage')
    @patch('pipeline_integration.NetworkXStorage')
    @patch('pipeline_integration.NanoVectorDBStorage')
    @patch('pipeline_integration.HNSWVectorStorage')
    def test_initialize_hirag_with_neo4j(self, mock_hnswlib, mock_nano, mock_networkx, mock_neo4j, mock_hirag):
        """Test initializing HiRAG with Neo4j storage"""
        config = load_config(self.config_file)
        
        # Mock HiRAG implementation to avoid actual initialization
        mock_hirag_instance = MagicMock()
        mock_hirag.return_value = mock_hirag_instance
        
        # Call the function
        graph_func = initialize_hirag(
            config=config,
            working_dir=self.output_dir,
            use_neo4j=True,
            neo4j_url="neo4j://localhost:7687",
            neo4j_user="neo4j",
            neo4j_password="password"
        )
        
        # Check that HiRAG was initialized with correct parameters
        mock_hirag.assert_called_once()
        # Check that Neo4j was used, not NetworkX
        self.assertEqual(mock_hirag.call_args[1]['graph_storage_cls'], mock_neo4j)
        self.assertNotEqual(mock_hirag.call_args[1]['graph_storage_cls'], mock_networkx)
        # Check that addon_params has neo4j configuration
        self.assertIn('neo4j_url', mock_hirag.call_args[1]['addon_params'])
        self.assertIn('neo4j_auth', mock_hirag.call_args[1]['addon_params'])
        
    @patch('pipeline_integration.HiRAG')
    @patch('pipeline_integration.Neo4jStorage')
    @patch('pipeline_integration.NetworkXStorage')
    @patch('pipeline_integration.NanoVectorDBStorage')
    @patch('pipeline_integration.HNSWVectorStorage')
    def test_initialize_hirag_with_hnswlib(self, mock_hnswlib, mock_nano, mock_networkx, mock_neo4j, mock_hirag):
        """Test initializing HiRAG with HNSWLib vector database"""
        config = load_config(self.config_file)
        
        # Mock HiRAG implementation to avoid actual initialization
        mock_hirag_instance = MagicMock()
        mock_hirag.return_value = mock_hirag_instance
        
        # Call the function
        graph_func = initialize_hirag(
            config=config,
            working_dir=self.output_dir,
            use_neo4j=False,
            use_hnswlib=True
        )
        
        # Check that HiRAG was initialized with correct parameters
        mock_hirag.assert_called_once()
        # Check that HNSWLib was used, not NanoVectorDB
        self.assertEqual(mock_hirag.call_args[1]['vector_db_storage_cls'], mock_hnswlib)
        self.assertNotEqual(mock_hirag.call_args[1]['vector_db_storage_cls'], mock_nano)
        
    @patch('pipeline_integration.HiRAG')
    def test_initialize_hirag_with_custom_chunking(self, mock_hirag):
        """Test initializing HiRAG with custom chunking parameters"""
        config = load_config(self.config_file)
        
        # Mock HiRAG implementation to avoid actual initialization
        mock_hirag_instance = MagicMock()
        mock_hirag.return_value = mock_hirag_instance
        
        # Custom chunking parameters
        custom_chunk_size = 2000
        custom_chunk_overlap = 300
        
        # Call the function
        graph_func = initialize_hirag(
            config=config,
            working_dir=self.output_dir,
            chunk_token_size=custom_chunk_size,
            chunk_overlap_token_size=custom_chunk_overlap
        )
        
        # Check that HiRAG was initialized with correct parameters
        mock_hirag.assert_called_once()
        # Check that custom chunking parameters were used
        self.assertEqual(mock_hirag.call_args[1]['chunk_token_size'], custom_chunk_size)
        self.assertEqual(mock_hirag.call_args[1]['chunk_overlap_token_size'], custom_chunk_overlap)
    
    def test_end_to_end(self):
        """Test end-to-end pipeline integration"""
        # Load data from each source
        chunked_data = load_chunked_data(self.chunker_dir)
        ner_data = load_ner_data(self.ner_dir)
        ingested_data = load_ingested_data(self.ingest_dir)
        
        # Combine data
        combined_data = combine_data(chunked_data, ner_data, ingested_data)
        
        # Save combined data to a file
        combined_file = os.path.join(self.output_dir, "combined_pipeline_data.txt")
        save_result = save_combined_data(combined_data, combined_file)
        
        # Validate results
        self.assertTrue(save_result)
        self.assertTrue(os.path.exists(combined_file))
        self.assertIn("test document", combined_data)
        self.assertIn("pipeline integration", combined_data)

if __name__ == "__main__":
    unittest.main() 