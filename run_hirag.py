#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import logging
import shutil
import glob
from pathlib import Path
from hi_rag_demo import initialize_hirag, index_document, process_query
from hirag import QueryParam

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def clean_vector_database(data_dir):
    """Clean the vector database files to fix dimension mismatch"""
    logger.info(f"Cleaning vector database in {data_dir}")
    
    # Files to delete
    vdb_files = [
        'vdb_entities.json',
        'vdb_entities.index',
        'graph_chunk_entity_relation.graphml',
        'graph_chunk_entity_relation.gra',
        'text_chunks.json',
        'full_docs.json',
        'community_reports.json',
        'llm_response_cache.json'
    ]
    
    for file in vdb_files:
        file_path = os.path.join(data_dir, file)
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"Deleted {file_path}")
            except Exception as e:
                logger.error(f"Error deleting {file_path}: {e}")
    
    # Also clean any other files that might be related
    for pattern in ['*.index', '*.gra', '*.graphml']:
        for file_path in glob.glob(os.path.join(data_dir, pattern)):
            try:
                os.remove(file_path)
                logger.info(f"Deleted {file_path}")
            except Exception as e:
                logger.error(f"Error deleting {file_path}: {e}")
    
    logger.info("Vector database cleaned successfully")
    return True

def main():
    """Main function to run HiRAG with automatic indexing"""
    parser = argparse.ArgumentParser(description='HiRAG Runner')
    parser.add_argument('--query', type=str, required=True, help='Query to process')
    parser.add_argument('--mode', type=str, default='hi', 
                        choices=['hi', 'naive', 'hi_nobridge', 'hi_local', 'hi_global', 'hi_bridge'],
                        help='Query mode')
    parser.add_argument('--document', type=str, default='sample_document.txt', 
                        help='Path to the document to index (default: sample_document.txt)')
    parser.add_argument('--force-reindex', action='store_true', 
                        help='Force reindexing even if vector store exists')
    parser.add_argument('--clean', action='store_true',
                        help='Clean the vector database before starting')
    
    args = parser.parse_args()
    
    # Get data directory path
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    
    # Clean vector database if requested
    if args.clean:
        clean_vector_database(data_dir)
    
    # Try to initialize HiRAG
    try:
        graph_func = initialize_hirag()
    except AssertionError as e:
        # Check if it's a dimension mismatch error
        error_msg = str(e)
        if "Embedding dim mismatch" in error_msg:
            logger.error(f"Dimension mismatch detected: {error_msg}")
            logger.info("Cleaning vector database to fix dimension mismatch...")
            
            if clean_vector_database(data_dir):
                # Try again after cleaning
                logger.info("Retrying initialization...")
                try:
                    graph_func = initialize_hirag()
                except Exception as retry_e:
                    logger.error(f"Failed to initialize after cleaning: {retry_e}")
                    return 1
            else:
                logger.error("Failed to clean vector database")
                return 1
        else:
            # If it's a different assertion error, re-raise it
            logger.error(f"Initialization error: {e}")
            return 1
    except Exception as e:
        logger.error(f"Error initializing HiRAG: {e}")
        return 1
    
    # Check if we need to index a document
    vdb_file = os.path.join(data_dir, 'vdb_entities.json')
    
    # Index document if vector store doesn't exist or force-reindex is set
    if args.force_reindex or not os.path.exists(vdb_file) or os.path.getsize(vdb_file) < 100:
        logger.info("Vector store not found or reindexing forced. Indexing document...")
        
        document_path = args.document
        if not os.path.exists(document_path):
            # Try to find the document relative to this script's directory
            document_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.document)
            
        if not os.path.exists(document_path):
            logger.error(f"Document not found: {args.document}")
            return 1
            
        success = index_document(graph_func, document_path)
        if not success:
            logger.error("Failed to index document.")
            return 1
            
        logger.info(f"Successfully indexed document: {document_path}")
    else:
        logger.info("Using existing vector store. Skipping indexing.")
    
    # Process the query
    logger.info(f"Processing query: {args.query} (mode: {args.mode})")
    
    try:
        result = process_query(graph_func, args.query, args.mode)
        print("\nQuery Result:")
        print("-" * 80)
        print(result)
        print("-" * 80)
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code) 