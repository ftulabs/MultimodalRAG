#!/usr/bin/env python
"""
Example script demonstrating the integration of MinerU parser with RAGAnything

This example shows how to:
1. Process documents with RAGAnything using MinerU parser
2. Perform pure text queries using aquery() method
3. Perform multimodal queries with specific multimodal content using aquery_with_multimodal() method
4. Handle different types of multimodal content (tables, equations) in queries

Updated to use Qwen3-VL-4B-Instruct and SentenceTransformer embeddings
"""

import os
import argparse
import asyncio
import logging
import logging.config
from pathlib import Path

# Add project root directory to Python path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from qwen_model import get_qwen_model
from lightrag.utils import EmbeddingFunc, logger, set_verbose_debug
from sentence_transformers import SentenceTransformer
from raganything import RAGAnything, RAGAnythingConfig

from dotenv import load_dotenv

load_dotenv(dotenv_path=".env", override=False)


def configure_logging():
    """Configure logging for the application"""
    # Get log directory path from environment variable or use current directory
    log_dir = os.getenv("LOG_DIR", os.getcwd())
    log_file_path = os.path.abspath(os.path.join(log_dir, "raganything_example.log"))

    print(f"\nRAGAnything example log file: {log_file_path}\n")
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)

    # Get log file max size and backup count from environment variables
    log_max_bytes = int(os.getenv("LOG_MAX_BYTES", 10485760))  # Default 10MB
    log_backup_count = int(os.getenv("LOG_BACKUP_COUNT", 5))  # Default 5 backups

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(levelname)s: %(message)s",
                },
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
            },
            "handlers": {
                "console": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stderr",
                },
                "file": {
                    "formatter": "detailed",
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": log_file_path,
                    "maxBytes": log_max_bytes,
                    "backupCount": log_backup_count,
                    "encoding": "utf-8",
                },
            },
            "loggers": {
                "lightrag": {
                    "handlers": ["console", "file"],
                    "level": "INFO",
                    "propagate": False,
                },
            },
        }
    )

    # Set the logger level to INFO
    logger.setLevel(logging.INFO)
    # Enable verbose debug if needed
    set_verbose_debug(os.getenv("VERBOSE", "false").lower() == "true")


async def process_with_rag(
    file_path: str,
    output_dir: str,
    working_dir: str = None,
    parser: str = None,
):
    """
    Process document with RAGAnything using Qwen model and SentenceTransformer

    Args:
        file_path: Path to the document
        output_dir: Output directory for RAG results
        working_dir: Working directory for RAG storage
        parser: Parser to use (mineru or docling)
    """
    try:
        logger.info("=" * 60)
        logger.info("Initializing RAGAnything with Qwen3-VL-4B-Instruct")
        logger.info("=" * 60)
        
        # Initialize Qwen model
        logger.info("Loading Qwen model...")
        qwen = get_qwen_model()
        
        # Create RAGAnything configuration
        config = RAGAnythingConfig(
            working_dir=working_dir or "./rag_storage",
            parser=parser or "mineru",  # Parser selection: mineru or docling
            parse_method="auto",  # Parse method: auto, ocr, or txt
            enable_image_processing=True,
            enable_table_processing=True,
            enable_equation_processing=True,
        )

        # Setup SentenceTransformer embedding function
        embedding_model_name = os.getenv(
            "EMBEDDING_MODEL", 
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        logger.info(f"Loading embedding model: {embedding_model_name}")
        
        hf_model = SentenceTransformer(embedding_model_name)
        embedding_dim = 384  # all-MiniLM-L6-v2 has 384 dimensions
        
        # If using a different model, adjust the dimension accordingly
        if "bge-m3" in embedding_model_name.lower():
            embedding_dim = 1024
        elif "large" in embedding_model_name.lower():
            embedding_dim = 1024
        
        # Create async wrapper for embedding function that properly handles batches
        async def async_embedding(texts):
            """
            Async wrapper for SentenceTransformer encoding
            
            Args:
                texts: Single string or list of strings to encode
                
            Returns:
                List of embeddings (list of lists of floats)
            """
            # Ensure texts is a list
            if isinstance(texts, str):
                texts = [texts]
            
            # Run encoding in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None, 
                lambda: hf_model.encode(
                    texts, 
                    convert_to_tensor=False,
                    show_progress_bar=False
                ).tolist()
            )
            
            return embeddings
        
        embedding_func = EmbeddingFunc(
            embedding_dim=embedding_dim,
            max_token_size=8192,
            func=async_embedding,
        )

        # Initialize RAGAnything with Qwen and SentenceTransformer
        logger.info("Initializing RAGAnything...")
        rag = RAGAnything(
            config=config,
            llm_model_func=qwen.llm_func,
            vision_model_func=qwen.vision_func,
            embedding_func=embedding_func,
        )

        # Process document
        logger.info(f"\nProcessing document: {file_path}")
        logger.info("-" * 60)
        await rag.process_document_complete(
            file_path=file_path, output_dir=output_dir, parse_method="auto"
        )
        logger.info("✓ Document processing completed!")

        # Example queries - demonstrating different query approaches
        logger.info("\n" + "=" * 60)
        logger.info("Querying Processed Document")
        logger.info("=" * 60)

        # 1. Pure text queries using aquery()
        text_queries = [
            "What is the main content of the document?",
            "What are the key topics discussed?",
            "Summarize the most important points",
        ]

        for i, query in enumerate(text_queries, 1):
            logger.info(f"\n[Text Query {i}]: {query}")
            result = await rag.aquery(query, mode="hybrid")
            logger.info(f"[Answer]: {result}")

        # 2. Multimodal query with specific multimodal content using aquery_with_multimodal()
        logger.info("\n" + "-" * 60)
        logger.info(
            "[Multimodal Query]: Analyzing performance data in context of document"
        )
        multimodal_result = await rag.aquery_with_multimodal(
            "Compare this performance data with any similar results mentioned in the document",
            multimodal_content=[
                {
                    "type": "table",
                    "table_data": """Method,Accuracy,Processing_Time
                                RAGAnything,95.2%,120ms
                                Traditional_RAG,87.3%,180ms
                                Baseline,82.1%,200ms""",
                    "table_caption": "Performance comparison results",
                }
            ],
            mode="hybrid",
        )
        logger.info(f"[Answer]: {multimodal_result}")

        # 3. Another multimodal query with equation content
        logger.info("\n" + "-" * 60)
        logger.info("[Multimodal Query]: Mathematical formula analysis")
        equation_result = await rag.aquery_with_multimodal(
            "Explain this formula and relate it to any mathematical concepts in the document",
            multimodal_content=[
                {
                    "type": "equation",
                    "latex": "F1 = 2 \\cdot \\frac{precision \\cdot recall}{precision + recall}",
                    "equation_caption": "F1-score calculation formula",
                }
            ],
            mode="hybrid",
        )
        logger.info(f"[Answer]: {equation_result}")
        
        logger.info("\n" + "=" * 60)
        logger.info("✓ All queries completed successfully!")
        logger.info(f"Output saved to: {output_dir}")
        logger.info("=" * 60)
        
        # Cleanup
        try:
            await rag.finalize()
        except Exception as cleanup_error:
            logger.warning(f"Cleanup warning: {cleanup_error}")

    except Exception as e:
        logger.error(f"Error processing with RAG: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())


def main():
    """Main function to run the example"""
    parser = argparse.ArgumentParser(
        description="RAGAnything Example with Qwen and SentenceTransformer"
    )
    parser.add_argument("file_path", help="Path to the document to process")
    parser.add_argument(
        "--working_dir", "-w", default="./rag_storage", help="Working directory path"
    )
    parser.add_argument(
        "--output", "-o", default="./output", help="Output directory path"
    )
    parser.add_argument(
        "--parser",
        "-p",
        default=os.getenv("PARSER", "mineru"),
        help="Parser to use: mineru or docling (default: mineru)",
    )

    args = parser.parse_args()

    # Check if file exists
    if not os.path.exists(args.file_path):
        logger.error(f"Error: File not found: {args.file_path}")
        return

    # Create output directory if specified
    if args.output:
        os.makedirs(args.output, exist_ok=True)

    # Create working directory
    os.makedirs(args.working_dir, exist_ok=True)

    # Process with RAG
    asyncio.run(
        process_with_rag(
            args.file_path,
            args.output,
            args.working_dir,
            args.parser,
        )
    )


if __name__ == "__main__":
    # Configure logging first
    configure_logging()

    print("\n" + "=" * 60)
    print("RAGAnything Example - Qwen3-VL + SentenceTransformer")
    print("=" * 60)
    print("Processing document with multimodal RAG pipeline")
    print("Using: Qwen3-VL-4B-Instruct + all-MiniLM-L6-v2")
    print("=" * 60 + "\n")

    main()