#!/usr/bin/env python
"""
Gradio Demo for RAGAnything + MinerU/Docling
Dựa hoàn toàn vào các hàm trong file example gốc:
- process_document_complete()
- aquery()
- aquery_with_multimodal()

Không thay đổi cách import hoặc cấu trúc hệ thống.
"""

import os
import asyncio
import logging
import logging.config
from pathlib import Path
import sys

# giữ nguyên import y hệt bản gốc
sys.path.append(str(Path(__file__).parent.parent))

from qwen_model import get_qwen_model
from lightrag.utils import EmbeddingFunc, logger, set_verbose_debug
from sentence_transformers import SentenceTransformer
from raganything import RAGAnything, RAGAnythingConfig

from dotenv import load_dotenv
import gradio as gr

load_dotenv(dotenv_path=".env", override=False)


# ------------------------------
# Logging setup
# ------------------------------
def configure_logging():
    log_dir = os.getenv("LOG_DIR", os.getcwd())
    log_file_path = os.path.abspath(os.path.join(log_dir, "raganything_gradio.log"))

    print(f"\nLog file: {log_file_path}\n")
    os.makedirs(log_dir, exist_ok=True)

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "handlers": {
                "console": {"class": "logging.StreamHandler"},
                "file": {
                    "class": "logging.FileHandler",
                    "filename": log_file_path,
                    "encoding": "utf-8",
                },
            },
            "loggers": {"lightrag": {"handlers": ["console", "file"], "level": "INFO"}},
        }
    )
    set_verbose_debug(os.getenv("VERBOSE", "false").lower() == "true")


# ------------------------------
# Init RAGAnything once
# ------------------------------
async def init_rag():
    logger.info("Loading Qwen3-VL model...")
    qwen = get_qwen_model()

    embedding_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    hf_model = SentenceTransformer(embedding_name)

    embedding_dim = 384
    if "bge" in embedding_name.lower() or "large" in embedding_name.lower():
        embedding_dim = 1024

    async def async_embedding(texts):
        if isinstance(texts, str):
            texts = [texts]

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: hf_model.encode(texts, convert_to_tensor=False).tolist(),
        )

    embedding_func = EmbeddingFunc(
        embedding_dim=embedding_dim,
        max_token_size=8192,
        func=async_embedding,
    )

    config = RAGAnythingConfig(
        working_dir="./rag_storage",
        parser=os.getenv("PARSER", "mineru"),
        parse_method="auto",
        enable_image_processing=True,
        enable_table_processing=True,
        enable_equation_processing=True,
    )

    return RAGAnything(
        config=config,
        llm_model_func=qwen.llm_func,
        vision_model_func=qwen.vision_func,
        embedding_func=embedding_func,
    )


rag_instance = None


# ------------------------------
# Document Processing
# ------------------------------
async def process_docs(files):
    global rag_instance

    if rag_instance is None:
        rag_instance = await init_rag()

    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)

    results = []
    for f in files:
        path = f.name
        logger.info(f"Processing: {path}")

        await rag_instance.process_document_complete(
            file_path=path,
            output_dir=output_dir,
            parse_method="auto",
        )

        results.append(f"Processed: {os.path.basename(path)}")

    return "\n".join(results)


# ------------------------------
# Text Query
# ------------------------------
async def text_query(q):
    global rag_instance
    if rag_instance is None:
        return "❌ No documents processed yet."

    ans = await rag_instance.aquery(q, mode="hybrid")
    return ans


# ------------------------------
# Multimodal Query (image/table/equation)
# ------------------------------
async def multimodal_query(query, image):
    global rag_instance
    if rag_instance is None:
        return "❌ No document processed."

    if image is None:
        return "❌ Please upload an image for multimodal query."

    mm = [
        {
            "type": "image",
            "image": image,
            "image_caption": "Uploaded image",
        }
    ]

    return await rag_instance.aquery_with_multimodal(
        query,
        multimodal_content=mm,
        mode="hybrid",
    )


# ------------------------------
# Gradio UI
# ------------------------------
def launch_ui():
    with gr.Blocks(title="RAGAnything Gradio Demo") as demo:

        gr.Markdown("# 📄 RAGAnything + MinerU / Docling Demo")
        gr.Markdown("Upload tài liệu → Query text → Query multimodal")

        with gr.Tab("Upload Documents"):
            files = gr.File(file_count="multiple", label="Upload tài liệu")
            btn = gr.Button("Process")
            out = gr.Textbox(label="Status")

            btn.click(process_docs, inputs=files, outputs=out)

        with gr.Tab("Text Query"):
            q = gr.Textbox(label="Your question")
            out2 = gr.Textbox(label="Answer")
            btn2 = gr.Button("Ask")

            btn2.click(text_query, inputs=q, outputs=out2)

        with gr.Tab("Multimodal Query"):
            q2 = gr.Textbox(label="Your question")
            img = gr.Image(label="Upload image")
            out3 = gr.Textbox(label="Answer")
            btn3 = gr.Button("Ask")

            btn3.click(multimodal_query, inputs=[q2, img], outputs=out3)

    demo.launch(server_name="0.0.0.0", server_port=7860)


# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    configure_logging()
    asyncio.run(init_rag())  # preload model
    launch_ui()
