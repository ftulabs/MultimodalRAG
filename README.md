# Ví dụ sử dụng
pip install -r requirements.txt
python examples/raganything_example.py path_to_document (ex: examples/test.txt)

# Giải thích Kiến trúc và Cấu trúc Repo RAG-Anything

## 1. **Kiến trúc Tổng quan**

```
RAG-Anything
├── Document Parsing Layer (MinerU/Docling)
├── Content Processing Layer (Text + Multimodal)
├── Storage Layer (LightRAG KV Storage)
├── Query Layer (Text + VLM Enhanced)
└── Batch Processing Layer
```

## 2. **Core Components - Các File Chính**

### **raganything.py** - Class Chính
```python
@dataclass
class RAGAnything(QueryMixin, ProcessorMixin, BatchMixin):
```
- **Vai trò**: Điểm vào chính của hệ thống, kết hợp tất cả các mixin
- **Chức năng**:
  - Khởi tạo LightRAG instance
  - Quản lý modal processors (image, table, equation)
  - Quản lý parse cache
  - Cung cấp API thống nhất

### **config.py** - Cấu hình
```python
@dataclass
class RAGAnythingConfig:
    working_dir: str
    parser: str  # "mineru" hoặc "docling"
    parse_method: str  # "auto", "ocr", "txt"
    enable_image_processing: bool
    enable_table_processing: bool
    enable_equation_processing: bool
```
- **Vai trò**: Quản lý tất cả cấu hình hệ thống
- **Hỗ trợ**: Environment variables

## 3. **Processing Pipeline - Luồng Xử lý**

### **A. Document Parsing (parser.py)**

```
File Input → Parser Selection → Conversion (if needed) → Content List
```

**Hai Parser chính**:

1. **MineruParser**: 
   - Hỗ trợ: PDF, images (.jpg, .png, .bmp, .tiff, .gif, .webp)
   - Office docs (convert qua PDF trước)
   - Text files (convert qua PDF trước)

2. **DoclingParser**:
   - Hỗ trợ: PDF, Office docs, HTML
   - Xử lý trực tiếp Office docs

**Output**: `content_list` - danh sách các block content với cấu trúc:
```json
[
  {"type": "text", "text": "...", "page_idx": 0},
  {"type": "image", "img_path": "...", "image_caption": [...], "page_idx": 1},
  {"type": "table", "table_body": "...", "table_caption": [...], "page_idx": 2},
  {"type": "equation", "latex": "...", "page_idx": 3}
]
```

### **B. Content Separation (utils.py)**

```python
def separate_content(content_list):
    text_parts = []
    multimodal_items = []
    # Tách text và multimodal content
    return text_content, multimodal_items
```

### **C. Text Processing (processor.py)**

```
Text Content → LightRAG.ainsert() → Chunking → Entity Extraction → Graph Building
```

### **D. Multimodal Processing (modalprocessors.py)**

**4 Processors chính**:

1. **ImageModalProcessor**:
   - Sử dụng Vision Model để phân tích ảnh
   - Tạo entity từ ảnh
   - Hỗ trợ context extraction

2. **TableModalProcessor**:
   - Phân tích cấu trúc bảng
   - Trích xuất data insights
   - Tạo entity từ bảng

3. **EquationModalProcessor**:
   - Phân tích công thức toán học
   - Giải thích ý nghĩa
   - Tạo entity từ equation

4. **GenericModalProcessor**:
   - Xử lý các loại content khác
   - Fallback processor

**Context Extraction**:
```python
class ContextExtractor:
    def extract_context(self, content_source, current_item_info):
        # Lấy context từ các trang/chunk xung quanh
        # Hỗ trợ page-based và chunk-based mode
```

**Batch Processing Flow**:
```
Stage 1: Generate Descriptions (parallel)
  └─> LLM/Vision Model calls

Stage 2: Convert to LightRAG Chunks
  └─> Apply chunk templates

Stage 3: Store to Storage
  └─> text_chunks + chunks_vdb

Stage 4: Entity Extraction
  └─> LightRAG's extract_entities

Stage 5: Add belongs_to Relations
  └─> Link entities to main modal entity

Stage 6: Merge to Knowledge Graph
  └─> LightRAG's merge_nodes_and_edges

Stage 7: Update doc_status
```

## 4. **Storage Layer**

```
./rag_storage/
├── graph_chunk_entity_relation.graphml  # Knowledge Graph
├── kv_store_full_docs.json             # Full documents
├── kv_store_text_chunks.json           # Text chunks
├── kv_store_full_entities.json         # All entities
├── kv_store_full_relations.json        # All relations
├── kv_store_llm_response_cache.json    # LLM cache
├── kv_store_parse_cache.json           # Parse cache
├── doc_status.json                      # Document status
└── vdb_*.json                          # Vector databases
```

**Parse Cache**:
- Key: MD5(file_path + mtime + parse_config)
- Value: {content_list, doc_id, mtime, parse_config}
- Tránh parse lại documents không thay đổi

## 5. **Query Layer (query.py)**

### **A. Text Query**
```python
async def aquery(self, query: str, mode: str = "mix"):
    # Gọi trực tiếp LightRAG.aquery()
```

### **B. Multimodal Query**
```python
async def aquery_with_multimodal(
    self, query: str, 
    multimodal_content: List[Dict]
):
    # Process multimodal content
    # Generate enhanced query
    # Execute với LightRAG
```

### **C. VLM Enhanced Query**
```python
async def aquery_vlm_enhanced(self, query: str):
    # 1. Get retrieval context
    # 2. Extract image paths
    # 3. Convert images to base64
    # 4. Build VLM messages
    # 5. Call Vision Model
```

**Flow**:
```
Query → Retrieve Context → [Has Images?]
                              ├─> Yes: Replace paths with base64
                              │         ↓
                              │   Call Vision Model
                              └─> No: Normal LLM call
```

## 6. **Batch Processing (batch.py, batch_parser.py)**

### **batch_parser.py** - Low-level Batch Parsing
```python
class BatchParser:
    def process_batch(self, file_paths, output_dir):
        # Parallel parsing với ThreadPoolExecutor
        # Progress tracking với tqdm
        # Error handling
```

### **batch.py** - High-level Batch Processing
```python
class BatchMixin:
    async def process_documents_batch(self):
        # Parse documents
        
    async def process_documents_with_rag_batch(self):
        # Parse + Insert vào RAG
```

## 7. **Prompt Templates (prompt.py)**

Chứa tất cả prompts cho:
- Image analysis
- Table analysis  
- Equation analysis
- Generic content analysis
- Các prompt có/không có context

## 8. **Examples**

### **raganything_example.py**
- Demo sử dụng Qwen3-VL-4B local model
- Ollama embeddings
- Process document + query

### **insert_content_list_example.py**
- Demo insert content list trực tiếp
- Không cần parse document

### **batch_processing_example.py**
- Demo xử lý nhiều files
- Async batch processing
- Error handling

### **enhanced_markdown_example.py**
- Convert markdown sang PDF
- Multiple backends (WeasyPrint, Pandoc)
- Custom CSS styling

## 9. **Luồng Xử lý Hoàn chỉnh**

```
1. Document Input
   ↓
2. Parse Document (parser.py)
   → content_list + doc_id
   ↓
3. Separate Content (utils.py)
   → text_content + multimodal_items
   ↓
4. Process Text (processor.py)
   → LightRAG.ainsert()
   ↓
5. Process Multimodal (modalprocessors.py)
   ├─> Stage 1: Generate descriptions (parallel)
   ├─> Stage 2: Create chunks
   ├─> Stage 3: Store chunks
   ├─> Stage 4: Extract entities
   ├─> Stage 5: Add relations
   └─> Stage 6: Merge to graph
   ↓
6. Update doc_status
   ↓
7. Ready for Query
```

## 10. **Key Features**

### **Caching System**
- Parse cache: Tránh parse lại documents
- LLM response cache: Tránh gọi LLM lại
- Multimodal query cache: Cache kết quả query

### **Context-Aware Processing**
- Extract context từ surrounding pages
- Cung cấp context cho modal processors
- Improve entity/relation quality

### **Flexible Architecture**
- Pluggable parsers (MinerU/Docling)
- Pluggable processors (Image/Table/Equation)
- Support custom LLM/Vision models

### **Error Handling**
- Graceful fallbacks
- Individual processing fallback
- Comprehensive error logging

## 11. **Dependencies**

**Core**:
- `lightrag-hku`: Knowledge graph RAG framework
- `transformers`: Qwen models
- `torch`: Deep learning

**Parsing**:
- `mineru`: Document parsing (primary)
- `docling`: Alternative parser
- `reportlab`: Text to PDF conversion

**Storage**:
- JSON-based KV stores
- NetworkX graph storage

**Query**:
- `ollama`: Local embeddings
- Vision models: Qwen3-VL
# MultimodalRAG
# MultimodalRAG
