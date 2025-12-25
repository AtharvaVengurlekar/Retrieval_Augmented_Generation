## Retrieval Augmented Generation (RAG) – PDF Question Answering
1)This repository demonstrates a Retrieval Augmented Generation (RAG) pipeline built using LangChain, enabling users to upload multiple PDF documents and ask questions that are answered strictly based on the content of those PDFs.
2)The system ensures no hallucinations by grounding responses only in retrieved document context.

# Features
1)Upload and process multiple PDF documents
2)Context-aware question answering
3)Local inference (no external APIs required)
4)Optimized for low VRAM GPUs using INT4 quantized LLMs
5)Strict document-grounded responses

# Setup Instructions
1️)Clone the Repository
```bash
git clone https://github.com/AtharvaVengurlekar/Retrieval_Augmented_Generation.git
cd Retrieval_Augmented_Generation
```
2)Create & Activate Conda Environment (Python 3.10.13)
```bash
conda create -n rag_env python=3.10.13 -y
conda activate rag_env
```

3️)Install Dependencies
```bash
pip install -r requirements.txt
```

## What is LangChain?
1)LangChain is a framework for building applications powered by Large Language Models (LLMs). It provides abstractions to:
2)Connect LLMs with external data sources (PDFs, databases, APIs)
3)Manage prompts, memory, and execution chains
4)Build Retrieval Augmented Generation (RAG) pipelines efficiently
5)LangChain acts as the orchestration layer between your documents, embeddings, vector databases, and LLMs.

## Architecture Overview

![LangChain_Architecture](https://github.com/user-attachments/assets/d9885981-aa63-4950-8b0f-d1a5f949e370)

## Application Workflow
The application follows this pipeline:
1)Document Ingestion:
Multiple PDF files are uploaded and their textual content is extracted.

2️)Text Segmentation:
Extracted text is split into smaller overlapping chunks to preserve context and improve retrieval accuracy.

3️)Embedding Generation:
Each text chunk is converted into a numerical vector using an embedding model.

4️)Semantic Retrieval:
User queries are embedded and matched against stored vectors to retrieve the most relevant document segments.

5)Answer Synthesis
Retrieved segments are passed to the language model, which generates answers strictly grounded in document content.

## Models Used
🔹 Large Language Model (LLM)

## Mistral 7B Instruct

🔹 Embedding Model

## Sentence-Transformer: all-mpnet-base-v2

## Minimum Hardware Requirements
# Recommended

| Component | Requirement |
|---------|------------|
| GPU | NVIDIA RTX GPU |
| Minimum VRAM | 6 GB+ |
| Inference Type | INT4 Quantized Inference |

## Embedding Model Comparison:
| Feature | Instructor-XL | all-mpnet-base-v2 |
|--------|---------------|------------------|
| Model Size | 1.3B parameters | 110M parameters |
| Embedding Size | 768 | 768 |
| GPU Dependency | Required | Not Required |
| Inference Speed | Slow | Fast |
| Memory Consumption | Very High | Low |
| Offline Support | Limited | Full |
| Production Readiness | Medium | High |

## LLM GPU Memory Consumption
> Approximate VRAM usage for **single-GPU inference**.  
> Actual usage may vary based on framework, batch size, and context length.

| Model | Parameters | FP16 / BF16 VRAM | INT8 VRAM | INT4 VRAM |
|-------|------------|------------------|-----------|-----------|
| Google T5-Large | 0.77B | 2–3 GB | 1.5 GB | 0.8–1 GB |
| Google T5-XL | 3B | 6–8 GB | 4 GB | 2–2.5 GB |
| LLaMA 3.2 3B Instruct | 3B | 6–7 GB | 4 GB | 2–2.5 GB |
| Mistral 7B Instruct | 7B | 13–15 GB | 8 GB | 4–5 GB |
| Qwen 7B Instruct | 7B | 13–15 GB | 8 GB | 4–5 GB |

## Downloading Models

Download required models locally by running:

python download.py

🔐 Hugging Face Token (Required for Google T5 Models)

Set your Hugging Face token as an environment variable:

Windows
set HF_TOKEN=your_huggingface_token

Linux / macOS
export HF_TOKEN=your_huggingface_token

▶️ Running the Application

Start the Streamlit app:

streamlit run main.py

🧪 Usage Steps

Upload one or more PDF documents

Click Process PDFs

Ask questions related to the uploaded content

View answers grounded strictly in the PDFs

📌 Expected Output

📌 (Insert expected output screenshot here)

📝 Notes

Designed for local inference

Optimized for INT4 quantized LLMs

Responses are strictly restricted to provided document context

⚠️ Limitations

Answers are limited to the content of uploaded PDFs

Large or complex documents may increase processing time

Answer quality depends on document clarity and structure
