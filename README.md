Retrieval Augmented Generation (RAG) – PDF Question Answering
This repository demonstrates a Retrieval Augmented Generation (RAG) pipeline using LangChain that allows you to upload multiple PDF documents and ask questions grounded strictly in their content.
Setup Instructions
1. Clone the Repository
git clone <repository-url>
cd <repository-directory>
2. Create & Activate Conda Environment (Python 3.10.13)
conda create -n rag_env python=3.10.13 -y
conda activate rag_env
3. Install Dependencies
pip install -r requirements.txt
What is LangChain?
LangChain is a framework that helps build applications powered by Large Language Models (LLMs). It provides tools to:
Connect LLMs with external data (PDFs, databases, APIs)
Manage prompts, memory, and chains
Implement Retrieval Augmented Generation (RAG) workflows efficiently
In short, LangChain acts as the orchestration layer between your data, embeddings, vector databases, and LLMs.
Architecture Overview
(Insert your architecture image here)
Application Workflow
The application processes your queries through the following pipeline:
Document Ingestion
Multiple PDF files are uploaded and their textual content is extracted.
Text Segmentation
The extracted text is broken into smaller, overlapping segments to preserve context and improve retrieval accuracy.
Embedding Generation
Each text segment is converted into a numerical vector representation using an embedding model.
Semantic Retrieval
When a question is asked, its embedding is compared against stored vectors to find the most relevant text segments.
Answer Synthesis
The retrieved segments are supplied to the language model, which generates a concise answer strictly based on the document content.
Models Used:
Large Language Model (LLM)
Mistral 7B Instruct
Embedding Model
Sentence-Transformer: all-mpnet-base-v2
Minimum Hardware Requirements:
Recommended:
- NVIDIA RTX GPU with 6 GB+ VRAM 
- INT4 quantized inference for Mistral 7B and Qwen 7B Parameters.
 
Embedding Model Specs Comparison
Property	Instructor-XL	Sentence-Transformer (all-mpnet-base-v2)
Parameters	
1.3B
	
110M

Embedding dim	768	768
GPU needed	
Yes 
	No
Speed	Slow	Fast
Memory usage	Very high	Low
Offline friendly	Limited	Yes
Production stability	Medium	High
LLM GPU Memory Consumption Table:
Model	Params	FP16 / BF16 VRAM	INT8 VRAM	INT4 VRAM
Google T5-Large	
 0.77B
	
 2–3 GB
	
 1.5 GB
	
 0.8–1 GB

Google T5-XL	
  3B
	
 6–8 GB
	
 4 GB
	
 2–2.5 GB

LLaMA 3.2 3B Instruct	
  3B
	
 6–7 GB
	
 4 GB
	
 2–2.5 GB

Mistral 7B Instruct	
  7B
	
 13–15 GB
	
 8 GB
	
 4–5 GB

Qwen 7B Instruct	
  7B
	
 13–15 GB
	
 8 GB
	
 4–5 GB
Downloading Models:
Run the following command to download required models locally:
python download.py
Hugging Face Token (Required for Google T5 Models)
For downloading Google T5 models, set your Hugging Face token:
set HF_TOKEN=your_huggingface_token   # Windows
export HF_TOKEN=your_huggingface_token # Linux / macOS
Running the Application:
Start the Streamlit app using:
streamlit run main.py
Usage Steps:
Upload one or more PDF documents
Click Process PDFs
Ask questions related to the uploaded content
View generated answers grounded in the PDFs
Expected Output:
(Insert your expected-answer image here)
Notes:
Designed for local inference
Optimized for INT4 quantized LLMs on limited VRAM GPUs
Responses are strictly restricted to provided document context
Limitations:
Responses are restricted strictly to the content of uploaded PDFs
Large or complex documents may increase processing time
Answer quality depends on document clarity and structure
