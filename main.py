import os
import debugpy

# debugpy.listen(5678)
# print("Waiting for debugger attach...")
# debugpy.wait_for_client()

import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig 
import torch
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from Template import css, bot_template, user_template


prompt_template = """
You are an expert answer assistant.

Your task:
- Answer ONLY using the information inside the <context> section.
- If the context does NOT contain the answer, reply exactly: “Not in the provided text.”
- Give the answer clearly and stop.
- Never add anything that is not supported by the context.
- No assumptions.
- No hallucinations.
- No extra sentences.
- No explanations about what you are doing.

<context>
{context}
</context>

Question: {question}

Answer:


"""
PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template
)

CONDENSE_QUESTION_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="Rewrite this question clearly: {question}"
)
# Extract text from uploaded PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Split text into chunks
def get_text_chunks(text):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=100,
        length_function=len
    )
    return splitter.split_text(text)

# Build FAISS vectorstore
def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="Your Embedding Model Name Here",
    )
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)

# LLM
def get_conversation_chain(vectorstore):
    model_path = "Your Large Language Model Path Here"

    # 4-bit Quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True,  trust_remote_code=True)

    # Load model with FP16 and automatic device mapping
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        trust_remote_code=True,
        device_map={"": "cuda"},
    )

    # Text-generation pipeline with max_new_tokens instead of max_length
    pipe = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=120,
        temperature = 0.1,
        eos_token_id=tokenizer.eos_token_id,  # how many tokens the model can generate
        #top_p=0.9,
        #repetition_penalty=1.1
    )

    llm = HuggingFacePipeline(pipeline=pipe)
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    conversation_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 1}),
    memory=memory,
    return_source_documents=False,
    combine_docs_chain_kwargs={"prompt": PROMPT}
    )

    return conversation_chain

# 5. Handle user input
def handle_userinput(user_question):
    if st.session_state.conversation is None:
        st.warning("Please process PDFs first!")
        return

    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        template = user_template if i % 2 == 0 else bot_template
        st.write(template.replace("{{MSG}}", message.content), unsafe_allow_html=True)


# 6. Streamlit main app
def main():
    load_dotenv()   
    st.set_page_config(page_title="Chat with PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    # Initialize session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Sidebar: Upload PDFs
    with st.sidebar:
        st.title("Upload PDFs")
        pdf_docs = st.file_uploader("Select PDFs", accept_multiple_files=True, type="pdf")

        if st.button("Process PDFs", use_container_width=True):
            if pdf_docs:
                with st.spinner("Processing PDFs..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    vectorstore = get_vectorstore(text_chunks)
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                    st.success("PDFs processed successfully!")
            else:
                st.error("Please select at least one PDF")

    # Main chat area
    st.title("Chat with PDFs")

    for i, message in enumerate(st.session_state.chat_history):
        template = user_template if i % 2 == 0 else bot_template
        st.write(template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

    # Input box
    user_question = st.text_input("Ask a question:", placeholder="What would you like to know?")
    if user_question:
        handle_userinput(user_question)

if __name__ == "__main__":
    main()


















