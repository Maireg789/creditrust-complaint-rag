# src/rag_pipeline.py
import os
import dotenv
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load env (not strictly needed for local, but good practice)
dotenv.load_dotenv()

# Configuration
VECTOR_STORE_PATH = "./chroma_db"

# 1. SETUP EMBEDDINGS (Free & Local)
EMBEDDING_MODEL = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def get_retriever():
    if not os.path.exists(VECTOR_STORE_PATH):
        print(f"⚠️ Vector store not found at {VECTOR_STORE_PATH}.")
        return None
        
    vectorstore = Chroma(
        persist_directory=VECTOR_STORE_PATH, 
        embedding_function=EMBEDDING_MODEL
    )
    return vectorstore.as_retriever(search_kwargs={"k": 3})

def get_rag_chain():
    # 2. SETUP LLM (Free & Local - Google Flan-T5)
    # The first time you run this, it will download ~900MB. That is normal.
    print("Loading local AI model (Flan-T5)... please wait...")
    
    llm = HuggingFacePipeline.from_model_id(
        model_id="google/flan-t5-base",
        task="text2text-generation",
        pipeline_kwargs={
            "max_new_tokens": 200,  # How long the answer can be
            "temperature": 0.1      # Keep it factual
        }
    )

    # 3. SETUP PROMPT (Simpler prompt for smaller models)
    template = """
    Use the context below to answer the question. If you don't know, say "I don't know".
    
    Context: {context}
    
    Question: {question}
    
    Answer:
    """
    prompt = PromptTemplate.from_template(template)

    # 4. BUILD CHAIN
    retriever = get_retriever()
    
    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain