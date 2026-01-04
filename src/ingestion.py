# src/ingestion.py
import pandas as pd
from sklearn.model_selection import train_test_split
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- CODE 1: STRATIFIED SAMPLING ---
def perform_stratified_split(df, stratify_col='risk_category', test_size=0.2):
    """
    Splits data while keeping the ratio of risk categories consistent.
    """
    print(f"Splitting data with stratification on {stratify_col}...")
    
    # Stratified split
    train_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        stratify=df[stratify_col], 
        random_state=42
    )
    return train_df, test_df

# --- CODE 2: EXPLICIT CHUNKING ---
def get_credit_text_splitter():
    """
    Configures chunking specifically for credit policy documents.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        # Keep paragraphs together, then lines, then words
        separators=["\n\n", "\n", " ", ""], 
        length_function=len
    )

def chunk_documents(documents):
    splitter = get_credit_text_splitter()
    chunks = splitter.split_documents(documents)
    return chunks