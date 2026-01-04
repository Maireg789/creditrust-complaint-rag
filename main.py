# main.py
import pandas as pd
import os
import sys

# Add 'src' to python path to ensure imports work
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from ingestion import perform_stratified_split, chunk_documents
from rag_engine import generate_answer_safe

def create_dummy_data():
    """Creates a sample CSV file for testing purposes."""
    if not os.path.exists("data"):
        os.makedirs("data")
    
    data = {
        'text': [
            "Standard Loan Policy: Minimum credit score is 600.",
            "Standard Loan Policy: Max DTI is 45%.",
            "High Risk Policy: Minimum credit score is 700.",
            "High Risk Policy: Requires manual underwriting.",
            "Jumbo Loan Policy: Minimum loan amount is $700k.",
        ] * 20, # Repeat to simulate dataset
        'risk_category': ['Standard', 'Standard', 'High Risk', 'High Risk', 'Jumbo'] * 20
    }
    df = pd.DataFrame(data)
    df.to_csv("data/credit_policies.csv", index=False)
    print("Created dummy data at data/credit_policies.csv")
    return df

def run_ingestion_pipeline():
    print("--- STARTING INGESTION ---")
    
    # 1. Load Data
    csv_path = "data/credit_policies.csv"
    if not os.path.exists(csv_path):
        print("Data file not found. Creating dummy data...")
        df = create_dummy_data()
    else:
        df = pd.read_csv(csv_path)

    # 2. Stratified Sampling (Your Task 1)
    # We use 'risk_category' to ensure we don't lose the rare "Jumbo" or "High Risk" policies
    kb_df, eval_df = perform_stratified_split(df, stratify_col='risk_category', test_size=0.2)
    print(f"Split complete. Training Docs: {len(kb_df)}, Eval Docs: {len(eval_df)}")

    # 3. Explicit Chunking (Your Task 2)
    # Convert DataFrame to a format LangChain likes (list of Document objects)
    from langchain_core.documents import Document
    documents = [
        Document(page_content=row['text'], metadata={"category": row['risk_category']}) 
        for _, row in kb_df.iterrows()
    ]
    
    chunks = chunk_documents(documents)
    print(f"Chunking complete. Created {len(chunks)} chunks using explicit separators.")
    print("--- INGESTION FINISHED ---\n")

def run_rag_test():
    print("--- TESTING RAG ENGINE ---")
    # This is a mock test to see if the imports work. 
    # Real generation requires an OpenAI Key.
    try:
        print("RAG Engine imports are successful.")
        print("To run a real query, ensure OPENAI_API_KEY is set in your environment.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    run_ingestion_pipeline()
    run_rag_test()