import os
import shutil
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.model_selection import train_test_split

# --- CONFIGURATION ---
DATA_PATH = "data/raw/complaints.csv" # Make sure your file is here!
DB_PATH = "./chroma_db"
TARGET_PRODUCTS = [
    "Credit card",
    "Credit card or prepaid card", # Handling CFPB naming variations
    "Personal loan",
    "Student loan",
    "Money transfer, virtual currency, or money service",
    "Checking or savings account"
]
SAMPLE_SIZE = 12500 # Target size per assignment

def run_ingestion():
    print("--- STARTING REAL INGESTION PIPELINE ---")
    
    # 1. Load Data
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: File not found at {DATA_PATH}. Please move your CSV there.")
        return

    print("Loading CSV... (this might take a moment)...")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    print(f"Raw Data Loaded: {len(df)} rows")

    # 2. Filter & Clean
    # Filter for products we care about
    df = df[df['Product'].isin(TARGET_PRODUCTS)]
    
    # Drop rows with no narrative (Empty text)
    df = df.dropna(subset=['Consumer complaint narrative'])
    print(f"Filtered (Relevant Products + Has Text): {len(df)} rows")

    # 3. Stratified Sampling
    # We want ~12,500 rows. If we have less, take them all.
    if len(df) > SAMPLE_SIZE:
        print(f"Downsampling to {SAMPLE_SIZE} rows using Stratified Sampling...")
        # Stratify by 'Product' to keep ratios
        sampled_df, _ = train_test_split(
            df, 
            train_size=SAMPLE_SIZE, 
            stratify=df['Product'], 
            random_state=42
        )
    else:
        sampled_df = df
        print("Dataset smaller than target sample. Using all available data.")

    # 4. Prepare Documents
    print("Converting to Documents...")
    documents = []
    for _, row in sampled_df.iterrows():
        # Combine relevant metadata
        text = row['Consumer complaint narrative']
        meta = {
            "product": row['Product'],
            "complaint_id": row.get('Complaint ID', 'Unknown'),
            "issue": row.get('Issue', 'Unknown')
        }
        documents.append(Document(page_content=text, metadata=meta))

    # 5. Chunking (Task 2 Requirement)
    print("Chunking Documents...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # Defined in assignment
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks from {len(documents)} complaints.")

    # 6. Embed & Store
    # Clear old database to start fresh
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)
        
    print("Embedding and Indexing... (This will take 5-10 minutes on CPU)...")
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Process in batches to avoid crashing memory
    batch_size = 5000
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        print(f"Processing batch {i} to {i+len(batch)}...")
        Chroma.from_documents(
            documents=batch,
            embedding=embedding_model,
            persist_directory=DB_PATH
        )

    print(f"\n✅ INGESTION COMPLETE! Database saved to {DB_PATH}")

if __name__ == "__main__":
    run_ingestion()