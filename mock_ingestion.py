# mock_ingestion.py
import os
import shutil
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

def create_mock_database():
    # Clear old db
    if os.path.exists("./chroma_db"):
        shutil.rmtree("./chroma_db")

    # Fake CrediTrust Data
    complaints = [
        "Credit Card: I was charged a late fee even though I paid on time. This is unfair.",
        "Credit Card: The interest rate jumped to 25% without notice.",
        "Money Transfer: My transfer to Kenya is stuck pending for 5 days.",
        "Money Transfer: Hidden exchange rate fees are too high.",
        "Savings: The advertised APY was 2% but I only got 0.5%.",
        "Personal Loan: Prepayment penalty was not disclosed.",
        "Service: Chatbot is useless, I need a human agent."
    ]

    print("Creating vector database... this creates the 'Brain' of the AI...")
    docs = [Document(page_content=t) for t in complaints]
    
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
        persist_directory="./chroma_db"
    )
    print("✅ Database created!")

if __name__ == "__main__":
    create_mock_database()