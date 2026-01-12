# 🏦 Intelligent Complaint Analysis for Financial Services (CrediTrust)

**(Final Submission)**  
**Author:** Maireg Azanaw  
**Date:** January 13, 2026  
**Status:** Complete (Tasks 1, 2, 3 & 4)

---

## 📖 Project Overview
**CrediTrust Financial** receives thousands of customer complaints monthly. Manual analysis is inefficient and reactive. This project builds a **Retrieval-Augmented Generation (RAG)** system that transforms unstructured customer feedback into actionable insights.

Stakeholders like Asha (Product Manager) can now use a **Streamlit Dashboard** to ask natural language questions (e.g., *"Why are customers complaining about Money Transfers?"*) and receive evidence-based answers grounded in real CFPB data.

## 📂 Repository Structure
This project has evolved from experimental notebooks to a production-ready modular architecture:

```text
rag-complaint-chatbot/
├── data/                  # Raw CSV files (Gitignored for size)
├── src/                   # Core Logic Modules
│   ├── ingestion.py       # Task 1 & 2: Loading, Stratified Sampling, Chunking, Indexing
│   └── rag_pipeline.py    # Task 3: RAG Chain (Retriever + Generator)
├── docs/                  # Documentation & Images
│   ├── images/            # Screenshots for the report
│   └── scripts/           # Scripts used to generate charts
├── app.py                 # Task 4: Streamlit Dashboard (User Interface)
├── evaluate_rag.py        # Script to test the system against "Gold Standard" questions
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
🚀 How to Run the Project
1. Setup Environment
code
Bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/creditrust-complaint-rag.git

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate  # Windows

# Install dependencies
pip install -r requirements.txt
2. Run Data Ingestion
This script loads the complaints.csv, performs sampling, and builds the Vector Database.
(Note: Takes ~30 mins on CPU due to embedding generation)
code
Bash
python src/ingestion.py
Output: ✅ INGESTION COMPLETE! Database saved to ./chroma_db
3. Launch the Dashboard
Start the web interface to chat with the data.
code
Bash
streamlit run app.py
Access the app at: http://localhost:8501
4. Run Evaluation (Optional)
To test the RAG accuracy via the terminal:
code
Bash
python evaluate_rag.py
📸 Screenshots
The CrediTrust Analyst Dashboard
![CrediTrust AI Dashboard interface showing the chatbot answering a query about credit card complaints with cited source evidence]docs/streamlit_screenshot.png.png