import pytest
import pandas as pd
import sys
import os
from langchain_core.documents import Document

# Fix path to allow importing from src
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from ingestion import perform_stratified_split, chunk_documents

def test_stratified_sampling_logic():
    """Test 1: Ensure the split maintains the right ratios."""
    # Create dummy data (100 rows: 80 'Standard', 20 'High Risk')
    data = {
        'text': ['policy doc'] * 100,
        'risk_category': ['Standard'] * 80 + ['High Risk'] * 20
    }
    df = pd.DataFrame(data)
    
    # Run the split
    train, test = perform_stratified_split(df, stratify_col='risk_category', test_size=0.2)
    
    # Check sizes (80/20 split of 100 rows -> 80 train, 20 test)
    assert len(train) == 80
    assert len(test) == 20
    
    # Check Stratification:
    # Test set should have roughly 20% of the 'High Risk' items (which is 4 items)
    high_risk_in_test = test[test['risk_category'] == 'High Risk'].shape[0]
    assert high_risk_in_test == 4

def test_chunking_constraints():
    """Test 2: Ensure chunks don't exceed the max size."""
    # Create a very long text document
    long_text = "word " * 500  # Approx 2500 characters
    doc = Document(page_content=long_text, metadata={"source": "test"})
    
    # Run chunking
    chunks = chunk_documents([doc])
    
    # Assertions
    assert len(chunks) > 1  # Should be split into multiple parts
    for chunk in chunks:
        assert len(chunk.page_content) <= 1200 # Max size + overlap buffer