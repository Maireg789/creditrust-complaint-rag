import pandas as pd
from src.rag_pipeline import get_rag_chain

def run_evaluation():
    print("--- STARTING RAG EVALUATION (Free Mode) ---")
    
    # 1. Initialize the Chain
    try:
        chain = get_rag_chain()
    except Exception as e:
        print(f"Error initializing RAG Chain: {e}")
        return

    # 2. Define the Test Questions
    test_questions = [
        "What are the complaints about Credit Cards?",
        "Why are Money Transfers being delayed?",
        "What is the issue with Savings Accounts?",
        "Do customers like the Customer Service?",
        "What happens if I pay off my loan early?"
    ]

    results = []

    # 3. Ask the Questions
    print("\nThinking... (This may take a moment on your CPU)\n")
    
    for q in test_questions:
        print(f"Asking: {q}...")
        try:
            # Run the RAG pipeline
            answer = chain.invoke(q)
            
            # Save result
            results.append({
                "Question": q,
                "Generated Answer": answer
            })
            print(f"Answer: {answer}\n")
            
        except Exception as e:
            print(f"Failed: {e}")

    # 4. Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv("rag_evaluation_results.csv", index=False)
    print("\n✅ Evaluation Complete. Results saved to 'rag_evaluation_results.csv'")

if __name__ == "__main__":
    run_evaluation()