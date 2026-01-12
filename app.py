import streamlit as st
import time
from src.rag_pipeline import get_rag_chain, get_retriever

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="CrediTrust Analyst",
    page_icon="🏦",
    layout="wide"
)

# --- 2. SIDEBAR (Context for the User) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4121/4121044.png", width=80)
    st.title("CrediTrust Financial")
    st.markdown("### Internal Complaint Analyst")
    st.info(
        """
        **User:** Asha (Product Manager)
        **Goal:** Analyze customer friction points.
        **Data Source:** CFPB Complaint Database (Real-time).
        """
    )
    st.divider()
    st.markdown("### Suggested Queries:")
    st.code("Why are customers complaining about overdraft fees?", language=None)
    st.code("What are the main issues with Student Loans?", language=None)
    st.code("Are there delays in Money Transfers?", language=None)

# --- 3. MAIN UI & LOGIC ---
st.title("🏦 Customer Insight Dashboard")
st.markdown("Welcome, Asha. Ask questions to analyze the latest complaint trends.")

# Cache the heavy model loading
@st.cache_resource
def load_system():
    # Load the RAG Chain and the Retriever
    return get_rag_chain(), get_retriever()

# Load the model with a spinner
with st.spinner("Initializing AI Brain & Loading Database..."):
    try:
        chain, retriever = load_system()
    except Exception as e:
        st.error(f"❌ System Error: {e}")
        st.stop()

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I have analyzed the latest complaint logs. What would you like to investigate today?"}
    ]

# Display History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- 4. CHAT INPUT & RESPONSE ---
if prompt := st.chat_input("Type your question here..."):
    
    # 1. User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Assistant Message
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        with st.spinner("Retrieving relevant complaints..."):
            try:
                # A. Retrieve Context (The "Retrieval" part of RAG)
                docs = retriever.invoke(prompt)
                
                # B. Generate Answer (The "Generation" part of RAG)
                response_text = chain.invoke(prompt)
                
                # C. Streaming Output Effect
                for chunk in response_text.split():
                    full_response += chunk + " "
                    time.sleep(0.05) # Adjust speed of typing here
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)

                # D. Show Evidence (Sources)
                with st.expander("📄 View Source Evidence (Retrieval Context)"):
                    if not docs:
                        st.warning("No relevant complaints found in the database.")
                    for i, doc in enumerate(docs):
                        st.markdown(f"**Complaint #{i+1} ({doc.metadata.get('product', 'Unknown Product')}):**")
                        st.caption(f"_{doc.page_content}_")
                        st.divider()

                # Save history
                st.session_state.messages.append({"role": "assistant", "content": full_response})

            except Exception as e:
                st.error(f"Error: {e}")