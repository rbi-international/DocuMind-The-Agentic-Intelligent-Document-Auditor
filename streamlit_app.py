import streamlit as st
import requests
import json

# Configuration
API_URL = "http://localhost:8000/audit"
st.set_page_config(page_title="DocuMind Enterprise", layout="wide")

# Custom CSS for Professional Look
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
    .stTextArea textarea {
        background-color: #ffffff;
        color: #333333;
    }
    .stButton>button {
        background-color: #0066cc;
        color: white;
        border-radius: 5px;
        width: 100%;
    }
    .success-box {
        padding: 20px;
        background-color: #e6fffa;
        border-left: 5px solid #00cc99;
        border-radius: 5px;
        color: #333333;
    }
    .header-style {
        font-size: 24px;
        font-weight: bold;
        color: #003366;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="header-style">DocuMind: Agentic Legal Auditor</p>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("System Status")
    try:
        # Ping the root endpoint to check if backend is alive
        status = requests.get("http://localhost:8000/")
        if status.status_code == 200:
            st.success("🟢 AI Engine Online")
            st.info(f"Backend: FastAPI\nModel: Qwen2.5-3B + DistilBERT")
        else:
            st.error("🔴 AI Engine Error")
    except:
        st.error("🔴 Backend Offline")
        st.warning("Please run 'python app.py' in a separate terminal.")

    st.markdown("---")
    st.write("### Instructions")
    st.write("1. Paste a legal clause into the text area.")
    st.write("2. Click 'Audit Document'.")
    st.write("3. The Agent will classify and analyze risks.")

# Main Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📄 Input Document")
    doc_text = st.text_area(
        "Paste Legal Text Here:", 
        height=400, 
        placeholder="Example: This Agreement shall be governed by the laws of the State of California..."
    )
    
    audit_btn = st.button("🔍 Run Audit Agent")

with col2:
    st.subheader("🤖 Agent Analysis")
    
    if audit_btn:
        if not doc_text or len(doc_text) < 10:
            st.warning("Please enter valid text (min 10 characters).")
        else:
            with st.spinner("Agent is analyzing... (This uses GPU)"):
                try:
                    # Call the FastAPI Backend
                    payload = {"text": doc_text}
                    response = requests.post(API_URL, json=payload)
                    
                    if response.status_code == 200:
                        data = response.json()
                        result = data["raw_agent_output"]
                        
                        # Display Result nicely
                        st.markdown(f"""
                        <div class="success-box">
                            {result}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Show raw JSON for debugging (optional)
                        with st.expander("View System Logs"):
                            st.json(data)
                            
                    else:
                        st.error(f"Server Error: {response.status_code}")
                        st.write(response.text)
                        
                except requests.exceptions.ConnectionError:
                    st.error("Failed to connect to Backend. Is 'python app.py' running?")
                except Exception as e:
                    st.error(f"An error occurred: {e}")

# Footer
st.markdown("---")
st.markdown("© 2025 DocuMind Enterprise AI | Powered by Qwen & LangGraph")