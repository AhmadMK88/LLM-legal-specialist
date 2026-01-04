import streamlit as st
import requests
from config.configs import API_URL, API_HEADERS
from langdetect import detect

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Legal AI Assistant",
    page_icon="⚖️",
    layout="wide"
)

# -----------------------------
# Helper: RTL-aware renderer
# -----------------------------
def render_message(text: str, role: str):
    try:
        lang = detect(text)
    except:
        lang = "en"

    if lang == "ar":
        st.markdown(
            f'<div dir="rtl" style="text-align:right; white-space:pre-line; margin:0;">{text}</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(text)

# -----------------------------
# Session state
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("⚖️ Legal AI Specialist")
st.sidebar.write("Ask legal questions with an LLM backend.")

if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()

# -----------------------------
# Display chat
# -----------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        render_message(msg["content"], msg["role"])

# -----------------------------
# User input
# -----------------------------
user_input = st.chat_input("Ask your legal question...")

if user_input:
    # Show user message immediately
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )
    st.session_state.pending_prompt = user_input
    st.rerun()

# -----------------------------
# Backend call
# -----------------------------
if st.session_state.pending_prompt:
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                res = requests.post(
                    API_URL,
                    headers=API_HEADERS,
                    json={"prompt": st.session_state.pending_prompt},
                    timeout=120
                )
                response_text = res.json().get(
                    "response", "No response from API"
                )
            except Exception as e:
                response_text = f"Error connecting to API:\n{e}"

    st.session_state.messages.append(
        {"role": "assistant", "content": response_text}
    )
    st.session_state.pending_prompt = None
    st.rerun()
