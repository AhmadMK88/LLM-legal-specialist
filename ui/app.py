import streamlit as st
import requests
from config.configs import API_URL, API_HEADERS

# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(page_title="Legal AI Assistant", page_icon="⚖️", layout="wide")

st.markdown("""
    <style>
    .chat-message {
        padding: 1rem;
        border-radius: 1rem;
        margin-bottom: 1rem;
        max-width: 80%;
    }
    .user-message {
        background-color: #343541; color: #FFFFFF;
        margin-left: auto;
    }
    .bot-message {
        background-color: #444654; color: #FFFFFF;
        margin-right: auto;
    }
    .chat-container {
        max-height: 70vh;
        overflow-y: auto;
        padding-right: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# Session State
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "loading" not in st.session_state:
    st.session_state.loading = False

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("⚖️ Legal AI Specialist")
st.sidebar.write("Ask legal questions with an LLM backend.")
if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []

# -----------------------------
# Chat Display
# -----------------------------
st.title("AI Legal Assistant Chat")
chat_container = st.container()

def render_chat():
    with chat_container:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for msg in st.session_state.messages:
            css_class = "user-message" if msg["role"] == "user" else "bot-message"
            content = msg["content"].replace("\n", "<br>") if msg["role"] == "assistant" else msg["content"]
            st.markdown(f'<div class="chat-message {css_class}">{content}</div>', unsafe_allow_html=True)

        # Show loading bubble if waiting for assistant
        if st.session_state.loading:
            st.markdown('<div class="chat-message bot-message">Typing...</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

render_chat()

# -----------------------------
# Input Box
# -----------------------------
user_input = st.chat_input("Ask your legal question...")

if user_input:
    # Add user message immediately
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.loading = True
    render_chat()  # Show user message + loading bubble

    # Make API request
    payload = {"prompt": user_input}
    try:
        res = requests.post(API_URL, headers=API_HEADERS, json=payload)
        response_text = res.json().get("response", "Error: No response from API")
    except Exception as e:
        response_text = f"Error connecting to API: {e}"

    # Add assistant response and remove loading
    st.session_state.loading = False
    st.session_state.messages.append({"role": "assistant", "content": response_text})
    render_chat()
    st.rerun()
