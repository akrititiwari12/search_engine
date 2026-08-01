import streamlit as st
from dotenv import load_dotenv
import os

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

# -----------------------------
# Load Environment Variables
# -----------------------------
load_dotenv()

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="AI Chat", page_icon="🤖")

st.title("🤖 AI Chat with Groq")

api_key = st.sidebar.text_input(
    "Enter your Groq API Key",
    type="password"
)

if not api_key:
    st.info("Please enter your Groq API Key in the sidebar.")
    st.stop()

# -----------------------------
# Session State
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -----------------------------
# User Input
# -----------------------------
prompt = st.chat_input("Ask me anything...")

if prompt:

    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=0,
    )

    response = llm.invoke(
        [HumanMessage(content=prompt)]
    )

    answer = response.content

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )