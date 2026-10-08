"""A feature-rich Streamlit chat app using the official Groq Python SDK."""

import json
import os
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
from groq import Groq


load_dotenv()

st.set_page_config(
    page_title="Groq Chat Studio",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODELS = {
    "GPT-OSS 20B — fast": "openai/gpt-oss-20b",
    "Llama 3.3 70B — balanced": "llama-3.3-70b-versatile",
}

CHAT_MODES = {
    "General assistant": (
        "You are a helpful, accurate, and friendly AI assistant. "
        "Use clear Markdown and say when you are uncertain."
    ),
    "Programming helper": (
        "You are a senior software engineer. Explain code clearly, provide "
        "safe and runnable examples, and call out assumptions or trade-offs."
    ),
    "Writing assistant": (
        "You are an expert writing partner. Improve clarity and structure "
        "while preserving the user's intended tone and meaning."
    ),
    "Study coach": (
        "You are a patient study coach. Teach step by step, use short examples, "
        "and ask a brief check-in question after explaining a difficult concept."
    ),
}

SUPPORTED_FILE_TYPES = ["txt", "md", "py", "js", "ts", "html", "css", "json", "csv"]
MAX_FILE_CHARS = 12_000
MAX_HISTORY_MESSAGES = 16


def now() -> str:
    """Return a friendly local timestamp for a message."""
    return datetime.now().strftime("%d %b %Y, %I:%M %p")


def initialize_session() -> None:
    """Create all app state once, before widgets render."""
    defaults = {
        "messages": [],
        "conversation_name": "New conversation",
        "feedback": {"helpful": 0, "not_helpful": 0},
        "custom_instructions": "",
    }
    for name, value in defaults.items():
        if name not in st.session_state:
            st.session_state[name] = value


def clear_chat() -> None:
    st.session_state.messages = []
    st.session_state.conversation_name = "New conversation"
    st.session_state.feedback = {"helpful": 0, "not_helpful": 0}


def get_api_key() -> str:
    """Use .env first; the sidebar can temporarily override it."""
    return st.session_state.get("api_key_input", "") or os.getenv("GROQ_API_KEY", "")


def file_to_context(uploaded_file) -> tuple[str, str]:
    """Read a small text-based file and prepare it for the next question."""
    raw_text = uploaded_file.getvalue().decode("utf-8", errors="replace")
    was_trimmed = len(raw_text) > MAX_FILE_CHARS
    content = raw_text[:MAX_FILE_CHARS]
    note = f"Attached file: {uploaded_file.name}"
    if was_trimmed:
        note += f" (first {MAX_FILE_CHARS:,} characters used)"
    return note, content


def message_for_api(message: dict) -> dict:
    """Remove local display-only fields before a message is sent to Groq."""
    return {"role": message["role"], "content": message["content"]}


def build_api_messages(system_prompt: str) -> list[dict]:
    """Keep the recent conversation so long chats do not grow without limit."""
    recent_messages = st.session_state.messages[-MAX_HISTORY_MESSAGES:]
    history = [message_for_api(message) for message in recent_messages]
    return [{"role": "system", "content": system_prompt}, *history]


def make_system_prompt(mode: str, custom_instructions: str) -> str:
    prompt = CHAT_MODES[mode]
    if custom_instructions.strip():
        prompt += f"\n\nAdditional user preferences:\n{custom_instructions.strip()}"
    return prompt


def markdown_export() -> str:
    """Make a readable version of the conversation for download."""
    lines = [f"# {st.session_state.conversation_name}", ""]
    for message in st.session_state.messages:
        speaker = "You" if message["role"] == "user" else "Assistant"
        display_text = message.get("display_content", message["content"])
        lines.extend([f"## {speaker}", display_text, ""])
    return "\n".join(lines)


def json_export() -> str:
    export = {
        "title": st.session_state.conversation_name,
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "messages": st.session_state.messages,
    }
    return json.dumps(export, ensure_ascii=False, indent=2)


def render_chat_history() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message.get("display_content", message["content"]))
            timestamp = message.get("timestamp")
            if timestamp:
                st.caption(timestamp)


def stream_answer(client: Groq, request_messages: list[dict], settings: dict) -> str:
    """Stream an answer to the screen and return the completed text."""
    response_stream = client.chat.completions.create(
        model=settings["model"],
        messages=request_messages,
        temperature=settings["temperature"],
        top_p=settings["top_p"],
        max_completion_tokens=settings["max_tokens"],
        stream=True,
    )

    answer = ""
    placeholder = st.empty()
    for chunk in response_stream:
        if not chunk.choices:
            continue
        text = chunk.choices[0].delta.content or ""
        answer += text
        placeholder.markdown(answer + "▌")

    placeholder.markdown(answer)
    return answer


initialize_session()

st.markdown(
    """
    <style>
        .block-container {max-width: 1100px; padding-top: 2rem;}
        [data-testid="stChatMessage"] {border: 1px solid rgba(128,128,128,.14); border-radius: 14px; padding: .5rem .8rem;}
        [data-testid="stSidebar"] {border-right: 1px solid rgba(128,128,128,.15);}
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.title("⚙️ Chat Studio")
    st.caption("Your settings apply to the next response.")

    st.text_input(
        "Groq API key",
        value=os.getenv("GROQ_API_KEY", ""),
        type="password",
        key="api_key_input",
        help="Stored only in this browser session. You can also set GROQ_API_KEY in .env.",
    )

    st.divider()
    st.subheader("Assistant")
    chat_mode = st.selectbox("Chat mode", options=list(CHAT_MODES))
    selected_model = st.selectbox("Model", options=list(MODELS))
    temperature = st.slider("Creativity", 0.0, 1.0, 0.2, 0.1)

    with st.expander("Advanced response settings"):
        top_p = st.slider("Response diversity", 0.1, 1.0, 1.0, 0.1)
        max_tokens = st.slider("Maximum answer length", 128, 2048, 1024, 128)
        st.caption("A smaller length makes responses faster and shorter.")

    st.text_area(
        "Custom instructions",
        placeholder="Example: Use simple English and short bullet points.",
        key="custom_instructions",
        height=105,
    )

    st.divider()
    st.subheader("Conversation")
    st.text_input("Chat title", key="conversation_name")
    st.button("🗑️ Start new chat", use_container_width=True, on_click=clear_chat)

    st.download_button(
        "⬇️ Download as Markdown",
        data=markdown_export(),
        file_name="groq-chat.md",
        mime="text/markdown",
        disabled=not st.session_state.messages,
        use_container_width=True,
    )
    st.download_button(
        "⬇️ Download as JSON",
        data=json_export(),
        file_name="groq-chat.json",
        mime="application/json",
        disabled=not st.session_state.messages,
        use_container_width=True,
    )

    st.divider()
    st.caption("Never share or commit your `.env` file. Keep it listed in `.gitignore`.")

api_key = get_api_key()
settings = {
    "model": MODELS[selected_model],
    "temperature": temperature,
    "top_p": top_p,
    "max_tokens": max_tokens,
}

st.title("🤖 Groq Chat Studio")
st.caption("A fast AI workspace with streaming, context files, and exports.")

chat_count = len(st.session_state.messages)
user_count = sum(message["role"] == "user" for message in st.session_state.messages)
assistant_count = sum(message["role"] == "assistant" for message in st.session_state.messages)
metric_1, metric_2, metric_3 = st.columns(3)
metric_1.metric("Messages", chat_count)
metric_2.metric("Your questions", user_count)
metric_3.metric("Assistant answers", assistant_count)

if not api_key:
    st.info("Add your Groq API key in the sidebar or in your `.env` file to begin.", icon="🔑")
    st.stop()

render_chat_history()

uploaded_file = st.file_uploader(
    "Optional: attach a text or code file for the next question",
    type=SUPPORTED_FILE_TYPES,
    help=f"The first {MAX_FILE_CHARS:,} characters are included in the next message.",
)

with st.expander("Need an idea?", expanded=not st.session_state.messages):
    st.write("Try one of these prompts:")
    st.code("Explain this Python error in simple words.", language=None)
    st.code("Create a seven-day study plan for learning SQL.", language=None)
    st.code("Review this code and suggest improvements.", language=None)

prompt = st.chat_input("Message the assistant...")

if prompt:
    display_content = prompt
    api_content = prompt

    if uploaded_file:
        file_note, file_content = file_to_context(uploaded_file)
        display_content = f"{prompt}\n\n📎 {file_note}"
        api_content = (
            f"{prompt}\n\n"
            f"Here is the content of `{uploaded_file.name}`:\n"
            f"```\n{file_content}\n```"
        )

    user_message = {
        "role": "user",
        "content": api_content,
        "display_content": display_content,
        "timestamp": now(),
    }
    st.session_state.messages.append(user_message)

    with st.chat_message("user"):
        st.markdown(display_content)
        st.caption(user_message["timestamp"])

    system_prompt = make_system_prompt(chat_mode, st.session_state.custom_instructions)
    request_messages = build_api_messages(system_prompt)

    with st.chat_message("assistant"):
        try:
            groq_client = Groq(api_key=api_key)
            answer = stream_answer(groq_client, request_messages, settings)

            if not answer.strip():
                answer = "I did not receive a response. Please try again."
                st.warning(answer)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "display_content": answer,
                    "timestamp": now(),
                }
            )
        except Exception as error:
            st.error("I could not reach Groq. Check your API key, model, and internet connection.")
            with st.expander("Show technical details"):
                st.code(str(error))

if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
    st.divider()
    st.caption("Was the last answer useful?")
    helpful, not_helpful, feedback_total = st.columns([1, 1, 4])
    if helpful.button("👍 Helpful", use_container_width=True):
        st.session_state.feedback["helpful"] += 1
        st.toast("Thanks for the feedback!")
    if not_helpful.button("👎 Improve", use_container_width=True):
        st.session_state.feedback["not_helpful"] += 1
        st.toast("Thanks — try adding more detail to your next question.")
    feedback_total.caption(
        f"Feedback this session: {st.session_state.feedback['helpful']} helpful, "
        f"{st.session_state.feedback['not_helpful']} to improve"
    )
