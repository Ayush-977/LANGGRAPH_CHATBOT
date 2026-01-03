import streamlit as st
# Import clear_database from backend
from langgraph_backend import chatbot, retrive_all_threads, clear_database
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
import uuid

# --- 1. Helper Functions ---

def generate_title(first_message_content):
    try:
        llm = ChatGroq(model="llama-3.1-8b-instant")
        messages = [
            SystemMessage(content="Generate a very short title (max 4 words) for this chat based on the user's overall message. Do not use quotes."),
            HumanMessage(content=first_message_content)
        ]
        response = llm.invoke(messages)
        title = response.content.strip()
        if len(title) > 50:
            return title[:47] + "..."
        return title
    except Exception:
        return "New Conversation"

def generate_thread_id():
    return str(uuid.uuid4())

def reset_chat():
    new_id = generate_thread_id()
    st.session_state['thread_id'] = new_id
    add_thread(new_id)
    st.session_state['thread_names'][new_id] = "New Chat..."
    st.session_state['message_history'] = []

def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)

def load_convo(thread_id):
    config = {"configurable": {"thread_id": thread_id}}
    state = chatbot.get_state(config)
    return state.values.get("messages", [])

# --- 2. Session State Initialization ---

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = retrive_all_threads()

if 'thread_names' not in st.session_state:
    st.session_state['thread_names'] = {}

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

# Ensure current thread is tracked
add_thread(st.session_state['thread_id'])
if st.session_state['thread_id'] not in st.session_state['thread_names']:
    st.session_state['thread_names'][st.session_state['thread_id']] = "New Chat..."

CONFIG = {"configurable": {"thread_id": st.session_state["thread_id"]}}

# --- 3. Sidebar (Menu Logic) ---

with st.sidebar:
    st.title("LangGraph Chat")
    
    if st.button("➕ New Chat", use_container_width=True):
        reset_chat()
        st.rerun()
    
    st.markdown("---")
    st.header("History")

    if st.session_state['chat_threads']:
        options = st.session_state['chat_threads'][::-1]
        try:
            curr_idx = options.index(st.session_state['thread_id'])
        except ValueError:
            curr_idx = 0

        selected_id = st.radio(
            "Select Chat",
            options=options,
            format_func=lambda x: (st.session_state['thread_names'].get(x, "New Chat")[:30] + '...') if len(st.session_state['thread_names'].get(x, "New Chat")) > 30 else st.session_state['thread_names'].get(x, "New Chat"),
            index=curr_idx,
            label_visibility="collapsed"
        )

        if selected_id != st.session_state['thread_id']:
            st.session_state['thread_id'] = selected_id
            messages = load_convo(selected_id)
            temp_msgs = []
            for msg in messages:
                role = 'user' if isinstance(msg, HumanMessage) else 'assistant'
                temp_msgs.append({'role': role, 'content': msg.content})
            st.session_state['message_history'] = temp_msgs
            st.rerun()

    # --- STYLE INJECTION: Red Button ---
    st.markdown("""
        <style>
        div.stButton > button:first-child {
            width: 100%;
        }
        div.stButton > button:last-child {
            background-color: #ff4b4b;
            color: white;
            border: none;
        }
        div.stButton > button:last-child:hover {
            background-color: #ff0000;
            color: white;
        }
        </style>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    
    # 5. Clear History Button
    if st.button("⚠️ Clear History"):
        # 1. Wipe DB
        clear_database()
        # 2. Wipe RAM
        st.session_state.clear()
        # 3. Reload
        st.rerun()

# --- 4. Main Chat Area ---

for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

user_input = st.chat_input("Type here...")

if user_input:
    # A. Generate Title (Only for first message)
    if len(st.session_state['message_history']) == 0:
        current_id = st.session_state['thread_id']
        new_name = generate_title(user_input)
        st.session_state['thread_names'][current_id] = new_name
    
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.markdown(user_input)

    with st.chat_message('assistant'):
        def stream_generator():
            for chunk, meta in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages"
            ):
                if chunk.content:
                    yield chunk.content
        
        ai_msg = st.write_stream(stream_generator)

    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_msg})

    # Force Refresh to update sidebar title
    if len(st.session_state['message_history']) <= 2:
        st.rerun()