import streamlit as st
import requests

st.title("Chat with RAG-based AI Agent")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["text"])

# Handle new input
query = st.chat_input("Ask something...")

if query:
    # Show user message
    with st.chat_message("user"):
        st.write(query)
    st.session_state.messages.append({"role": "user", "text": query})

    # Stream assistant response
    with st.chat_message("assistant"):
        response = requests.post(
            "http://127.0.0.1:8000/chat",
            json={"query": query, "user_id": "user1"},
            stream=True
        )

        def stream():
            for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                if chunk:  # avoid empty chunks
                    yield chunk

        # Stream text live
        full_text = st.write_stream(stream)

        # Re-render clean markdown at the end
        st.markdown(full_text)

    st.session_state.messages.append({"role": "assistant", "text": full_text})