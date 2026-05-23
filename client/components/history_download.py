import json
import streamlit as st


def render_history_download():
    messages = st.session_state.get("messages") or []
    if not messages:
        return
    st.sidebar.divider()
    txt = "\n\n".join(
        f"{m.get('role','user').upper()}: {m.get('content','')}" for m in messages
    )
    st.sidebar.download_button(
        "⬇️ Download chat (.txt)",
        data=txt,
        file_name="chat_history.txt",
        mime="text/plain",
        use_container_width=True,
    )
    st.sidebar.download_button(
        "⬇️ Download chat (.json)",
        data=json.dumps(messages, ensure_ascii=False, indent=2),
        file_name="chat_history.json",
        mime="application/json",
        use_container_width=True,
    )
