import uuid

import streamlit as st

from utils.api import delete_session_api, get_session_api, list_sessions_api


def _ensure_session_id() -> str:
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = uuid.uuid4().hex
    return st.session_state["session_id"]


def render_sessions_panel():
    st.sidebar.divider()
    st.sidebar.header("💬 Conversations")

    _ensure_session_id()

    if st.sidebar.button("➕ New chat", use_container_width=True):
        st.session_state["session_id"] = uuid.uuid4().hex
        st.session_state["messages"] = []
        st.session_state["last_references"] = []
        st.rerun()

    sessions = list_sessions_api()
    if not sessions:
        st.sidebar.caption("No saved chats yet.")
        return

    for s in sessions[:25]:
        sid = s.get("session_id")
        title = s.get("title") or "(untitled)"
        cols = st.sidebar.columns([5, 1])
        if cols[0].button(
            f"📝 {title[:36]}",
            key=f"load_{sid}",
            use_container_width=True,
            help=f"{s.get('messageCount',0)} messages · {s.get('updatedAt','')}",
        ):
            data = get_session_api(sid) or {}
            st.session_state["session_id"] = sid
            st.session_state["messages"] = data.get("messages") or []
            last_assistant = next(
                (m for m in reversed(st.session_state["messages"]) if m.get("role") == "assistant"),
                None,
            )
            st.session_state["last_references"] = (
                (last_assistant or {}).get("references") or []
            )
            st.rerun()
        if cols[1].button("🗑", key=f"del_{sid}", help="Delete chat"):
            delete_session_api(sid)
            if st.session_state.get("session_id") == sid:
                st.session_state["session_id"] = uuid.uuid4().hex
                st.session_state["messages"] = []
            st.rerun()
