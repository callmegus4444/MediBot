import streamlit as st
from utils.api import ask_question_strict


_STATUS_LABEL = {
    "answered": ("Answered", "#16a34a"),
    "partial": ("Partial", "#d97706"),
    "insufficient_evidence": ("No verified answer", "#6b7280"),
    "conflicting_evidence": ("Conflicting evidence", "#dc2626"),
}


def _inject_styles():
    st.markdown(
        """
<style>
.medi-status-pill {
    display:inline-block; padding:4px 10px; border-radius:999px;
    font-size:12px; font-weight:600; color:#fff;
}
.medi-confidence {
    display:inline-block; margin-left:8px; padding:4px 10px;
    border-radius:999px; font-size:12px; font-weight:600;
    background:#1f2937; color:#e5e7eb;
}
.medi-ref-card {
    border:1px solid #1f2937; background:#0f172a; border-radius:10px;
    padding:12px 14px; margin-bottom:10px;
}
.medi-ref-num {
    display:inline-block; min-width:22px; height:22px; line-height:22px;
    text-align:center; border-radius:11px; background:#1d4ed8; color:#fff;
    font-size:12px; font-weight:700; margin-right:8px;
}
.medi-ref-title { font-weight:600; color:#e5e7eb; line-height:1.35; }
.medi-ref-meta { color:#9ca3af; font-size:12px; margin:4px 0 8px; }
.medi-ref-link a { color:#60a5fa !important; font-weight:600; text-decoration:none; }
.medi-ref-link a:hover { text-decoration:underline; }
.medi-ref-snippet {
    color:#cbd5e1; font-size:12.5px; line-height:1.45; margin-top:6px;
    max-height:96px; overflow:hidden;
}
.medi-empty {
    color:#9ca3af; font-style:italic; padding:10px 4px;
}
</style>
""",
        unsafe_allow_html=True,
    )


def _render_reference_panel(container, references, empty_message):
    container.markdown("### Sources")
    if not references:
        container.markdown(
            f"<div class='medi-empty'>{empty_message}</div>", unsafe_allow_html=True
        )
        return
    for i, ref in enumerate(references, start=1):
        title = ref.get("title") or "(untitled)"
        source = ref.get("source") or ""
        url = ref.get("url") or ""
        published = ref.get("publishedAt") or ""
        findings = ref.get("keyFindings") or []
        snippet = ""
        for f in findings:
            if f and not f.lower().startswith("journal:"):
                snippet = f
                break
        meta_bits = [b for b in [source, published] if b]
        link_html = (
            f"<div class='medi-ref-link'><a href='{url}' target='_blank'>Open ↗</a></div>"
            if url
            else ""
        )
        snippet_html = (
            f"<div class='medi-ref-snippet'>{snippet}</div>" if snippet else ""
        )
        container.markdown(
            f"""
<div class='medi-ref-card'>
  <div><span class='medi-ref-num'>{i}</span><span class='medi-ref-title'>{title}</span></div>
  <div class='medi-ref-meta'>{' · '.join(meta_bits)}</div>
  {link_html}
  {snippet_html}
</div>
""",
            unsafe_allow_html=True,
        )


def _render_strict_answer(container, data):
    answer = data.get("answer") or ""
    confidence = int(data.get("confidenceScore") or 0)
    status = data.get("status") or "insufficient_evidence"
    label, color = _STATUS_LABEL.get(status, (status, "#6b7280"))
    refs = data.get("references") or []

    container.markdown(
        f"<span class='medi-status-pill' style='background:{color}'>{label}</span>"
        f"<span class='medi-confidence'>Confidence: {confidence}/100</span>",
        unsafe_allow_html=True,
    )
    container.markdown(answer)
    if refs:
        tags = " ".join(
            f"<a href='{r.get('url','')}' target='_blank' "
            f"style='display:inline-block;margin:4px 6px 0 0;padding:2px 8px;"
            f"border-radius:8px;background:#1f2937;color:#93c5fd;"
            f"text-decoration:none;font-size:12px;'>[{i}] {(r.get('source') or 'src')}</a>"
            for i, r in enumerate(refs, start=1)
        )
        container.markdown(
            f"<div style='margin-top:8px'>Sources: {tags}</div>", unsafe_allow_html=True
        )


def render_chat():
    _inject_styles()
    st.subheader("Chat with your assistant")
    st.caption(
        "Ask any clinical question. The assistant searches PubMed live, verifies "
        "evidence, and shows the source links on the right. Upload PDFs in the "
        "sidebar to add internal references."
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    chat_col, ref_col = st.columns([2, 1], gap="large")

    with chat_col:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                if msg.get("html"):
                    st.markdown(msg["html"], unsafe_allow_html=True)
                else:
                    st.markdown(msg["content"])

        user_input = st.chat_input("Type your question....")

    last_refs = st.session_state.get("last_references", [])
    last_empty_msg = st.session_state.get(
        "last_empty_message", "Ask a question to see sources here."
    )
    _render_reference_panel(ref_col, last_refs, empty_message=last_empty_msg)

    if not user_input:
        return

    with chat_col:
        st.chat_message("user").markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.spinner("Searching PubMed and verifying evidence..."):
            response = ask_question_strict(user_input)

        if response.status_code in (200, 202):
            data = response.json()
            with st.chat_message("assistant"):
                _render_strict_answer(st, data)
            st.session_state.messages.append(
                {"role": "assistant", "content": data.get("answer", "")}
            )
            st.session_state["last_references"] = data.get("references", []) or []
            st.session_state["last_empty_message"] = (
                (data.get("ui") or {}).get("emptyReferencesMessage")
                or "No related sources found."
            )
            st.rerun()
        else:
            st.error(f"Error: {response.text}")
