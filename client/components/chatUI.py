import streamlit as st
from utils.api import ask_question, ask_question_strict


def _render_reference_panel(container, references, empty_message="No verified sources found"):
    container.markdown("### References")
    if not references:
        container.info(empty_message)
        return
    for ref in references:
        title = ref.get("title", "(untitled)")
        source = ref.get("source", "")
        url = ref.get("url", "")
        confidence = ref.get("confidenceScore", 0)
        published = ref.get("publishedAt") or ""
        with container.container(border=True):
            container.markdown(f"**{title}**")
            meta_bits = [b for b in [source, published] if b]
            if meta_bits:
                container.caption(" · ".join(meta_bits))
            if url:
                container.markdown(f"[Open source]({url})")
            container.progress(min(max(int(confidence), 0), 100) / 100.0)
            container.caption(f"Source confidence: {confidence}/100")


def _render_strict_answer(answer_col, data):
    answer = data.get("answer", "")
    confidence = data.get("confidenceScore", 0)
    status = data.get("status", "insufficient_evidence")
    verification = data.get("verification", {}) or {}

    badge = {
        "answered": "OK",
        "partial": "PARTIAL",
        "insufficient_evidence": "ABSTAINED",
        "conflicting_evidence": "CONFLICTING",
        "processing": "WORKING",
    }.get(status, status.upper())

    answer_col.markdown(f"**Status:** `{badge}` &nbsp; **Confidence:** {confidence}/100")
    answer_col.progress(min(max(int(confidence), 0), 100) / 100.0)
    answer_col.markdown(answer)
    if verification.get("conflictsDetected"):
        answer_col.warning("Conflicting evidence detected across sources.")
    if verification.get("unsupportedClaimsRemoved", 0):
        answer_col.caption(
            f"{verification['unsupportedClaimsRemoved']} unsupported claim(s) were removed by verification."
        )


def render_chat():
    st.subheader("Chat with your assistant")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    strict_mode = st.toggle(
        "Strict evidence mode (doctor-facing)",
        value=st.session_state.get("strict_mode", False),
        help="Only answers backed by verified sources. Otherwise abstains.",
    )
    st.session_state["strict_mode"] = strict_mode

    ref_col, chat_col = st.columns([1, 2])

    last_refs = st.session_state.get("last_references", [])
    last_empty_msg = st.session_state.get(
        "last_empty_message", "No verified sources found"
    )
    _render_reference_panel(ref_col, last_refs, empty_message=last_empty_msg)

    with chat_col:
        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).markdown(msg["content"])

        user_input = st.chat_input("Type your question....")
        if not user_input:
            return

        st.chat_message("user").markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        if strict_mode:
            with st.spinner("Verifying evidence..."):
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
                    or "No verified sources found"
                )
                st.rerun()
            else:
                st.error(f"Error: {response.text}")
        else:
            response = ask_question(user_input)
            if response.status_code == 200:
                data = response.json()
                answer = data["response"]
                st.chat_message("assistant").markdown(answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )
            else:
                st.error(f"Error: {response.text}")
