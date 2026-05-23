import streamlit as st

from utils.api import save_session_api, stream_strict


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
.medi-status-pill { display:inline-block; padding:4px 10px; border-radius:999px;
    font-size:12px; font-weight:600; color:#fff; }
.medi-confidence { display:inline-block; margin-left:8px; padding:4px 10px;
    border-radius:999px; font-size:12px; font-weight:600;
    background:#1f2937; color:#e5e7eb; }
.medi-ref-card { border:1px solid #1f2937; background:#0f172a; border-radius:10px;
    padding:12px 14px; margin-bottom:10px; }
.medi-ref-num { display:inline-block; min-width:22px; height:22px; line-height:22px;
    text-align:center; border-radius:11px; background:#1d4ed8; color:#fff;
    font-size:12px; font-weight:700; margin-right:8px; }
.medi-ref-title { font-weight:600; color:#e5e7eb; line-height:1.35; }
.medi-ref-meta { color:#9ca3af; font-size:12px; margin:4px 0 8px; }
.medi-ref-link a { color:#60a5fa !important; font-weight:600; text-decoration:none; }
.medi-ref-link a:hover { text-decoration:underline; }
.medi-ref-snippet { color:#cbd5e1; font-size:12.5px; line-height:1.45; margin-top:6px;
    max-height:120px; overflow:hidden; }
.medi-empty { color:#9ca3af; font-style:italic; padding:10px 4px; }
</style>
""",
        unsafe_allow_html=True,
    )


def _badge_for(ref: dict) -> str:
    source_type = (ref.get("sourceType") or "").lower()
    source = ref.get("source") or ""
    tier = ref.get("credibilityTier") or ""
    palette = {
        "peer_reviewed_journal": ("PubMed", "#0ea5e9"),
        "web": ("Web", "#10b981"),
        "internal_pdf": ("Internal", "#a78bfa"),
        "drug_label": ("FDA", "#f59e0b"),
        "clinical_trial": ("Trial", "#ec4899"),
    }
    label, color = palette.get(source_type, (source or "Source", "#64748b"))
    tier_html = (
        f" <span style='background:#334155;color:#e2e8f0;padding:1px 6px;"
        f"border-radius:6px;font-size:10px;margin-left:4px;'>Tier {tier}</span>"
        if tier
        else ""
    )
    return (
        f"<span style='background:{color};color:#fff;padding:2px 8px;"
        f"border-radius:6px;font-size:11px;font-weight:600;'>{label}</span>"
        f"{tier_html}"
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
        badge_html = _badge_for(ref)
        meta_bits = [badge_html] + [b for b in [source, published] if b]
        link_html = (
            f"<div class='medi-ref-link'><a href='{url}' target='_blank'>Open ↗</a></div>"
            if url and not url.startswith("local://")
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


def _render_status(container, status, confidence):
    label, color = _STATUS_LABEL.get(status, (status, "#6b7280"))
    container.markdown(
        f"<span class='medi-status-pill' style='background:{color}'>{label}</span>"
        f"<span class='medi-confidence'>Confidence: {confidence}/100</span>",
        unsafe_allow_html=True,
    )


def _history_for_backend(messages):
    out = []
    for m in messages:
        if m.get("role") not in ("user", "assistant"):
            continue
        content = m.get("content") or ""
        if not content:
            continue
        out.append({"role": m["role"], "content": content})
    return out


def render_chat():
    _inject_styles()

    top_row = st.columns([3, 1, 1])
    top_row[0].subheader("Chat with your assistant")
    use_web = top_row[1].toggle(
        "🌐 Web search", value=st.session_state.get("use_web", True),
        help="Search trusted medical websites alongside PubMed."
    )
    st.session_state["use_web"] = use_web
    library = st.session_state.get("active_library", "default")
    top_row[2].caption(f"📚 {library}")

    st.caption(
        "Searches PubMed, trusted medical websites (WHO/CDC/FDA/NIH/Mayo/NEJM/BMJ), "
        "the FDA drug label database (OpenFDA), and ClinicalTrials.gov in parallel. "
        "Toggle web search off above if you want PubMed-only."
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    chat_col, ref_col = st.columns([2, 1], gap="large")

    with chat_col:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                if msg["role"] == "assistant" and msg.get("status"):
                    _render_status(st, msg["status"], msg.get("confidenceScore", 0))
                st.markdown(msg.get("content") or "")

        user_input = st.chat_input("Type your clinical question...")

    last_refs = st.session_state.get("last_references", [])
    last_empty_msg = st.session_state.get(
        "last_empty_message", "Ask a question to see sources here."
    )
    _render_reference_panel(ref_col, last_refs, empty_message=last_empty_msg)

    if not user_input:
        return

    history_payload = _history_for_backend(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with chat_col:
        st.chat_message("user").markdown(user_input)
        assistant_box = st.chat_message("assistant")
        status_slot = assistant_box.empty()
        text_slot = assistant_box.empty()

    status_slot.markdown(
        "<span class='medi-status-pill' style='background:#475569'>Searching sources…</span>",
        unsafe_allow_html=True,
    )
    text_slot.markdown("_Retrieving PubMed, web, OpenFDA, ClinicalTrials.gov…_")

    collected = []
    final_status = "answered"
    final_conf = 0
    final_refs: list = []
    streaming_refs: list = []
    error_message = None

    for event in stream_strict(
        user_input,
        library=library,
        use_web=use_web,
        history=history_payload,
    ):
        ev = event.get("event")
        data = event.get("data") or {}
        if ev == "references":
            streaming_refs = data if isinstance(data, list) else []
            st.session_state["last_references"] = streaming_refs
            _render_reference_panel(ref_col, streaming_refs, last_empty_msg)
        elif ev == "delta":
            text = (data or {}).get("text") if isinstance(data, dict) else None
            if text:
                if not collected:
                    status_slot.markdown(
                        "<span class='medi-status-pill' style='background:#0ea5e9'>Streaming…</span>",
                        unsafe_allow_html=True,
                    )
                collected.append(text)
                text_slot.markdown("".join(collected))
        elif ev == "done":
            final_status = data.get("status") or "answered"
            final_conf = int(data.get("confidenceScore") or 0)
            final_refs = data.get("references") or streaming_refs
            if not collected and data.get("answer"):
                collected = [data["answer"]]
                text_slot.markdown(data["answer"])
        elif ev == "error":
            error_message = (data or {}).get("message") if isinstance(data, dict) else str(data)

    if error_message:
        status_slot.error(f"Error: {error_message}")
        return

    final_text = "".join(collected).strip() or "(no answer)"
    status_slot.empty()
    _render_status(assistant_box, final_status, final_conf)
    text_slot.markdown(final_text)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": final_text,
            "status": final_status,
            "confidenceScore": final_conf,
            "references": final_refs,
        }
    )
    st.session_state["last_references"] = final_refs or streaming_refs

    session_id = st.session_state.get("session_id")
    if session_id:
        save_session_api(
            session_id,
            st.session_state.messages,
            title=st.session_state.messages[0]["content"][:60] if st.session_state.messages else None,
            library=library,
        )
