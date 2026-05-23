import streamlit as st
from utils.api import list_libraries_api, upload_pdfs_api


def render_uploader():
    st.sidebar.header("📚 Library")
    existing = list_libraries_api() or ["default"]
    options = existing + ["➕ Create new library..."]
    selected = st.sidebar.selectbox(
        "Active library",
        options,
        key="library_selector",
        help="A library is an isolated Pinecone namespace. PDFs uploaded here stay separate from other libraries.",
    )
    if selected == "➕ Create new library...":
        new_name = st.sidebar.text_input(
            "New library name", key="new_library_name", placeholder="e.g. cardiology"
        )
        active_library = new_name.strip() or "default"
    else:
        active_library = selected
    st.session_state["active_library"] = active_library
    st.sidebar.caption(f"Querying library: **{active_library}**")

    st.sidebar.divider()
    st.sidebar.header("📄 Upload PDFs")
    uploaded_files = st.sidebar.file_uploader(
        "Add to this library", type="pdf", accept_multiple_files=True
    )
    if st.sidebar.button("Upload to library") and uploaded_files:
        with st.spinner(f"Indexing into library '{active_library}'..."):
            response = upload_pdfs_api(uploaded_files, library=active_library)
        if response is not None and response.status_code == 200:
            st.sidebar.success(f"Uploaded to '{active_library}'.")
        else:
            err = response.text if response is not None else "Upload failed"
            st.sidebar.error(f"Error: {err}")
