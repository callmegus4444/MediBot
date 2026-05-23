import streamlit as st

from components.chatUI import render_chat
from components.history_download import render_history_download
from components.sessions import render_sessions_panel
from components.upload import render_uploader


st.set_page_config(page_title="MediBot — AI Medical Assistant", layout="wide")
st.title("🩺 MediBot — Medical Assistant Chatbot")

render_uploader()
render_sessions_panel()
render_history_download()
render_chat()
