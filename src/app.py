# -*- coding: utf-8 -*-
"""Streamlit chat UI. Run from the repo root:

    streamlit run src/app.py

檢索與生成邏輯在 retriever.py / generate.py；此檔為 Streamlit 介面。"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import streamlit as st
from retriever import get_store, retrieve
from generate import answer

st.set_page_config(page_title="沐光選物 客服助手 / MiraLoom CS", page_icon="💬")
st.title("沐光選物 客服知識庫問答")
st.caption("問問看退換貨、運費、付款、會員…（中文或英文皆可）")


@st.cache_resource
def _store():
    return get_store()


if "history" not in st.session_state:
    st.session_state.history = []

for role, text in st.session_state.history:
    with st.chat_message(role):
        st.markdown(text)

query = st.chat_input("輸入你的問題…")
if query:
    st.session_state.history.append(("user", query))
    with st.chat_message("user"):
        st.markdown(query)

    from agent import handle
    out = handle(query, _store())

    reply = out["answer"]
    if out["sources"]:
        cites = "  \n".join(f"- `{s['doc_id']}` · {s['title']}" for s in out["sources"])
        reply += f"\n\n---\n**來源 / Sources**  \n{cites}"

    st.session_state.history.append(("assistant", reply))
    with st.chat_message("assistant"):
        st.markdown(reply)
