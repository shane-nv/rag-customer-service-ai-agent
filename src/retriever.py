# -*- coding: utf-8 -*-
"""Retrieval over the embedded Chroma store: language-aware filtering + a refuse gate."""
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

import config


def get_store() -> Chroma:
    """Open the persisted store."""
    return Chroma(
        collection_name=config.COLLECTION,
        persist_directory=str(config.CHROMA_DIR),
        embedding_function=OllamaEmbeddings(model=config.EMBED_MODEL),
        collection_metadata={"hnsw:space": "cosine"}
    )

import re

def detect_lang(text: str) -> str:
    """判斷語言：含 CJK 字元回 'zh-Hant'，否則 'en'。查詢短、兩語字元不重疊，簡單啟發式即足夠。"""
    return "zh-Hant" if re.search(r"[\u4e00-\u9fff]", text) else "en"

def retrieve(query: str, store: Chroma):
    """回傳 (results, refused)。results 為 (Document, score) 的 list。
    - 語言過濾：FILTER_BY_LANG 為真時只檢索同語言文件。
    - 拒答閘：最佳距離超過門檻即 refused=True（Chroma score 為距離，越小越近）。
    """
    lang = detect_lang(query)
    flt = {"lang": lang} if config.FILTER_BY_LANG else None

    results = store.similarity_search_with_score(query, k=config.TOP_K, filter=flt)

    best = results[0][1] if results else None
    refused = (best is None) or (best > config.SCORE_THRESHOLD)

    return results, refused
