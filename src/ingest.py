# -*- coding: utf-8 -*-
"""Ingest the KB markdown into an embedded Chroma store.
Run once from the repo root (re-run after editing the KB):

    python src/ingest.py
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import frontmatter
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

import config


def load_kb_documents(kb_dir) -> list:
    """Read every .md under the KB, parse YAML front matter, and return LangChain
    Documents carrying metadata: doc_id, category, lang, title, source,
    plus optional numeric policy fields (e.g. return_window_days) when present."""
    docs = []
    for md_path in sorted(pathlib.Path(kb_dir).rglob("*.md")):
        post = frontmatter.load(md_path)
        meta = {
            "doc_id": post.get("doc_id"),
            "category": post.get("category"),
            "lang": post.get("lang"),
            "title": post.get("title"),
            "source": str(md_path.relative_to(kb_dir)),
        }
        # 選擇性欄位：只有該篇有才放進 metadata（Chroma 不接受 None 值）
        for k in ("return_window_days", "return_window_days_defect"):
            v = post.get(k)
            if v is not None:
                meta[k] = v          # YAML 的 14 會 parse 成 int，正好給日期計算
        docs.append(Document(page_content=post.content.strip(), metadata=meta))
    return docs


def split_documents(docs: list) -> list:
    """依 config 的 CHUNK_SIZE / CHUNK_OVERLAP 切塊；本語料為短政策文件，採 doc-level。"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""],  # zh + en aware
    )
    return splitter.split_documents(docs)


def build_store(chunks: list) -> Chroma:
    """以 bge-m3 embedding 並寫入嵌入式 Chroma（cosine 空間）。"""
    embeddings = OllamaEmbeddings(model=config.EMBED_MODEL)
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=config.COLLECTION,
        persist_directory=str(config.CHROMA_DIR),
        collection_metadata={"hnsw:space": "cosine"}
    )


def main():
    docs = load_kb_documents(config.KB_DIR)
    print(f"loaded {len(docs)} KB documents from {config.KB_DIR}")
    if not docs:
        print("!! 找不到 .md：請將知識庫放在 data/kb_ecom_cs/（zh/ 與 en/）下。")
        return
    chunks = split_documents(docs)
    print(f"split into {len(chunks)} chunks")
    build_store(chunks)
    print(f"embedded + persisted to {config.CHROMA_DIR}")


if __name__ == "__main__":
    main()
