# -*- coding: utf-8 -*-
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
KB_DIR = REPO_ROOT / "data" / "kb_ecom_cs"   # 知識庫（zh/ 與 en/）
CHROMA_DIR = REPO_ROOT / "chroma_store"      # 嵌入式 Chroma 持久化目錄
COLLECTION = "cs_kb"

# --- models: all local via Ollama ---
EMBED_MODEL = "bge-m3"
LLM_MODEL = "qwen2.5:7b"

CHUNK_SIZE = 900        # 量過文件長度（EN max≈791、ZH max≈255）→ 900 使兩語皆整篇（doc-level）
CHUNK_OVERLAP = 0       # doc-level 不切塊，overlap 不適用
TOP_K = 3               # 多數問題對應單篇；+2 篇作為脈絡安全邊際

SCORE_THRESHOLD = 0.49  # 以 68 題（含對抗題）校準：可答/離題距離間隔收窄至 0.467–0.508，取中
                        # NOTE: Chroma 的 score 是距離（distance），越小越相近
FILTER_BY_LANG = True   # 只檢索與查詢同語言的文件（語言感知檢索）
