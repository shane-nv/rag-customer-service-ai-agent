# -*- coding: utf-8 -*-
"""LLM-planner multi-tool orchestration on top of the RAG.

Design principle : JUDGMENT to the model, EXECUTION/CALCULATION to code.
  - Planner (LLM)   : 出嚴格 JSON 計劃，決定叫哪些工具、順序。工具 ∈ {get_order_status, lookup_policy}。
  - Validator (code): 白名單 / 步數上限 / 參數即真相；非法或空 → 確定性降級 _fallback()。
  - Executor (code) : 依序跑工具；湊齊 signoff_date + window_days 就「自動」算退貨視窗（模型碰不到）。
                      並偵測「送達爭議」（系統顯示已送達 vs 客人說沒收到）→ 不給誤導的退貨建議。
  - Composer        : 在地化組合 + 引用。

app.py wiring 不變：
    from agent import handle
    out = handle(query, _store())
    # out 內含 out["plan"]（規劃層輸出，給評估集比對用）
"""
from datetime import datetime
from datetime import date, timedelta
import json
import re

from langchain_ollama import ChatOllama
from langchain_core.tools import tool

import config
from retriever import get_store, retrieve, detect_lang
from generate import answer as rag_answer


# --- mock order backend (clearly synthetic; a real system would hit an API/DB) ---
# 日期相對今天算 → demo 不會隨時間失效。ML240003 在退貨視窗內、ML240004 已超過 14 天（測 not-eligible）。
MOCK_ORDERS = {
    "ML240001": {"status": "shipped",    "carrier": "DHL Express", "tracking": "JD0099887766", "eta": (date.today() + timedelta(days=2)).strftime('%Y-%m-%d')},
    "ML240002": {"status": "processing", "carrier": None,          "tracking": None,           "eta": (date.today() + timedelta(days=5)).strftime('%Y-%m-%d')},
    "ML240003": {"status": "delivered",  "carrier": "黑貓宅急便",   "tracking": "T-1234567890", "eta": (date.today() - timedelta(days=7)).strftime('%Y-%m-%d'),  "signoff_date": (date.today() - timedelta(days=7)).strftime('%Y-%m-%d')},
    "ML240004": {"status": "delivered",  "carrier": "新竹貨運",     "tracking": "T-2468024345", "eta": (date.today() - timedelta(days=16)).strftime('%Y-%m-%d'), "signoff_date": (date.today() - timedelta(days=16)).strftime('%Y-%m-%d')},
}

# This store's order-id format（參數即真相）。
# 不用 \b：Python 把中文字當 \w，"訂單ML240001的" 這種黏字會讓 \b 失效 → 改用 ASCII 英數的
# lookbehind/lookahead，中文緊貼也抓得到，且不誤中更長的英數 token（XML2400012 / 7碼 / ML240001A）。
ORDER_ID_RE = re.compile(r"(?<![A-Za-z0-9])ML\d{6}(?![A-Za-z0-9])", re.I)

# 「客人表示沒收到」的信號（關鍵詞啟發式，與 detect_lang 同精神；paraphrase 可能漏，屬已知限制）。
_NOT_RECEIVED_RE = re.compile(
    r"沒收到|沒有收到|未收到|還沒收到|還沒到|沒拿到|沒送到|未送達|沒送達|沒到貨|還沒到貨"
    r"|not\s+receiv|haven'?t\s+receiv|didn'?t\s+receiv|never\s+arriv|hasn'?t\s+arriv|not\s+arriv",
    re.I,
)


def _claims_not_received(query: str) -> bool:
    """客人是否在抱怨『沒收到』。"""
    return bool(_NOT_RECEIVED_RE.search(query))


# ======================================================================
# Tools (曝露給規劃器的 2 個) ＋ 政策問答的唯一實作
# ======================================================================

@tool
def get_order_status(order_id: str) -> dict:
    """Look up the LIVE status of a customer order by its order id (e.g. ML240001).
    Use ONLY when the user asks about a specific order.
    Returns status/carrier/tracking/eta/signoff_date, or {'found': False} when unknown."""
    o = MOCK_ORDERS.get(order_id.strip().upper())
    return {"found": bool(o), "order_id": order_id.strip().upper(), **(o or {})}


def _answer_policy(query: str, store) -> dict:
    """唯一的政策問答路徑：檢索 + grounded 生成（含拒答閘）。
    - rag_answer 只呼叫一次。
    - 從命中文件的 metadata 帶出 return_window_days（掃 top-k，不只 rank-1），供執行層算退貨視窗。"""
    results, refused = retrieve(query, store)
    out = rag_answer(query, results, refused)          # {'answer', 'sources'}
    out["refused"] = refused
    out["window_days"] = next(
        (d.metadata.get("return_window_days") for d, _ in results
         if d.metadata.get("return_window_days") is not None),
        None,
    )
    return out


@tool
def lookup_policy(sub_query: str) -> dict:
    """Answer ANY policy/FAQ question (shipping, payment, returns, membership...) from the KB.
    薄包裝：模型介面拿不到 store，故自行 get_store()；執行層請直接呼叫 _answer_policy(query, store)。"""
    return _answer_policy(sub_query, get_store())


# ======================================================================
# 執行層確定性計算（不是工具、模型看不到）
# ======================================================================

def check_return_window(signoff_date: str, window_days: int, today: date = None) -> dict:
    try:
        signoff = datetime.strptime(signoff_date, "%Y-%m-%d").date()
    except ValueError:
        raise ValueError(f"signoff_date must be 'YYYY-MM-DD' but got {signoff_date!r}")
    today = today or date.today()
    deadline = signoff + timedelta(days=window_days)
    days_left = (deadline - today).days
    return {
        "eligible": days_left >= 0,
        "days_left": days_left,
        "deadline": deadline.strftime("%Y-%m-%d"),
    }

# ======================================================================
# 規劃器（LLM）
# ======================================================================

PLANNER_SYSTEM = """
  1. FORMAT(STRICTLY):
      Only output in JSON format. For example: {"steps": [{"tool": "<name>", "args": {...}}]}
      NOTE: 以下範例僅示範格式與判斷模式，其查詢句與訂單編號刻意「不」與任何評估題重複（避免 train/test 洩漏）。

  2. TOOLS CALLING:
       - Order tool: get_order_status. Only when the user mentions an order ID (e.g. ML123456) or asks about a specific order, use get_order_status to check the status of the order.
         Example:
         User: ML999001 現在到哪了?
         {"steps": [{"tool": "get_order_status", "args": {"order_id": "ML999001"}}]}

       - Policy tool: lookup_policy. When the user asks about questions answerable by policies (shipping / delivery regions / address change / payment / returns / refunds / membership, etc.), use lookup_policy. 「寄不寄到某國」「運費怎麼算」「改地址」這類都是政策問題，要用 lookup_policy，不要當成離題。
         Example:
         User: 你們可以寄到新加坡嗎?
         {"steps": [{"tool": "lookup_policy", "args": {"sub_query": "配送範圍與國家"}}]}

       - Multi-tools: get_order_status + lookup_policy. One question may require multiple steps.
         Example (退貨資格 + 退款時程，各出一個 lookup_policy 子問題):
         User: 訂單 ML999002 可以退貨嗎？退款多久會到帳？
         {"steps": [
           {"tool": "get_order_status", "args": {"order_id": "ML999002"}},
           {"tool": "lookup_policy", "args": {"sub_query": "退貨資格與時效"}},
           {"tool": "lookup_policy", "args": {"sub_query": "退款到帳時間"}}
         ]}
         Example (多段口語；訂單相關的「在哪/還要多久」由 get_order_status 一併涵蓋，政策意圖另出一步):
         User: ML999003 出貨了嗎？順便想改收件地址
         {"steps": [
           {"tool": "get_order_status", "args": {"order_id": "ML999003"}},
           {"tool": "lookup_policy", "args": {"sub_query": "出貨後可以修改收件地址嗎"}}
         ]}

  4. Off-topic/casual conversation → return empty steps: {"steps": []}
        Example:
        User: 可以幫我推薦附近餐廳嗎?
        {"steps": []}

  5. Partial off-topic/casual conversation → 
        If the conversation contain partial parts of off-topic, ignore the off-topic parts and only return the steps for the policy-related / tool-calling parts.
        Example:
        User: ML999003 的貨到哪了？你們月薪多少？
        {"steps": [{"tool": "get_order_status", "args": {"order_id": "ML999003"}}]}

  6. Not-received complaints ("沒收到" / "haven't received" / "never arrived" ...):
        這仍然要先查訂單 → 出 get_order_status 步。不要輸出 steps 以外的任何欄位。
        Example:
        User: ML999003 的貨沒收到
        {"steps": [{"tool": "get_order_status", "args": {"order_id": "ML999003"}}]}
"""

ALLOWED_TOOLS = {"get_order_status", "lookup_policy"}
MAX_STEPS = 4


def plan(query: str) -> dict | None:
    """LLM 出計劃。format='json' 強制合法 JSON（A 能穩的關鍵）。解析失敗回 None → 交給降級。"""
    llm = ChatOllama(model=config.LLM_MODEL, temperature=0, format="json")
    resp = llm.invoke([("system", PLANNER_SYSTEM), ("human", query)])
    try:
        return json.loads(resp.content)
    except (json.JSONDecodeError, TypeError):
        return None


def validate_plan(p: dict | None, query: str) -> list | None:
    """白名單 + 步數上限 + 參數即真相。回乾淨的 steps（可能為空 []），或 None（格式壞掉 → 降級）。
    健壯性：小模型可能吐出 steps=["get_order_status", ...] 這種「字串而非物件」的形狀，
    validator 必須擋住而不是自己爆掉（整套設計的前提就是 planner 會亂吐）。"""
    if not p or "steps" not in p or not isinstance(p["steps"], list):
        return None
    clean = []
    for s in p["steps"][:MAX_STEPS]:
        if isinstance(s, str):                        # 只給了工具名 → 補成標準形狀
            s = {"tool": s, "args": {}}
        if not isinstance(s, dict):                   # 其他非預期型別 → 丟掉這步
            continue
        tool_name = s.get("tool")
        if tool_name not in ALLOWED_TOOLS:            # 白名單外 → 整個計劃視為不可信
            return None
        if tool_name == "get_order_status":
            m = ORDER_ID_RE.search(query)             # 參數即真相：query 沒有真編號就丟這步
            if not m:
                continue
            s["args"] = {"order_id": m.group(0)}
        elif tool_name == "lookup_policy":
            if not (s.get("args") or {}).get("sub_query"):
                s["args"] = {"sub_query": query}      # 沒給子問題就用原 query 兜底
        clean.append(s)

    # 確定性補救（延伸「參數即真相」）：客人說沒收到 + 查詢裡有真編號 → 一定要查訂單，
    # 否則 execute 拿不到訂單狀態、dispute 永遠觸發不了。不倚賴 planner 每次都排對。
    m = ORDER_ID_RE.search(query)
    if m and _claims_not_received(query) and not any(s["tool"] == "get_order_status" for s in clean):
        clean.insert(0, {"tool": "get_order_status", "args": {"order_id": m.group(0)}})

    return clean


# ======================================================================
# 執行 + 組合 + 降級
# ======================================================================

def execute(steps: list, query: str, store) -> dict:
    """跑步驟、湊事實。
    - 湊齊簽收日 + 政策視窗天數 → 自動算退貨視窗（模型碰不到這步）。
    - 送達爭議（已送達 vs 客人說沒收到）→ 標記，且不算退貨視窗（避免給誤導的退貨建議）。"""
    facts = {"order": None, "policy": [], "return_window": None, "dispute": False}
    for s in steps:
        if s["tool"] == "get_order_status":
            facts["order"] = get_order_status.invoke(s["args"])
        elif s["tool"] == "lookup_policy":
            facts["policy"].append(_answer_policy(s["args"].get("sub_query", query), store))

    order = facts["order"] or {}
    # 送達爭議：系統顯示已送達，但客人表示沒收到 → 這是查件/轉真人的事，不是標準退貨
    if order.get("status") == "delivered" and _claims_not_received(query):
        facts["dispute"] = True

    signoff = order.get("signoff_date")
    window = next((p["window_days"] for p in facts["policy"]
               if not p.get("refused") and p.get("window_days") is not None), None)
    if signoff and window is not None and not facts["dispute"]:   # 爭議時不算退貨視窗
        facts["return_window"] = check_return_window(signoff, window, today=date.today())
    return facts


def compose(query: str, facts: dict) -> dict:
    order_exists = bool(facts.get("order") and facts["order"].get("found"))
    policy_answers = facts.get("policy", [])
    return_window = facts.get("return_window")
    dispute = facts.get("dispute")

    # 依 facts 內容決定 via 類型
    if dispute:
        via = "dispute"
    elif order_exists and return_window:
        via = "multi"
    elif policy_answers:
        via = "policy"
    else:
        via = "tool"   # 純訂單、或訂單查不到的兜底

    parts = []
    sources = []

    # 送達爭議：只回爭議處理訊息，壓下訂單/政策/退貨各段（避免自相矛盾與誤導）
    if dispute:
        parts.append(_format_dispute(query, facts["order"]))
        return {"answer": "\n\n".join(parts), "sources": sources, "via": via}

    # 訂單資訊（found=False 的訊息 _format_order_reply 自己會處理）
    if facts.get("order"):
        parts.append(_format_order_reply(query, facts["order"]))

    # 政策資訊與來源（去重）；檢索不到的子問題不併進多段回覆
    for policy_result in policy_answers:
        if policy_result.get("refused"):
            continue
        if policy_result.get("answer"):
            parts.append(policy_result["answer"])
        for s in policy_result.get("sources", []):
            if s not in sources:
                sources.append(s)

    # 退貨視窗（算出來的判定）
    if return_window:
        parts.append(_format_return_window(query, facts["return_window"]))

    if parts:
        return {"answer": "\n\n".join(parts), "sources": sources, "via": via}
    return {"answer": "很抱歉，我無法回答您的問題。", "sources": [], "via": via}


def _format_order_reply(query: str, result: dict) -> str:
    """Deterministically phrase the order-status result in the user's language."""
    lang = detect_lang(query)
    order_id = result.get("order_id", "")

    # 1. 找不到訂單 → 提醒正確的 ID 格式
    if not result.get("found"):
        if lang == "zh-Hant":
            return f"抱歉，系統中找不到訂單「{order_id}」。請確認您的訂單編號格式是否正確（應為 'ML' 開頭加上 6 位數字，例如：ML240001）。"
        else:
            return f"Sorry, order '{order_id}' could not be found. Please ensure your order ID is correct (it should start with 'ML' followed by 6 digits, e.g., ML240001)."

    # 2. 處理訂單狀態字串與空值防護
    status_raw = result.get("status", "").lower().strip()
    carrier = result.get("carrier") or "尚未指派"
    tracking = result.get("tracking") or "尚未產生"
    eta = result.get("eta") or "計算中"

    if lang == "zh-Hant":
        status_map = {"processing": "處理中", "shipped": "已出貨", "delivered": "已送達"}
        status = status_map.get(status_raw, status_raw)
        if status == "處理中":
            return f"您的訂單「{order_id}」目前正在處理中，尚未出貨。"
        elif status == "已出貨":
            return (f"您的訂單「{order_id}」已出貨！\n物流公司：{carrier}\n運單號碼：{tracking}\n預計送達：{eta}")
        elif status == "已送達":
            return (f"您的訂單「{order_id}」已送達！\n物流公司：{carrier}\n運單號碼：{tracking}")
        else:
            return f"訂單「{order_id}」狀態：{status}"
    else:
        carrier_en = result.get("carrier") or "Not assigned"
        tracking_en = result.get("tracking") or "Not generated"
        eta_en = result.get("eta") or "Calculating"
        status = status_raw
        if status == "processing":
            return f"Your order '{order_id}' is currently being processed and has not yet shipped."
        elif status == "shipped":
            return (f"Your order '{order_id}' has been shipped!\nCarrier: {carrier_en}\nTracking Number: {tracking_en}\nEstimated Delivery: {eta_en}")
        elif status == "delivered":
            return (f"Your order '{order_id}' has been delivered!\nCarrier: {carrier_en}\nTracking Number: {tracking_en}")
        else:
            return f"Order '{order_id}' status: {status}"


def _format_return_window(query, return_window) -> str:
    """Format return window information based on user query and result."""
    lang = detect_lang(query)
    if return_window.get("eligible"):
        if lang == 'zh-Hant':
            return f"您可以在 {return_window.get('days_left')} 天內退貨，期限至 {return_window.get('deadline')}。若商品為瑕疵或出貨錯誤，退貨期限為 30 天。"
        else:
            return f"You are eligible to return the product within {return_window.get('days_left')} days, until {return_window.get('deadline')}. If the product is defective or the order was shipped in error, the return window is 30 days."
    else:
        overdue = -return_window["days_left"]
        if lang == "zh-Hant":
            return f"您的訂單已超過退貨期限（期限為 {return_window['deadline']}，已逾期 {overdue} 天）。若商品為瑕疵或出貨錯誤，退貨期限為 30 天。"
        else:
            return f"The return window has passed (deadline was {return_window['deadline']}, {overdue} day(s) ago). If the item is defective or mis-shipped, the window is 30 days."


def _format_dispute(query: str, order: dict) -> str:
    """送達爭議：系統顯示已送達 vs 客人說沒收到。措辭可依你 LQA 調整。"""
    lang = detect_lang(query)
    oid = order.get("order_id", "")
    d = order.get("signoff_date") or order.get("eta") or ""
    if lang == "zh-Hant":
        return (f"系統顯示您的訂單「{oid}」已於 {d} 送達，但您表示尚未收到。"
                f"為保障您的權益，我們將為您啟動「未送達查件」流程並轉接真人客服協助，"
                f"恕無法逕行退貨或退款。")
    else:
        return (f"Our records show order '{oid}' was delivered on {d}, but you indicate it hasn't arrived. "
                f"To protect you, we'll open an undelivered-package investigation and connect you with a "
                f"human agent; a standard return or refund can't be processed in this case.")


def _fallback(query: str, store) -> dict:
    """確定性降級：planner 不可用時走這條（沿用 v1 哲學：regex 為真相）。純 plumbing，非新判斷。"""
    m = ORDER_ID_RE.search(query)
    if m:
        result = get_order_status.invoke({"order_id": m.group(0)})
        return {"answer": _format_order_reply(query, result), "sources": [], "via": "fallback_tool", "plan": None}
    out = _answer_policy(query, store)
    out["via"] = "fallback_refuse" if out.get("refused") else "fallback_policy"
    out["plan"] = None
    return out


def handle(query: str, store) -> dict:
    """入口：plan → validate → execute → compose；規劃不可用則降級。out['plan'] 帶出規劃層輸出。"""
    steps = validate_plan(plan(query), query)
    if steps is None:                 # planner 壞掉/格式錯 → 降級（plan 維持 None）
        return _fallback(query, store)
    if not steps:                     # 合法空計劃：planner 判「無需工具」（含離題）
        out = _fallback(query, store)  # 仍走確定性路徑（保留 regex 訂單救援）
        out["plan"] = []              # ★ 覆蓋 _fallback 的 None，留住「planner 給空計劃」訊號
        return out
    facts = execute(steps, query, store)
    out = compose(query, facts)
    out["plan"] = steps
    return out
