# -*- coding: utf-8 -*-
"""純函式層的單元測試 —— 不呼叫 Ollama、不開 Chroma。
凡是被推到程式那一側的東西，都能離線、確定性地驗證。
執行：   pytest            （repo 根目錄；pytest.ini 已設 pythonpath=src）
"""
from datetime import date

import pytest

from agent import (
    ORDER_ID_RE,
    MAX_STEPS,
    _claims_not_received,
    check_return_window,
    validate_plan,
)


# ======================================================================
# 1. check_return_window —— 可稽核數字的來源，錯一天就是客訴
# ======================================================================

def test_return_window_within_period():
    """簽收日 + 14 天，今天在期限內 → eligible=True 且 days_left 為正。
    順便確認 deadline 字串是 'YYYY-MM-DD'（它會直接被印進客人看到的回覆）。"""
    # 1. Arrange: 設定固定基準日 (today)、簽收日 (在 14 天期限內)、視窗天數
    today = date(2026, 9, 6)
    signoff = "2026-09-01"
    window_days = 14

    # 2. Act: 計算退貨視窗
    result = check_return_window(signoff, window_days, today=today)

    # 3. Assert: 驗證 eligible (True)、days_left (正數/正確天數)、deadline 格式與日期
    assert result["eligible"] is True
    assert result["days_left"] == 9
    assert result["deadline"] == "2026-09-15"


def test_return_window_on_deadline_day():
    """邊界：今天 == deadline。
    程式判定是 days_left >= 0，所以當天應仍 eligible=True、days_left == 0。
    這題是整份測試最重要的一題 —— off-by-one 會讓客人在最後一天被拒退。"""
    # 1. Arrange: 設定今天剛好等於 deadline (簽收日 + 14天)
    signoff = "2026-09-01"
    window_days = 14
    today = date(2026, 9, 15)

    # 2. Act
    result = check_return_window(signoff, window_days, today=today)

    # 3. Assert: 驗證當天仍符合資格 (eligible=True) 且剩餘天數剛好為 0
    assert result["eligible"] is True
    assert result["days_left"] == 0


def test_return_window_expired():
    """已逾期 → eligible=False 且 days_left 為負。
    負值本身是對的（compose 會轉成「已逾期 X 天」），不要在這裡把它夾成 0。"""
    # 1. Arrange: 設定今天已超過 deadline
    signoff = "2026-09-01"
    window_days = 14
    today = date(2026, 9, 20)

    # 2. Act
    result = check_return_window(signoff, window_days, today=today)

    # 3. Assert: 驗證 eligible=False 且 days_left < 0 (保留真實負數)
    assert result["eligible"] is False
    assert result["days_left"] == -5
    assert result["deadline"] == "2026-09-15"


def test_return_window_defect_uses_same_function():
    """瑕疵品 30 天走的是同一個函式、只是 window_days 不同。
    證明 window_days 是外部傳入的參數而非常數"""
    # 1. Arrange: 瑕疵品政策 window_days = 30，設定在第 20 天退貨
    signoff = "2026-09-01"
    window_days = 30
    today = date(2026, 9, 21)

    # 2. Act
    result = check_return_window(signoff, window_days, today=today)

    # 3. Assert: 驗證 30 天視窗下的計算結果 (一般退貨已過，但瑕疵退貨仍 eligible)
    assert result["eligible"] is True
    assert result["days_left"] == 10
    assert result["deadline"] == "2026-10-01"


def test_return_window_rejects_bad_date_format():
    """壞格式的 signoff_date 應 raise ValueError，而不是安靜地算出錯的日期。
    提示：pytest.raises(ValueError)。"""
    # 1. Arrange: 準備非 'YYYY-MM-DD' 格式的字串 (例如 "2026/09/01" 或 "bad-date")
    bad_date = "2026/09/01"

    # 2. Act & Assert: 使用 pytest.raises 捕捉 ValueError
    with pytest.raises(ValueError):
        check_return_window(bad_date, 14)


# ======================================================================
# 2. ORDER_ID_RE —— 這組是回歸測試，對應你實際踩過的 \b 與 CJK 的坑
# ======================================================================

def test_order_id_matches_when_glued_to_chinese():
    """「訂單ML240003到哪了」必須抓得到。
    原本寫成 \\bML\\d{6}\\b 時全數失敗 —— 因為 Python 把中日韓字視為 \\w，
    中文黏著時左右都不構成 word boundary。"""
    # 1. Arrange: 中文前後緊密黏著訂單號
    text = "訂單ML240003到哪了"

    # 2. Act: 進行正則搜尋
    match = ORDER_ID_RE.search(text)

    # 3. Assert: 驗證成功命中且抓出的訂單號正確
    assert match is not None
    assert match.group(0) == "ML240003"


def test_order_id_case_insensitive():
    """小寫 ml240003 也要抓得到（re.I）。"""
    # 1. Arrange: 小寫訂單號
    text = "查詢 ml240003 狀態"

    # 2. Act
    match = ORDER_ID_RE.search(text)

    # 3. Assert: 驗證不分大小寫皆能匹配
    assert match is not None
    assert match.group(0).upper() == "ML240003"


def test_order_id_rejects_alphanumeric_neighbours():
    """「XML2400031」不該被當成訂單編號 —— 前後接英數字要擋掉。
    這是 (?<![A-Za-z0-9]) / (?![A-Za-z0-9]) 存在的理由。"""
    # 1. Arrange: 前後黏著英文字母或額外數字的無效編號
    invalid_cases = ["XML240003", "ML240003A", "XML2400031"]

    # 2. Act & Assert: 驗證所有案例皆不該被判定為有效訂單
    for case in invalid_cases:
        assert ORDER_ID_RE.search(case) is None


def test_order_id_rejects_wrong_digit_count():
    """位數不是 6 位就不該命中（例如 ML24000、ML2400034）。"""
    # 1. Arrange: 位數不足 (5位) 或多於 (7位) 6位數的案例
    wrong_digits = ["ML24000", "ML2400034"]

    # 2. Act & Assert
    for case in wrong_digits:
        assert ORDER_ID_RE.search(case) is None


# ======================================================================
# 3. _claims_not_received —— 含一題「誠實記錄的已知失敗」
# ======================================================================

@pytest.mark.parametrize("q", [
    "我還沒收到包裹",
    "東西沒送到欸",
    "I haven't received my order",
])
def test_claims_not_received_positive(q):
    """各種「沒收到」的說法都要命中。"""
    # 1. Act: 呼叫 _claims_not_received
    # 2. Assert: 驗證命中為 True
    assert _claims_not_received(q) is True


def test_claims_not_received_negative():
    """一般查件（如「ML240003 到哪了?」）不該被誤判成爭議 —— 誤報會讓正常訂單也被轉真人。"""
    # 1. Arrange: 一般詢問句（無未收到/抱怨字眼）
    query = "ML240003 到哪了?"

    # 2. Act & Assert: 驗證未命中 (False)
    assert _claims_not_received(query) is False


@pytest.mark.xfail(reason="已知限制：關鍵詞比對抓不到換句話說，正確修法是改語意判斷", strict=True)
def test_claims_not_received_paraphrase_known_gap():
    """「shows delivered but I got nothing」目前抓不到 —— 關鍵詞比對的已知限制（換句話說抓不到）。

    這題刻意標成 xfail(strict=True)：
      - 現在它「預期失敗」，所以 CI 是綠的，但限制被寫在程式裡而不是只寫在文件裡；
      - 未來真的改成語意判斷、它開始通過時，strict=True 會讓這題以 XPASS 報錯，
        必須回來把 xfail 拿掉。
    """
    # 1. Arrange: 換句話說的抱怨句（已知 regex 限制）
    query = "shows delivered but I got nothing"

    # 2. Act & Assert: 預期它應該為 True (但目前實作會是 False，故 xfail 會通過)
    assert _claims_not_received(query) is True


# ======================================================================
# 4. validate_plan —— 不呼叫 LLM，直接餵假 plan 驗防護層
# ======================================================================

def test_validate_plan_rejects_unknown_tool():
    """計劃裡出現白名單外的工具 → 整份計劃作廢（回 None），交給確定性降級。"""
    # 1. Arrange: 包含不在 ALLOWED_TOOLS 白名單的步驟
    bad_plan = {"steps": [{"tool": "unauthorized_tool", "args": {}}]}

    # 2. Act
    result = validate_plan(bad_plan, query="查詢訂單")

    # 3. Assert: 驗證整份計劃被作廢 (回傳 None)
    assert result is None


def test_validate_plan_handles_none():
    """plan() 回 None（JSON 解析失敗）時不該炸掉。"""
    # 1. Act: 傳入 None 或壞 dict
    result = validate_plan(None, query="任意查詢")

    # 2. Assert: 驗證安全回傳 None
    assert result is None


def test_validate_plan_coerces_bare_string_step():
    """回歸測試：模型曾經回 {"steps": ["get_order_status"]}（只給工具名、不是 dict），
    當時直接 AttributeError: 'str' object has no attribute 'get'。
    現在應被補成 {"tool": ..., "args": {}} 而非拋例外。"""
    # 1. Arrange: steps 內為純字串，且 query 內含合法訂單號
    bare_plan = {"steps": ["get_order_status"]}
    query = "ML240003 到哪了"

    # 2. Act
    result = validate_plan(bare_plan, query)

    # 3. Assert: 驗證被標準化為 dict 格式且自動從 query 補上 order_id
    assert result == [{"tool": "get_order_status", "args": {"order_id": "ML240003"}}]


def test_validate_plan_drops_garbage_step():
    """步驟是數字、None 這類非預期型別 → 丟掉那一步，其餘保留。"""
    # 1. Arrange: steps 混雜非 dict / 非 str 的型別，但包含合法步驟
    garbage_plan = {"steps": [123, None, {"tool": "lookup_policy", "args": {"sub_query": "退貨政策"}}]}

    # 2. Act
    result = validate_plan(garbage_plan, query="想了解退貨")

    # 3. Assert: 驗證垃圾步驟被過濾，只留下合法的 policy 步驟
    assert len(result) == 1
    assert result[0]["tool"] == "lookup_policy"


def test_validate_plan_truncates_to_max_steps():
    """超過 MAX_STEPS 的計劃要被截斷 —— 這是成本與延遲的上限保護。"""
    # 1. Arrange: 構造超過 MAX_STEPS 步數的計劃
    long_plan = {"steps": [{"tool": "lookup_policy", "args": {"sub_query": f"問題{i}"}} for i in range(MAX_STEPS + 3)]}

    # 2. Act
    result = validate_plan(long_plan, query="多個問題")

    # 3. Assert: 驗證步驟數被截斷為 MAX_STEPS
    assert len(result) == MAX_STEPS


def test_validate_plan_regex_overrides_model_order_id():
    """『參數即真相』：模型把 order_id 抄錯時，以 query 裡 regex 抽到的為準。
    模型負責判斷「要不要查訂單」，不負責記住編號。"""
    # 1. Arrange: 模型幻覺/抄錯的 order_id，但 query 裡有真實的 order_id
    hallucinated_plan = {"steps": [{"tool": "get_order_status", "args": {"order_id": "ML999999"}}]}
    query = "幫我查 ML240003"

    # 2. Act
    result = validate_plan(hallucinated_plan, query)

    # 3. Assert: 驗證 order_id 被 regex 抽到的真實編號 (ML240003) 覆蓋
    assert result[0]["args"]["order_id"] == "ML240003"


def test_validate_plan_injects_order_step_on_dispute():
    """確定性注入：query 同時滿足「有真編號」+「客人說沒收到」，
    但模型的計劃沒有 get_order_status → 應被強制插在第一步。

    這條規則存在的理由：早期是靠在 PLANNER_SYSTEM 加規則修的，
    結果修好一題、另一題迴歸（打地鼠）。改成程式層強制注入後才穩定。"""
    # 1. Arrange: query 有訂單號且表示沒收到，但模型只規劃了 lookup_policy
    missed_order_plan = {"steps": [{"tool": "lookup_policy", "args": {"sub_query": "沒收到包裹"}}]}
    query = "ML240003 我還沒收到包裹"

    # 2. Act
    result = validate_plan(missed_order_plan, query)

    # 3. Assert: 驗證 get_order_status 被強制注入在第 0 步 (第一步)
    assert result[0]["tool"] == "get_order_status"
    assert result[0]["args"]["order_id"] == "ML240003"
    assert len(result) == 2
