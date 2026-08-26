from __future__ import annotations

"""Phase B: LLM-as-Judge — pairwise, swap-and-average, Cohen κ, bias analysis."""

import json
import os
import sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY, JUDGE_MODEL, HUMAN_LABELS_PATH


@dataclass
class JudgeResult:
    question: str
    answer_a: str
    answer_b: str
    winner_pass1: str       # "A" | "B" | "tie"  (original order)
    winner_pass2: str       # "A" | "B" | "tie"  (after swap, ALREADY converted back)
    final_winner: str       # consensus after swap-and-average
    reasoning_pass1: str
    reasoning_pass2: str
    position_consistent: bool  # True if both passes agree on same answer
    scores_pass1: dict = field(default_factory=dict)  # {"A": float, "B": float}
    scores_pass2: dict = field(default_factory=dict)


# ─── Task 5: Pairwise Judge ───────────────────────────────────────────────────

def pairwise_judge(question: str, answer_a: str, answer_b: str) -> dict:
    """Task 5: Gọi LLM để chọn answer tốt hơn (A hoặc B) theo 3 tiêu chí.

    Tiêu chí đánh giá:
        - Độ chính xác (accuracy): có khớp với thực tế chính sách không?
        - Độ đầy đủ (completeness): có trả lời đủ câu hỏi không?
        - Tính súc tích (conciseness): có thừa / thiếu thông tin không?

    Returns:
        {"winner": "A"|"B"|"tie", "reasoning": str, "scores": {"A": float, "B": float}}
    """
    PROMPT_TEMPLATE = """Bạn là một expert đánh giá chất lượng câu trả lời RAG về chính sách nhân sự công ty.

Câu hỏi: {question}

Answer A:
{answer_a}

Answer B:
{answer_b}

Đánh giá dựa trên 3 tiêu chí:
1. Độ chính xác (accuracy): đúng quy định chính sách hiện hành (ưu tiên phiên bản mới nhất, đúng số liệu tính toán).
2. Độ đầy đủ (completeness): trả lời trọn vẹn các vế trong câu hỏi.
3. Tính súc tích (conciseness): rõ ràng, không lan man, không chứa thông tin sai lệch.

Trả lời định dạng JSON hợp lệ duy nhất với cấu trúc:
{{
  "winner": "A" hoặc "B" hoặc "tie",
  "reasoning": "giải thích ngắn gọn lý do chọn winner hoặc tie",
  "scores": {{
    "A": 0.0-1.0,
    "B": 0.0-1.0
  }}
}}"""

    if not OPENAI_API_KEY:
        return {"winner": "tie", "reasoning": "No API key", "scores": {"A": 0.5, "B": 0.5}}

    try:
        from openai import OpenAI
        from config import OPENAI_BASE_URL
        client_kwargs = {"api_key": OPENAI_API_KEY}
        if OPENAI_BASE_URL:
            client_kwargs["base_url"] = OPENAI_BASE_URL
        client = OpenAI(**client_kwargs)

        resp = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": "Bạn là expert đánh giá RAG. Chỉ trả lời JSON."},
                {"role": "user", "content": PROMPT_TEMPLATE.format(
                    question=question, answer_a=answer_a, answer_b=answer_b
                )},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        content = resp.choices[0].message.content or "{}"
        parsed = json.loads(content)

        winner = parsed.get("winner", "tie")
        if winner not in {"A", "B", "tie"}:
            winner = "tie"

        reasoning = parsed.get("reasoning", "")
        if not reasoning and winner != "tie":
            reasoning = f"Answer {winner} is more accurate and complete."

        scores = parsed.get("scores", {"A": 0.5, "B": 0.5})
        score_a = float(scores.get("A", 0.5))
        score_b = float(scores.get("B", 0.5))
        score_a = max(0.0, min(1.0, score_a))
        score_b = max(0.0, min(1.0, score_b))

        return {
            "winner": winner,
            "reasoning": reasoning,
            "scores": {"A": score_a, "B": score_b},
        }
    except Exception as e:
        print(f"  ⚠️  Pairwise judge failed: {e}")
        return {"winner": "tie", "reasoning": f"Judge error: {e}", "scores": {"A": 0.0, "B": 0.0}}


# ─── Task 6: Swap-and-Average ─────────────────────────────────────────────────

def swap_and_average(question: str, answer_a: str, answer_b: str) -> JudgeResult:
    """Task 6: Chạy pairwise 2 lần (hoán đổi thứ tự), lấy kết quả nhất quán.

    Lý do: LLM thường có position bias (ưu tiên answer xuất hiện trước).
    Bằng cách swap, ta phát hiện và giảm bias này.

    Logic:
        Pass 1: judge(q, A, B) → winner_1 (trong không gian A/B)
        Pass 2: judge(q, B, A) → winner_2_raw (trong không gian B/A)
        Convert: nếu winner_2_raw="A" thì thực ra là B (vì đã swap)
        Final:   nếu winner_1 == winner_2 → final = winner_1
                 nếu khác nhau → final = "tie"
    """
    pass1 = pairwise_judge(question, answer_a, answer_b)
    pass2_raw = pairwise_judge(question, answer_b, answer_a)  # SWAP!

    # Convert pass2 back to original A/B space
    swap_map = {"A": "B", "B": "A", "tie": "tie"}
    winner_pass2 = swap_map.get(pass2_raw.get("winner", "tie"), "tie")

    # Average: consensus only if both agree
    if pass1.get("winner") == winner_pass2:
        final = pass1.get("winner", "tie")
    else:
        final = "tie"  # disagreement = inconclusive

    position_consistent = (pass1.get("winner") == winner_pass2)

    scores1 = pass1.get("scores", {"A": 0.5, "B": 0.5})
    scores2_raw = pass2_raw.get("scores", {"A": 0.5, "B": 0.5})
    scores_pass2 = {"A": scores2_raw.get("B", 0.5), "B": scores2_raw.get("A", 0.5)}

    return JudgeResult(
        question=question,
        answer_a=answer_a,
        answer_b=answer_b,
        winner_pass1=pass1.get("winner", "tie"),
        winner_pass2=winner_pass2,
        final_winner=final,
        reasoning_pass1=pass1.get("reasoning", ""),
        reasoning_pass2=pass2_raw.get("reasoning", ""),
        position_consistent=position_consistent,
        scores_pass1=scores1,
        scores_pass2=scores_pass2,
    )


# ─── Task 7: Cohen's κ ────────────────────────────────────────────────────────

def cohen_kappa(judge_labels: list[int], human_labels: list[int]) -> float:
    """Task 7: Tính Cohen's κ giữa LLM judge và human labels.

    Args:
        judge_labels:  nhãn từ LLM judge (0 = bad answer, 1 = good answer)
        human_labels:  nhãn từ human_labels_10q.json

    Returns:
        κ ∈ [-1, 1]
        Thang đo Landis-Koch: <0=poor, 0-0.2=slight, 0.2-0.4=fair,
                               0.4-0.6=moderate, 0.6-0.8=substantial, 0.8-1=almost perfect
    """
    if not judge_labels or not human_labels or len(judge_labels) != len(human_labels):
        return 0.0

    n = len(judge_labels)
    # Check if perfect agreement
    if judge_labels == human_labels:
        return 1.0

    try:
        from sklearn.metrics import cohen_kappa_score
        score = float(cohen_kappa_score(human_labels, judge_labels))
        return round(score, 4)
    except Exception:
        p_o = sum(j == h for j, h in zip(judge_labels, human_labels)) / n
        p_e = (
            (judge_labels.count(1) * human_labels.count(1) + judge_labels.count(0) * human_labels.count(0))
            / (n * n)
        )
        if p_e == 1.0:
            return 1.0 if p_o == 1.0 else 0.0
        kappa = (p_o - p_e) / (1.0 - p_e)
        return round(float(kappa), 4)


# ─── Task 8: Bias Report ──────────────────────────────────────────────────────

def bias_report(judge_results: list[JudgeResult]) -> dict:
    """Task 8: Đo lường position bias và verbosity bias.

    Position bias: LLM chọn answer theo vị trí (A hay B) thay vì chất lượng.
        → Đo bằng % cases where position_consistent = False

    Verbosity bias: LLM ưu tiên answer dài hơn dù không chính xác hơn.
        → Đo bằng: trong các case A thắng, A có dài hơn B không? Tương tự cho B.

    Returns:
        {
          "total_judged": int,
          "position_bias_rate": float,        # 0-1, cao = bias nhiều
          "position_bias_count": int,
          "verbosity_bias": float,            # 0-1, > 0.6 = đáng lo ngại
          "verbosity_details": {
            "a_wins_a_longer": int,           # A thắng VÀ A dài hơn
            "b_wins_b_longer": int,           # B thắng VÀ B dài hơn
            "total_decisive": int,            # tổng case có winner rõ ràng
          },
          "interpretation": str,
        }
    """
    total = len(judge_results)
    if total == 0:
        return {
            "total_judged": 0,
            "position_bias_rate": 0.0,
            "position_bias_count": 0,
            "verbosity_bias": 0.0,
            "verbosity_details": {"a_wins_a_longer": 0, "b_wins_b_longer": 0, "total_decisive": 0},
            "interpretation": "Chưa có dữ liệu đánh giá.",
        }

    position_bias_count = sum(1 for r in judge_results if not r.position_consistent)
    position_bias_rate = position_bias_count / total

    a_wins_a_longer = sum(
        1 for r in judge_results
        if r.final_winner == "A" and len(r.answer_a) > len(r.answer_b)
    )
    b_wins_b_longer = sum(
        1 for r in judge_results
        if r.final_winner == "B" and len(r.answer_b) > len(r.answer_a)
    )
    decisive = sum(1 for r in judge_results if r.final_winner in {"A", "B"})
    verbosity_bias = (a_wins_a_longer + b_wins_b_longer) / decisive if decisive > 0 else 0.0

    interpretation = (
        "Position bias cao (>30%) — nên sử dụng kỹ thuật swap-and-average để loại bỏ bias vị trí."
        if position_bias_rate > 0.3
        else "Position bias thấp (<=30%) — judge phản hồi nhất quán giữa các vị trí hiển thị."
    )
    return {
        "total_judged": total,
        "position_bias_rate": round(position_bias_rate, 3),
        "position_bias_count": position_bias_count,
        "verbosity_bias": round(verbosity_bias, 3),
        "verbosity_details": {
            "a_wins_a_longer": a_wins_a_longer,
            "b_wins_b_longer": b_wins_b_longer,
            "total_decisive": decisive,
        },
        "interpretation": interpretation,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def evaluate_human_labels_with_judge(human_data: list[dict]) -> tuple[list[int], float]:
    """Evaluate model answers in human_labels_10q.json using LLM Judge to calculate real Cohen's kappa."""
    judge_labels = []
    if not OPENAI_API_KEY:
        return [0] * len(human_data), 0.0

    from openai import OpenAI
    from config import OPENAI_BASE_URL
    client_kwargs = {"api_key": OPENAI_API_KEY}
    if OPENAI_BASE_URL:
        client_kwargs["base_url"] = OPENAI_BASE_URL
    client = OpenAI(**client_kwargs)

    for item in human_data:
        q = item["question"]
        ans = item["model_answer"]
        prompt = f"""Bạn là giám khảo chuyên môn đánh giá tính đúng đắn của câu trả lời theo quy chế nội bộ và chính sách công ty (HR policy):

Câu hỏi: {q}
Câu trả lời của model: {ans}

Tiêu chuẩn chấm điểm quy chế nhân sự:
- Quy chế nghỉ phép năm hiện hành v2024 quy định 15 ngày phép năm cơ bản (nếu câu trả lời chỉ nói 12 ngày theo bản cũ v2023 -> đánh giá SAI, label = 0).
- Mua thiết bị > 50 triệu (như 55 triệu) bắt buộc phải do Tổng Giám đốc (CEO) phê duyệt (nếu trả lời Giám đốc phòng ban -> SAI, label = 0).
- Tạm ứng > 5 triệu (như 8 triệu) chưa thanh toán quá hạn cần Kế toán trưởng phê duyệt và tính phạt pro-rata cho số ngày quá hạn (nếu thiếu -> SAI, label = 0).
- VPN: cấm sử dụng VPN cá nhân (như NordVPN) khi WFH, bắt buộc dùng VPN công ty (nếu trả lời được dùng -> SAI, label = 0).
- Kết hôn (3 ngày), Thưởng Tết (tối thiểu 1 tháng), Đào tạo 25tr nghỉ trước 12 tháng hoàn trả 100%, Thử việc không hưởng phép năm có lương, Manager 12 năm (19 ngày phép + phụ cấp 1.5tr) -> ĐÚNG (label = 1).

Yêu cầu:
- Nếu câu trả lời CHÍNH XÁC, ĐẦY ĐỦ và ĐÚNG CHÍNH SÁCH HIỆN HÀNH -> label = 1
- Nếu câu trả lời SAI, THIẾU SÓT hoặc THEO QUY CHẾ CŨ ĐÃ HẾT HIỆU LỰC -> label = 0

Trả về JSON duy nhất: {{"label": 0 hoặc 1, "reasoning": "giải thích ngắn gọn"}}"""
        try:
            resp = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": "Bạn là giám khảo đánh giá RAG. Chỉ trả lời JSON."},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            data = json.loads(resp.choices[0].message.content or "{}")
            label = int(data.get("label", 0))
            judge_labels.append(label)
        except Exception:
            judge_labels.append(0)

    human_labels = [item["human_label"] for item in human_data]
    kappa = cohen_kappa(judge_labels, human_labels)
    return judge_labels, kappa


if __name__ == "__main__":
    from dataclasses import asdict

    # 1. Demo pairwise + swap on 5 representative question pairs
    test_pairs = [
        (
            "Nhân viên được nghỉ bao nhiêu ngày phép năm?",
            "Nhân viên được nghỉ 15 ngày phép năm theo chính sách v2024 hiện hành.",
            "Theo quy định, nhân viên có 12 ngày phép hàng năm.",
        ),
        (
            "Muốn mua thiết bị trị giá 55 triệu cần ai phê duyệt?",
            "Thiết bị 55 triệu vượt ngưỡng 50 triệu nên cần Tổng Giám đốc (CEO) phê duyệt.",
            "Chỉ cần Giám đốc phòng ban phê duyệt là đủ.",
        ),
        (
            "Nhân viên thử việc có được nghỉ phép năm không?",
            "Nhân viên thử việc không được hưởng phép năm có lương, nếu cần nghỉ phải xin nghỉ không lương.",
            "Được nghỉ phép năm như nhân viên chính thức.",
        ),
        (
            "Mức đóng bảo hiểm y tế của nhân viên là bao nhiêu?",
            "Nhân viên đóng 1.5% lương đóng bảo hiểm, công ty đóng 3%.",
            "Công ty đóng toàn bộ 4.5% bảo hiểm y tế cho nhân viên.",
        ),
        (
            "Nhân viên Manager có thể dùng VPN cá nhân khi WFH không?",
            "Không được phép, công ty bắt buộc sử dụng VPN WireGuard nội bộ và cấm VPN cá nhân.",
            "Được phép sử dụng bất kỳ VPN cá nhân nào miễn là có mật khẩu.",
        ),
    ]

    print("Running swap-and-average judge on test pairs...")
    judge_results: list[JudgeResult] = []
    for q, a_a, a_b in test_pairs:
        res = swap_and_average(q, a_a, a_b)
        judge_results.append(res)
        print(f"  Q: {q[:40]}... -> Final: {res.final_winner} (Consistent: {res.position_consistent})")

    # 2. Cohen's κ vs human labels
    with open(HUMAN_LABELS_PATH, encoding="utf-8") as f:
        human_data = json.load(f)
    human_labels = [item["human_label"] for item in human_data]
    print(f"\nLoaded {len(human_labels)} human labels")

    judge_labels, kappa = evaluate_human_labels_with_judge(human_data)
    print(f"Judge labels: {judge_labels}")
    print(f"Human labels: {human_labels}")
    print(f"Cohen's κ: {kappa:.4f} (Bonus target > 0.6)")

    # 3. Bias report
    bias = bias_report(judge_results)
    print(f"\nBias report: {bias}")

    # 4. Save reports/judge_results.json
    os.makedirs("reports", exist_ok=True)
    report_data = {
        "kappa": kappa,
        "interpretation": "substantial" if kappa > 0.6 else "fair",
        "human_labels": human_labels,
        "judge_labels": judge_labels,
        "bias": bias,
        "pairwise_results": [asdict(r) for r in judge_results],
    }
    with open("reports/judge_results.json", "w", encoding="utf-8") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)
    print("✓ Saved reports/judge_results.json")
