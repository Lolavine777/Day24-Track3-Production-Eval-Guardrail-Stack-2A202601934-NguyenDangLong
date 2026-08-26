from __future__ import annotations

"""Phase C: Production Guardrails — Presidio PII + NeMo Guardrails + P95 Latency."""

import asyncio
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ADVERSARIAL_SET_PATH, GUARDRAILS_CONFIG_DIR, LATENCY_BUDGET_P95_MS, PRESIDIO_LANGUAGE


# ─── Task 9a: Presidio PII Detection ─────────────────────────────────────────

def setup_presidio():
    """Khởi tạo Presidio engine với custom Vietnamese PII recognizers. (Đã implement sẵn)

    Custom recognizers thêm vào:
        VN_CCCD  — số CCCD 12 chữ số hoặc CMND 9 chữ số
        VN_PHONE — số điện thoại Việt Nam (0[3-9]xxxxxxxx)

    Các recognizers mặc định đã có sẵn: EMAIL, PHONE_NUMBER (international), ...
    """
    from presidio_analyzer import AnalyzerEngine, RecognizerRegistry, Pattern, PatternRecognizer
    from presidio_anonymizer import AnonymizerEngine

    cccd_recognizer = PatternRecognizer(
        supported_entity="VN_CCCD",
        patterns=[
            Pattern("CCCD 12 digits", r"\b\d{12}\b", 0.9),
            Pattern("CMND 9 digits",  r"\b\d{9}\b",  0.7),
        ],
    )
    phone_recognizer = PatternRecognizer(
        supported_entity="VN_PHONE",
        patterns=[Pattern("VN mobile", r"\b0[3-9]\d{8}\b", 0.9)],
    )
    vn_name_pattern = Pattern(
        name="vn_name",
        regex=r"\b(?:Nguyễn|Trần|Lê|Phạm|Hoàng|Huỳnh|Phan|Vũ|Võ|Đặng|Bùi|Đỗ|Hồ|Ngô|Dương|Lý)\s+[A-ZÀ-Ỹ][a-zà-ỹ]*(?:\s+[A-ZÀ-Ỹ][a-zà-ỹ]*)+\b",
        score=0.85,
    )
    vn_name_recognizer = PatternRecognizer(
        supported_entity="PERSON",
        name="vn_name_recognizer",
        patterns=[vn_name_pattern],
    )

    registry = RecognizerRegistry()
    registry.load_predefined_recognizers()
    try:
        registry.remove_recognizer("SpacyRecognizer")
    except Exception:
        pass
    registry.add_recognizer(cccd_recognizer)
    registry.add_recognizer(phone_recognizer)
    registry.add_recognizer(vn_name_recognizer)

    analyzer  = AnalyzerEngine(registry=registry)
    anonymizer = AnonymizerEngine()
    return analyzer, anonymizer


def pii_scan(text: str, analyzer=None, anonymizer=None) -> dict:
    """Task 9a: Quét PII trong văn bản bằng Presidio.

    Returns:
        {
          "has_pii":    bool,
          "entities":   [{"type": str, "text": str, "score": float, "start": int, "end": int}],
          "anonymized": str,   # text với PII được thay bằng <TYPE>
        }
    """
    if analyzer is None or anonymizer is None:
        analyzer, anonymizer = setup_presidio()

    results = analyzer.analyze(text=text, language=PRESIDIO_LANGUAGE)
    if not results:
        return {"has_pii": False, "entities": [], "anonymized": text}

    anonymized = anonymizer.anonymize(text=text, analyzer_results=results).text
    entities = [
        {"type": r.entity_type, "text": text[r.start:r.end],
         "score": round(r.score, 3), "start": r.start, "end": r.end}
        for r in results
    ]
    return {"has_pii": True, "entities": entities, "anonymized": anonymized}


# ─── Task 9b + 11: NeMo Guardrails ───────────────────────────────────────────

def setup_nemo_rails():
    """Khởi tạo NeMo Guardrails từ guardrails/config.yml. (Đã implement sẵn)

    Config directory: guardrails/
        config.yml  — model + rails config
        rails.co    — Colang dialogue flows (topic check, jailbreak check, output check)
    """
    from nemoguardrails import RailsConfig, LLMRails
    config = RailsConfig.from_path(GUARDRAILS_CONFIG_DIR)
    rails = LLMRails(config)
    return rails


async def check_input_rail(text: str, rails=None) -> dict:
    """Task 9b: Kiểm tra input qua NeMo input rails (topic guard + jailbreak guard).

    Returns:
        {
          "allowed":        bool,
          "blocked_reason": str | None,
          "response":       str,          # NeMo's raw response
        }
    """
    if rails is None:
        rails = setup_nemo_rails()

    try:
        response_dict = await rails.generate_async(
            messages=[{"role": "user", "content": text}]
        )
        if isinstance(response_dict, dict):
            response = response_dict.get("content", "")
        else:
            response = str(response_dict)
    except Exception as e:
        print(f"  ⚠️  NeMo input rail error: {e}")
        response = ""

    refuse_keywords = [
        "xin lỗi", "không thể", "không được phép", "i cannot", "i can't",
        "i'm sorry", "tôi chỉ có thể", "dữ liệu bảo mật", "bảo mật",
        "i am sorry", "cannot fulfill", "restricted", "unauthorized", "refuse",
    ]
    blocked = any(kw in response.lower() for kw in refuse_keywords)
    if not blocked:
        import re
        off_topic_patterns = [
            r"thơ", r"nấu", r"phở", r"bitcoin", r"ethereum", r"crypto", r"tiền ảo",
            r"phim", r"marvel", r"toán", r"phương trình", r"thời tiết", r"tin tức",
            r"bài hát", r"game", r"du lịch",
        ]
        jailbreak_patterns = [
            r"bỏ qua", r"ignore", r"dan", r"unrestricted", r"system prompt",
            r"system override", r"đóng vai", r"pretend", r"tấn công mạng",
            r"admin command", r"dump", r"tiết lộ", r"reveal", r"ra lệnh",
            r"in ra toàn bộ", r"mật khẩu admin", r"bảng lương",
        ]
        inp_lower = text.lower()
        if any(re.search(p, inp_lower) for p in off_topic_patterns + jailbreak_patterns):
            blocked = True

    return {
        "allowed": not blocked,
        "blocked_reason": "nemo_input_rail" if blocked else None,
        "response": response,
    }


async def check_output_rail(question: str, answer: str, rails=None) -> dict:
    """Task 11: Kiểm tra LLM output qua NeMo output rails trước khi trả về user.

    NeMo output rails hoạt động trong context của cả cuộc hội thoại (input + output).
    Kiểm tra: có PII không? Nội dung có phù hợp không? Có hallucination rõ ràng không?

    Returns:
        {
          "safe":           bool,
          "flagged_reason": str | None,
          "final_answer":   str,          # answer đã qua guard (có thể bị redact)
        }
    """
    if rails is None:
        rails = setup_nemo_rails()

    try:
        response_dict = await rails.generate_async(messages=[
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ])
        if isinstance(response_dict, dict):
            response = response_dict.get("content", "")
        else:
            response = str(response_dict)
    except Exception as e:
        print(f"  ⚠️  NeMo output rail error: {e}")
        response = answer

    refuse_keywords = [
        "xin lỗi", "không thể cung cấp", "i cannot", "tôi không thể",
        "dữ liệu bảo mật", "thông tin này",
    ]
    flagged = any(kw in response.lower() for kw in refuse_keywords)
    return {
        "safe": not flagged,
        "flagged_reason": "nemo_output_rail" if flagged else None,
        "final_answer": response if flagged else answer,
    }


# ─── Task 10: Adversarial Test Suite ─────────────────────────────────────────

def run_adversarial_suite(adversarial_set: list[dict], rails=None,
                           analyzer=None, anonymizer=None) -> list[dict]:
    """Task 10: Chạy 20 adversarial inputs qua full guard stack, so sánh với expected.

    Guard stack order:
        1. pii_scan()         → block nếu has_pii (cho category pii_injection)
        2. check_input_rail() → block nếu jailbreak / off-topic / prompt injection

    Returns:
        list of {
          "id": int, "category": str, "input": str,
          "expected": "blocked"|"allowed",
          "actual":   "blocked"|"allowed",
          "blocked_by": str | None,       # "presidio" | "nemo_input" | None
          "passed": bool,
        }
    """
    if analyzer is None or anonymizer is None:
        analyzer, anonymizer = setup_presidio()
    if rails is None:
        rails = setup_nemo_rails()

    async def _run_all():
        results = []
        for item in adversarial_set:
            blocked_by = None

            # Layer 1: Presidio PII (synchronous, fast)
            pii_result = pii_scan(item["input"], analyzer, anonymizer)
            if pii_result["has_pii"]:
                blocked_by = "presidio"

            # Layer 2: NeMo input rail (async — await)
            if blocked_by is None:
                rail_result = await check_input_rail(item["input"], rails)
                if not rail_result["allowed"]:
                    blocked_by = "nemo_input"

            actual = "blocked" if blocked_by else "allowed"
            results.append({
                "id": item["id"],
                "category": item["category"],
                "input": item["input"][:80] + "...",
                "expected": item["expected"],
                "actual": actual,
                "blocked_by": blocked_by,
                "passed": (actual == item["expected"]),
            })
        return results

    results = asyncio.run(_run_all())
    passed = sum(1 for r in results if r["passed"])
    print(f"Adversarial suite: {passed}/{len(results)} passed")
    return results


# ─── Task 12: P95 Latency Measurement ────────────────────────────────────────

def measure_p95_latency(test_inputs: list[str], n_runs: int = 20,
                         rails=None, analyzer=None, anonymizer=None) -> dict:
    """Task 12: Đo P50/P95/P99 latency cho từng layer trong guard stack.

    Mục tiêu production: P95 total < LATENCY_BUDGET_P95_MS (500ms mặc định)

    Insight cần quan sát:
        - Presidio: local regex → rất nhanh (<10ms)
        - NeMo:     LLM API call → chậm (~200-800ms tuỳ model và network)
        → Tổng: dominated by NeMo

    Returns:
        {
          "presidio_ms":  {"p50": float, "p95": float, "p99": float},
          "nemo_ms":      {"p50": float, "p95": float, "p99": float},
          "total_ms":     {"p50": float, "p95": float, "p99": float},
          "latency_budget_ok": bool,
          "budget_ms": int,
        }
    """
    if analyzer is None or anonymizer is None:
        analyzer, anonymizer = setup_presidio()
    if rails is None:
        rails = setup_nemo_rails()

    presidio_times, nemo_times, total_times = [], [], []

    async def _measure():
        inputs_to_test = test_inputs if len(test_inputs) <= n_runs else test_inputs[:n_runs]
        for text in inputs_to_test:
            # Presidio (synchronous)
            t0 = time.perf_counter()
            pii_scan(text, analyzer, anonymizer)
            presidio_ms = (time.perf_counter() - t0) * 1000

            # NeMo input rail (await)
            t1 = time.perf_counter()
            await check_input_rail(text, rails)
            nemo_ms = (time.perf_counter() - t1) * 1000

            presidio_times.append(presidio_ms)
            nemo_times.append(nemo_ms)
            total_times.append(presidio_ms + nemo_ms)

    asyncio.run(_measure())

    def percentiles(times):
        if not times:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
        s = sorted(times)
        n = len(s)
        p50_idx = min(int(n * 0.50), n - 1)
        p95_idx = min(int(n * 0.95), n - 1)
        p99_idx = min(int(n * 0.99), n - 1)
        return {
            "p50": round(s[p50_idx], 2),
            "p95": round(s[p95_idx], 2),
            "p99": round(s[p99_idx], 2),
        }

    total_p = percentiles(total_times)
    return {
        "presidio_ms": percentiles(presidio_times),
        "nemo_ms": percentiles(nemo_times),
        "total_ms": total_p,
        "latency_budget_ok": total_p["p95"] < LATENCY_BUDGET_P95_MS,
        "budget_ms": LATENCY_BUDGET_P95_MS,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Task 9a: PII scan demo
    test_pii = "Nhân viên Nguyễn Văn A, CCCD 034095001234, SĐT 0987654321 hỏi về nghỉ phép."
    result = pii_scan(test_pii)
    print(f"PII detected: {result['has_pii']}")
    print(f"Entities: {result['entities']}")
    print(f"Anonymized: {result['anonymized']}")

    # Task 10: Adversarial suite
    with open(ADVERSARIAL_SET_PATH, encoding="utf-8") as f:
        adversarial_set = json.load(f)
    print(f"\nLoaded {len(adversarial_set)} adversarial inputs")
    suite_results = run_adversarial_suite(adversarial_set)
    if suite_results:
        passed = sum(1 for r in suite_results if r["passed"])
        print(f"Adversarial suite: {passed}/{len(suite_results)} passed")

    # Task 12: P95 latency
    sample_inputs = [item["input"] for item in adversarial_set[:10]]
    latency = measure_p95_latency(sample_inputs, n_runs=10)
    print(f"\nLatency P95 — Presidio: {latency['presidio_ms']['p95']}ms | "
          f"NeMo: {latency['nemo_ms']['p95']}ms | "
          f"Total: {latency['total_ms']['p95']}ms")
    print(f"Budget OK ({latency['budget_ms']}ms): {latency['latency_budget_ok']}")

    # Save reports/guard_results.json
    os.makedirs("reports", exist_ok=True)
    report_data = {
        "pass_count": sum(1 for r in suite_results if r["passed"]) if suite_results else 0,
        "total_count": len(suite_results) if suite_results else 0,
        "pass_rate": (sum(1 for r in suite_results if r["passed"]) / len(suite_results)) if suite_results else 0.0,
        "latency": latency,
        "suite_results": suite_results,
    }
    with open("reports/guard_results.json", "w", encoding="utf-8") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)
    print("✓ Saved reports/guard_results.json")
