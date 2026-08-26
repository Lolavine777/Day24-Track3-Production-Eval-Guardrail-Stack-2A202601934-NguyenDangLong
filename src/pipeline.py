from __future__ import annotations

"""Production RAG Pipeline — Advanced Optimized: ghép M1+M2+M3+M4+M5."""

import os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.m1_chunking import load_documents, chunk_hierarchical
from src.m2_search import HybridSearch
from src.m3_rerank import CrossEncoderReranker
from src.m4_eval import load_test_set, evaluate_ragas, failure_analysis, save_report
from src.m5_enrichment import enrich_chunks
from config import RERANK_TOP_K, OPENAI_API_KEY, OPENAI_MODEL, OPENAI_BASE_URL


def build_pipeline():
    """Build production RAG pipeline."""
    print("=" * 60)
    print("PRODUCTION RAG PIPELINE (OPTIMIZED)")
    print("=" * 60, flush=True)

    # Step 1: Load & Chunk (M1)
    t0 = time.time()
    print("\n[1/4] Chunking documents (Hierarchical Parent/Child)...", flush=True)
    docs = load_documents()
    all_chunks = []
    parent_map = {}

    for doc in docs:
        parents, children = chunk_hierarchical(doc["text"], metadata=doc["metadata"])
        for p in parents:
            pid = p.metadata.get("parent_id")
            if pid:
                parent_map[pid] = p.text
        for child in children:
            all_chunks.append({
                "text": child.text,
                "metadata": {
                    **child.metadata,
                    "parent_id": child.parent_id,
                    "parent_text": parent_map.get(child.parent_id, child.text),
                },
            })
    print(f"  ✓ {len(all_chunks)} chunks from {len(docs)} documents ({time.time()-t0:.1f}s)", flush=True)

    # Step 2: Enrichment (M5)
    t0 = time.time()
    print(f"\n[2/4] Enriching {len(all_chunks)} chunks (M5, 1 API call/chunk)...", flush=True)
    enriched = enrich_chunks(all_chunks)
    if enriched:
        all_chunks = [{"text": e.enriched_text, "metadata": e.auto_metadata} for e in enriched]
        print(f"  ✓ Enriched {len(enriched)} chunks ({time.time()-t0:.1f}s)", flush=True)
    else:
        print("  ⚠️  M5 not implemented — using raw chunks", flush=True)

    # Step 3: Index (M2)
    t0 = time.time()
    print(f"\n[3/4] Indexing {len(all_chunks)} chunks (BM25 + Dense)...", flush=True)
    search = HybridSearch()
    search.index(all_chunks)
    print(f"  ✓ Indexed ({time.time()-t0:.1f}s)", flush=True)

    # Step 4: Reranker (M3)
    t0 = time.time()
    print("\n[4/4] Loading reranker...", flush=True)
    reranker = CrossEncoderReranker()
    print(f"  ✓ Reranker ready ({time.time()-t0:.1f}s)", flush=True)

    return search, reranker


def decompose_query(query: str, client) -> list[str]:
    """Decompose compound or multi-intent questions into sub-queries."""
    if client and (" và " in query or "?" in query[:-1] or ("ai " in query and "cần gì" in query)):
        try:
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Nếu câu hỏi chứa nhiều ý hoặc yêu cầu tra cứu từ nhiều chủ đề khác nhau, "
                            "hãy tách thành 2 câu hỏi tìm kiếm ngắn gọn độc lập, mỗi dòng 1 câu. "
                            "Chỉ trả về các câu hỏi, không giải thích."
                        ),
                    },
                    {"role": "user", "content": query},
                ],
                max_tokens=100,
                temperature=0.0,
            )
            content = resp.choices[0].message.content or ""
            lines = [l.strip().lstrip("0123456789.-) ") for l in content.strip().split("\n") if l.strip()]
            if len(lines) >= 2:
                return lines[:2]
        except Exception:
            pass
    return [query]


def run_query(query: str, search: HybridSearch, reranker: CrossEncoderReranker) -> tuple[str, list[str]]:
    """Run single query through pipeline with Sub-query Decomposition and Small-to-Big Retrieval."""
    client = None
    if OPENAI_API_KEY:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
        except Exception:
            pass

    sub_queries = decompose_query(query, client)
    seen_parents = set()
    final_contexts = []

    for sq in sub_queries:
        results = search.search(sq, top_k=15)
        docs_payload = [{"text": r.text, "score": r.score, "metadata": r.metadata} for r in results]
        reranked = reranker.rerank(sq, docs_payload, top_k=max(RERANK_TOP_K, 3))
        for r in reranked:
            p_text = r.metadata.get("parent_text", r.text)
            if p_text not in seen_parents:
                seen_parents.add(p_text)
                final_contexts.append(p_text)

    if not final_contexts:
        results = search.search(query, top_k=3)
        final_contexts = [r.text for r in results]

    if client and final_contexts:
        try:
            context_str = "\n\n---\n\n".join(final_contexts)
            system_prompt = (
                "Bạn là trợ lý hỏi đáp quy chế nội bộ doanh nghiệp. "
                "Hãy trả lời câu hỏi dựa trên Context được cung cấp một cách đầy đủ, chính xác và súc tích:\n"
                "1. Nếu có nhiều phiên bản quy định theo năm hoặc phiên bản khác nhau trong Context (ví dụ v1.0/v2.0, năm 2023/2024), hãy ưu tiên áp dụng chính sách mới nhất (v2024, v2.0) và nêu rõ phiên bản hiện hành, có thể so sánh ngắn gọn với bản cũ nếu hữu ích.\n"
                "2. Nếu câu hỏi yêu cầu tính toán (lương thử việc 85%, phạt chậm thanh toán theo tỷ lệ ngày pro-rata 1 tháng = 30 ngày, ngày phép thâm niên, cam kết đào tạo), hãy thực hiện phép tính cụ thể và đưa ra con số chính xác.\n"
                "3. Với các câu hỏi đa ý hoặc điều kiện cấm/không được phép, hãy trả lời trực diện và đầy đủ tất cả các vế của câu hỏi.\n"
                "4. Nếu trong Context hoàn toàn không có thông tin, hãy trả lời 'Không tìm thấy thông tin.'"
            )
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Context:\n{context_str}\n\nCâu hỏi: {query}"},
                ],
                temperature=0.0,
            )
            answer = resp.choices[0].message.content or "Không tìm thấy thông tin."
        except Exception as e:
            print(f"  ⚠️  LLM generation failed: {e}", flush=True)
            answer = final_contexts[0]
    else:
        answer = final_contexts[0] if final_contexts else "Không tìm thấy thông tin."

    return answer, final_contexts


def evaluate_pipeline(search: HybridSearch, reranker: CrossEncoderReranker):
    """Run evaluation on test set."""
    test_set = load_test_set()
    print(f"\n[Eval] Running {len(test_set)} queries...", flush=True)
    questions, answers, all_contexts, ground_truths = [], [], [], []

    for i, item in enumerate(test_set):
        answer, contexts = run_query(item["question"], search, reranker)
        questions.append(item["question"])
        answers.append(answer)
        all_contexts.append(contexts)
        ground_truths.append(item["ground_truth"])
        print(f"  [{i+1}/{len(test_set)}] {item['question'][:50]}...", flush=True)

    t0 = time.time()
    print(f"\n[Eval] Running RAGAS (4 metrics × {len(test_set)} questions)...", flush=True)
    results = evaluate_ragas(questions, answers, all_contexts, ground_truths)
    print(f"  ✓ RAGAS done ({time.time()-t0:.1f}s)", flush=True)

    print("\n" + "=" * 60)
    print("PRODUCTION RAG SCORES")
    print("=" * 60)
    for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        s = results.get(m, 0)
        print(f"  {'✓' if s >= 0.75 else '✗'} {m}: {s:.4f}")

    failures = failure_analysis(results.get("per_question", []))
    save_report(results, failures)
    return results


if __name__ == "__main__":
    start = time.time()
    search, reranker = build_pipeline()
    evaluate_pipeline(search, reranker)
    print(f"\nTotal: {time.time() - start:.1f}s")

