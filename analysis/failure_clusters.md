# Failure Cluster Analysis - Phase A

**Sinh viên:** Nguyễn Đăng Long  
**Ngày:** 26/08/2026

---

## 1. Aggregate RAGAS Scores theo Distribution

| Metric | factual | multi_hop | adversarial |
|---|---|---|---|
| faithfulness | 1.0000 | 0.6911 | 0.9133 |
| answer_relevancy | 0.9513 | 0.7425 | 0.9365 |
| context_precision | 0.8500 | 0.7500 | 0.8500 |
| context_recall | 0.9000 | 0.5875 | 0.5500 |
| **avg_score** | **0.9253** | **0.6928** | **0.8124** |

---

## 2. Bottom 10 Questions

| Rank | Distribution | Question | avg_score | worst_metric |
|---|---|---|---|---|
| 1 | multi_hop | Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu? | 0.3750 | answer_relevancy |
| 2 | multi_hop | Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào? | 0.3750 | answer_relevancy |
| 3 | multi_hop | Nhân viên vừa kết hôn và cùng tuần đó có con kết hôn. Tổng số ngày nghỉ đặc biệt có lương là bao nhiêu? | 0.4707 | faithfulness |
| 4 | multi_hop | Nhân viên có thâm niên 7 năm theo v2024 được nghỉ bao nhiêu ngày phép? Nếu bị trừ thêm 4 ngày ốm không có giấy y tế thì còn lại bao nhiêu? | 0.4921 | context_recall |
| 5 | multi_hop | Nếu cần mua một chiếc laptop 30 triệu cho nhân viên mới, ai phê duyệt và cần gì từ phòng CNTT? | 0.5208 | answer_relevancy |
| 6 | multi_hop | So sánh yêu cầu mật khẩu giữa policy v1.0 và v2.0 về độ dài tối thiểu, thời hạn đổi và MFA. | 0.5696 | context_recall |
| 7 | multi_hop | So sánh quyền lợi bảo hiểm giữa nhân viên thử việc và nhân viên chính thức. | 0.6250 | answer_relevancy |
| 8 | adversarial | Nhân viên Manager có thể dùng VPN cá nhân (như NordVPN) khi WFH để tăng bảo mật thêm không? | 0.6715 | context_recall |
| 9 | adversarial | Nhân viên được nghỉ bao nhiêu ngày phép năm? | 0.6824 | context_recall |
| 10 | multi_hop | Nhân viên đi công tác nước ngoài 4 ngày tại thành phố lớn, ở khách sạn 200 USD/đêm (3 đêm). Công ty thanh toán bao nhiêu tiền khách sạn? | 0.7064 | faithfulness |

---

## 3. Failure Cluster Matrix

*(Mỗi ô = số câu có worst_metric = row, thuộc distribution = col)*

| worst_metric | factual | multi_hop | adversarial | Total |
|---|---|---|---|---|
| faithfulness | 0 | 7 | 0 | 7 |
| answer_relevancy | 14 | 4 | 1 | 19 |
| context_precision | 3 | 2 | 2 | 7 |
| context_recall | 3 | 7 | 7 | 17 |

---

## 4. Dominant Failure Analysis

**Dominant distribution:** factual  
**Dominant metric:** answer_relevancy

**Lý do phân tích:**

Tập câu hỏi factual tuy có điểm trung bình chung cao nhất (0.9253) nhưng lại tập trung nhiều câu có answer_relevancy là metric tương đối thấp nhất trong 4 metric của nhóm này.
Nguyên nhân chủ yếu do câu trả lời của LLM thường sinh ra các câu mở đầu hoặc kết luận dài mang tính giải thích thêm thay vì đi thẳng vào con số cụ thể theo format hỏi - đáp ngắn gọn.
Đối với tiếng Việt trong văn bản HR, mô hình generator có xu hướng lặp lại ngữ cảnh hoặc đưa thêm lời khuyên hỗ trợ nhân sự làm loãng độ cô đọng của câu trả lời.
Trong khi đó, ở nhóm multi_hop và adversarial, context_recall và faithfulness giảm sút đáng kể do thông tin phân tán trên nhiều chunk tài liệu hoặc bị xung đột phiên bản giữa v2023 và v2024.

---

## 5. Suggested Fixes

| Metric yếu | Root cause | Suggested fix |
|---|---|---|
| faithfulness | LLM hallucinating khi tổng hợp nhiều điều kiện logic phức tạp | Thêm CoT prompt và strict constraint chỉ suy luận dựa trên context |
| context_recall | Missing relevant chunks do thông tin nằm ở 2 file policy khác nhau | Tăng top_k retrieval từ 20 lên 30 và áp dụng ParentDocument / Contextual chunking |
| context_precision | Too many irrelevant chunks lọt vào prompt sau rerank | Điều chỉnh CrossEncoder reranker threshold và lọc bớt chunk có score thấp |
| answer_relevancy | Answer dài dòng hoặc không đúng trọng tâm câu hỏi | Tối ưu system prompt yêu cầu trả lời trực diện, ngắn gọn và có cấu trúc |

---

## 6. Nhận xét về Adversarial Distribution

Điểm trung bình của nhóm adversarial đạt 0.8124, thấp hơn rõ rệt so với factual (0.9253) đúng như kỳ vọng đánh giá hệ thống.
Nhóm adversarial làm lộ rõ điểm yếu context_recall (0.5500) khi xuất hiện các câu hỏi bẫy về phiên bản quy chế cũ (như quy định phép năm v2023 12 ngày so với v2024 15 ngày) hoặc các câu hỏi phủ định quyền lợi (như dùng VPN cá nhân).
Hai câu hỏi adversarial lọt vào bottom 10 gồm #50 (VPN cá nhân) và #41 (Phép năm v2023 vs v2024) do retriever lấy nhầm các chunk chính sách cũ hoặc không truy xuất đủ bối cảnh quy định nghiêm cấm.
Điều này khẳng định sự cần thiết phải có cơ chế metadata filtering theo version hiệu lực và Guardrail layer để ngăn chặn thông tin lỗi thời trước khi đưa vào inference.
