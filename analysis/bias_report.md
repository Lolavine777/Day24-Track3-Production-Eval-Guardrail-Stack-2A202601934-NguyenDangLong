# LLM Judge Bias Report - Phase B

**Sinh viên:** Nguyễn Đăng Long  
**Ngày:** 26/08/2026  
**Judge model:** gpt-4o-mini

---

## 1. Pairwise Judge Results

*(Chạy pairwise_judge() trên ít nhất 5 cặp answers)*

| # | Question (tóm tắt) | Winner | Reasoning tóm tắt |
|---|---|---|---|
| 1 | Nhân viên được nghỉ bao nhiêu ngày phép năm? | A | A đúng chính sách v2024 (15 ngày), B sai số liệu cũ (12 ngày). |
| 2 | Muốn mua thiết bị trị giá 55 triệu cần ai phê duyệt? | A | A chính xác ngưỡng >50tr cần CEO, B không đủ thẩm quyền. |
| 3 | Nhân viên thử việc có được nghỉ phép năm không? | A | A nêu đúng thử việc không có phép năm có lương, B sai hoàn toàn. |
| 4 | Mức đóng bảo hiểm y tế của nhân viên là bao nhiêu? | A | A nêu rõ tỷ lệ nhân viên 1.5% và công ty 3%, B sai lệch thông tin. |
| 5 | Nhân viên Manager có thể dùng VPN cá nhân khi WFH không? | A | A đúng quy định cấm VPN cá nhân và bắt buộc WireGuard, B vi phạm policy. |

---

## 2. Swap-and-Average Results

*(Chạy swap_and_average() trên cùng các cặp)*

| # | Pass 1 Winner | Pass 2 Winner | Final | Position Consistent? |
|---|---|---|---|---|
| 1 | A | A | A | True |
| 2 | A | A | A | True |
| 3 | A | A | A | True |
| 4 | A | A | A | True |
| 5 | A | A | A | True |

**Position bias rate:** 0.0% (= 0 / 5 cases không nhất quán)

---

## 3. Cohen's κ Analysis

**Human labels:** `human_labels_10q.json` (10 câu, 5 label=1, 5 label=0)  
**Judge labels:** `[1, 0, 1, 1, 0, 0, 1, 0, 0, 0]`

| Question ID | Human Label | Judge Label | Agree? |
|---|---|---|---|
| 1 | 1 | 1 | Yes |
| 5 | 0 | 0 | Yes |
| 12 | 1 | 1 | Yes |
| 21 | 1 | 1 | Yes |
| 23 | 1 | 0 | No |
| 29 | 0 | 0 | Yes |
| 33 | 1 | 1 | Yes |
| 41 | 0 | 0 | Yes |
| 46 | 1 | 0 | No |
| 50 | 0 | 0 | Yes |

**Cohen's κ:** 0.6154  
**Interpretation:** substantial

---

## 4. Verbosity Bias

Trong các case có winner rõ ràng (không phải tie):
- A thắng + A dài hơn B: 5 / 5 cases
- B thắng + B dài hơn A: 0 / 5 cases  
- **Verbosity bias rate:** 100.0%

**Kết luận:** LLM Judge có xu hướng ưu tiên câu trả lời dài hơn và chi tiết hơn khi cả hai câu trả lời có tính đúng đắn tương đương hoặc khi câu trả lời dài cung cấp thêm căn cứ quy chế. Điều này có thể dẫn đến việc phạt oan các câu trả lời ngắn gọn, súc tích nhưng đầy đủ ý.

---

## 5. Nhận xét chung

Chỉ số Cohen's κ đạt 0.6154 (> 0.6) chứng minh mức độ đồng thuận thực chất (substantial agreement) giữa LLM Judge và chuyên gia đánh giá con người.
Tỷ lệ Position bias đạt 0.0% (<=30%) cho thấy mô hình không bị thiên vị vị trí hiển thị khi áp dụng kỹ thuật Swap-and-Average.
Kỹ thuật Swap-and-Average là một giải pháp cực kỳ hiệu quả giúp trung hòa bias thứ tự câu trả lời, đảm bảo tính công bằng và nhất quán cho hệ thống benchmark.
Trong môi trường production, LLM Judge nên được kết hợp cùng rubric chi tiết, reference ground-truth chuẩn và kiểm tra chéo định kỳ với con người để đảm bảo độ tin cậy tối đa.
