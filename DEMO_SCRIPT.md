# Support Ticket Agent — Demo Script

Mở `http://127.0.0.1:8765/demo.html`.

## Demo nhanh (2–3 phút)

1. Bấm **Guided demo**. Không cần nhập API key; UI sẽ tự chạy các trace mẫu.
2. Khi scenario `S01` chạy, nói: “Đây là happy path: intake chuẩn hóa ticket, classify chọn `simple`, answer tạo phản hồi, rồi finalize kết thúc.”
3. Khi `S02` chạy, chỉ vào graph: “Tool route đi qua `tool → evaluate → answer`; tool result được giữ trong state.”
4. Khi `S03` chạy: “Thiếu context không được đoán bừa; graph đi vào `clarify` rồi finalize.”
5. Khi `S04` chạy: “Đây là risky path. `risky_action` chuẩn bị đề xuất, approval xuất hiện trước tool. Không có side effect trước gate.”
6. Khi `S05` chạy: “Tool lỗi sẽ quay qua retry có giới hạn. Mỗi lần retry được ghi vào audit log; không retry vô hạn.”
7. Chọn **Dead-letter** nếu muốn nhấn mạnh `max_attempts`: retry chạm giới hạn sẽ đi `dead_letter → finalize`.

## Các điểm cần chỉ trên màn hình

- **Graph trace:** node đang chạy được highlight; node đã chạy chuyển xanh; mọi branch kết thúc ở `finalize → END`.
- **Evidence console:** `thread_id`, route, attempt/max attempts, approval gate và checkpoint history.
- **Audit log:** mỗi node tạo một event có timestamp; `retry`, `approval`, `dead_letter`, `finalize` nhìn thấy trực tiếp.
- **Chatbot:** câu trả lời cuối nằm bên trái, cùng với ticket đầu vào.

## Demo approval rejection

Trong scenario risky, nói thêm: “Nếu reviewer reject, route đúng là `approval → clarify → finalize`; tool không được gọi.” Contract test của lab đã chứng minh nhánh này; UI hiển thị nhánh approved để giữ guided flow ngắn.

## Kết luận

“Bằng chứng runtime nằm trong `outputs/metrics.json`, `outputs/audit_events.jsonl` và `outputs/persistence_evidence.json`. Metrics có 7/7 scenario thành công; retry, approval, event trail và state history đều truy vết được.”
