# Drawing OCR

Công cụ cục bộ để mở PDF / ảnh bản vẽ, phát hiện blue box và kiểm tra kết quả OCR.

## Chạy ứng dụng

Mở `START-MINI-TOOL.bat`, sau đó truy cập `http://127.0.0.1:8090`.

Giao diện gồm panel thông tin có thể thu gọn ở bên trái, bản vẽ ở giữa và kết quả OCR ở bên phải. Nhấn **F5** để scan; lăn chuột để zoom; giữ Ctrl và kéo để pan. Tab **Ảnh cắt** hiển thị ảnh nguồn và các vùng nominal / dung sai. Có thể sửa số, xác nhận đã kiểm tra, đánh dấu quan trọng và xuất các dòng được chọn sang `.xlsx`.

Trạng thái **OK** chỉ xuất hiện sau khi người dùng xác nhận. OCR có thể đọc sai dù confidence cao. Thông tin người đo, lần quét và thay đổi được lưu trên trình duyệt; dữ liệu xuất Excel lấy từ kết quả đang hiển thị.

## Cấu trúc thư mục

| Đường dẫn | Nội dung |
| --- | --- |
| `index.html`, `assets/` | Giao diện và thư viện dùng offline |
| `server.py`, `engine.py`, `preprocessor.py` | HTTP server và pipeline xử lý |
| `models/` | Hai YOLO ONNX, một OCR ONNX và một từ điển |
| `sample_pdfs/`, `uploads/` | Bản vẽ mẫu và file tải lên |
| `CheckVal/`, `ocrtool-pdfscan-render-cache/` | Dữ liệu ảnh để đánh giá / tạo dữ liệu |
| `ppocr_v6_finetune/`, `runs_3regions_v3/` | Dữ liệu, script và báo cáo huấn luyện |
| `audit_results/` | Báo cáo kiểm tra và nhật ký dọn thư mục |
| `backups/` | Hai gói RAR sao lưu, không được nạp vào app |

Ba model sử dụng:

- `models/onnx_yolo/yolo_best_1class.onnx`: detect blue box toàn trang.
- `models/onnx_yolo/yolo_3regions_31026.onnx`: tách nominal và hai vùng dung sai.
- `models/onnx_ppocr/rec_nominal_stack_regions.onnx`: nhận dạng nội dung.

Không tự nhập lại model từ thư mục Export. Thư viện icon và xuất Excel được lưu trong `assets/`, không cần CDN khi chạy app.

## Kiểm tra

`..\.venv\Scripts\python.exe -m unittest test_mini_tool -v`

Kiểm tra giao diện và các khác biệt với mẫu nằm trong `design-qa.md`. Dọn ngày 06/10/2026: xóa cache Python và hai bản sao ZIP/notebook giống hệt, giải phóng khoảng 34 MB; chuyển gói RAR vào `backups/`; giữ dữ liệu và các model đang dùng.
