# Báo cáo lab: chọn tracker cho 5 video

**Hình thức:** Cá nhân
**Sinh viên:** Hà Mạnh Tuân — 2A202602982

Detector cố định: `yolo26n.pt`, ảnh đầu vào 640 px, lớp người (`class 0`).
Các tracker có Re-ID dùng `osnet_x0_25_msmt17.pt`. Toàn bộ lần chạy nộp
bài dùng `--device cuda:0` trên NVIDIA GeForce GTX 1660 Ti.

## 1. Cấu hình đã chọn

Mỗi bản nộp chạy đủ số frame; `conf` và `iou` là ngưỡng của detector YOLO.

| Video | Số frame | Tracker | conf | iou | Quan sát trên preview và lý do chọn | Đã thử nhưng loại |
|---|---:|---|---:|---:|---|---|
| video_1 | 600 | StrongSORT | 0.20 | 0.40 | Re-ID giữ danh tính tốt hơn ở người đi ngang nhau; HOTA cao nhất trong sáu cấu hình đã chấm. | ByteTrack 0.30/0.50, OCSORT 0.15/0.50 và StrongSORT với các ngưỡng khác. |
| video_2 | 1050 | StrongSORT | 0.20 | 0.50 | Cảnh đêm đông; ở frame 120, StrongSORT gắn ID cho thêm người ở giữa và phía xa. Trong 150 frame đầu có 1.876 hộp so với 1.453 của ByteTrack. | ByteTrack 0.15/0.50: bỏ sót thêm người ở vùng đông. |
| video_3 | 837 | OCSORT | 0.15 | 0.50 | Ảnh nhỏ và camera di động khiến đặc trưng ngoại hình kém rõ; trong 150 frame đầu OCSORT có 667 hộp/21 ID, DeepOCSORT có 660 hộp/23 ID. | DeepOCSORT 0.20/0.50: không cho thấy lợi thế rõ về độ phủ hay độ liên tục ID. |
| video_4 | 900 | BoT-SORT | 0.20 | 0.50 | Camera tiến về phía trước, người cắt nhau gần kính và mặt sàn phản chiếu. Trong 150 frame đầu BoT-SORT có 10 ID trên 776 hộp, ByteTrack có 19 ID trên 873 hộp; preview cho thấy ít track ngắn hơn. | ByteTrack 0.20/0.50: nhiều ID phân mảnh hơn. |
| video_5 | 750 | DeepOCSORT | 0.20 | 0.50 | Góc quay từ xe di chuyển; hai tracker đều có 26 ID trong 150 frame đầu, nhưng DeepOCSORT có 13 track dài ít nhất 20 frame, OCSORT có 11. | OCSORT 0.20/0.50: ít track dài hơn trên đoạn thử. |

Các thống kê video 2–5 chỉ mô tả đoạn thử và preview, không phải điểm HOTA.
Ảnh so sánh được lấy cùng frame 120 của hai tracker cho mỗi video.

## 2. Số liệu video_1

Chấm bằng TrackEval trên nhãn `video_1` đủ 600 frame:

| Tracker | conf | iou | HOTA | MOTA | IDF1 |
|---|---:|---:|---:|---:|---:|
| ByteTrack | 0.30 | 0.50 | 26.390 | 18.196 | 25.838 |
| OCSORT | 0.15 | 0.50 | 28.184 | 19.019 | 28.839 |
| StrongSORT | 0.10 | 0.50 | 28.529 | 16.840 | 33.307 |
| StrongSORT | 0.20 | 0.70 | 28.188 | 20.360 | 30.981 |
| StrongSORT | 0.20 | 0.50 | 29.449 | 20.903 | 32.513 |
| **StrongSORT — bản nộp** | **0.20** | **0.40** | **29.452** | **21.199** | **32.571** |

Với cấu hình nộp, TrackEval còn ghi nhận 4.608 true positive, 13.973 false
negative, 611 false positive và 58 ID switch. Điểm còn thấp chủ yếu vì
detector cố định bỏ sót nhiều người nhỏ hoặc bị che, nhất là ở phía xa;
không thay detector hay kích thước ảnh vì luật lab yêu cầu giữ cố định.

## 3. Phân tích

**Video 1:** Camera đứng yên, nhưng người đi ngang nhau và bị che khuất.
ByteTrack ở cấu hình baseline có độ chính xác hộp cao nhưng bỏ sót nhiều người,
IDF1 chỉ 25,838. StrongSORT dùng đặc trưng ngoại hình để nối lại người sau
khi khuất một phần; ở `conf=0.20`, nó cải thiện IDF1 lên 32,571. Hạ `iou`
NMS từ 0,50 xuống 0,40 giảm hộp giả và ID switch, dù HOTA chỉ tăng rất nhẹ.

**Video 3:** Camera di chuyển, nhưng ảnh chỉ 640×480 và người xa có ít chi tiết.
Trong đoạn so sánh 150 frame, DeepOCSORT không tạo thêm track dài so với
OCSORT. Re-ID khó phân biệt ngoại hình khi người chỉ chiếm ít pixel, nên chọn
OCSORT để giữ độ phủ tương đương với ít ID hơn. Cần xem thêm các đoạn camera
quay mạnh để đánh giá đầy đủ vì video này không có nhãn.

**Video 4:** BoT-SORT có cơ chế dùng ngoại hình và bù chuyển động camera,
phù hợp cảnh camera tiến trong trung tâm thương mại. Kính và sàn bóng có thể
gây hộp giả, nên chọn `conf=0.20` thay vì hạ quá thấp. Trên đoạn thử,
BoT-SORT tạo ít ID hơn ByteTrack và preview cho thấy các người gần máy được
giữ ID qua nhiều frame. Không có nhãn, vì vậy kết luận này là quan sát định tính.

**Video 5:** Camera trên xe làm toàn bộ nền dịch chuyển, gây khó khăn cho
tracker chỉ dựa vào vị trí hộp. DeepOCSORT kết hợp chuyển động và ngoại hình,
và có nhiều track dài hơn OCSORT trong đoạn thử. Ở frame 120, hai phương án
đều bỏ sót vài người rất nhỏ phía xa; đây là giới hạn của detector cố định.

## 4. Nếu có thêm thời gian

Xem kỹ các thời điểm ID đổi quanh vùng che khuất và thử `conf` bước 0,05
quanh cấu hình đã chọn. Video 2–5 cần nhãn riêng mới có thể kết luận bằng
HOTA/MOTA/IDF1; báo cáo này không suy diễn điểm số cho chúng.
