# Face Liveness Detection (REAL vs FAKE)

Hệ thống phát hiện khuôn mặt **thật (REAL)** và **giả (FAKE: ảnh in, ảnh/video trên màn hình)** qua camera theo thời gian thực, viết bằng Python trong **một Jupyter Notebook duy nhất** ([Face_Liveness_Detection.ipynb](Face_Liveness_Detection.ipynb)). Notebook tự tải model, đánh giá trên dataset công khai **NUAA Photograph Imposter Database** và có chế độ chạy camera.

> **Lưu ý trung thực:** trên NUAA, MiniVision tách REAL/FAKE gần như hoàn hảo, nhưng NUAA **chỉ có tấn công bằng ảnh in** (không có video phát lại, màn hình), ảnh từ webcam cũ. Kết quả này **không** suy ra được độ chính xác với điện thoại/màn hình hay camera của bạn. Xem mục [Giới hạn](#giới-hạn).

## Cách hoạt động

| Thành phần | Công nghệ | Vai trò |
|---|---|---|
| Phân loại REAL/FAKE | [MiniVision Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing) (MiniFASNetV2 + MiniFASNetV1SE) | **Quyết định chính** (chỉ nhìn ảnh, không cần người dùng làm động tác) |
| Phát hiện mặt, chớp mắt, góc đầu | [MediaPipe Face Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker) | Cấp hộp mặt cho MiniVision; hiển thị chớp mắt và hướng đầu |
| Chuyển động khuôn mặt | Độ dịch chuyển landmark giữa các khung | Chỉ hiển thị |
| Khóa ánh sáng | Cổng kiểm tra độ sáng + khóa phơi sáng camera | Không kết luận khi ánh sáng xấu |

**Quyết định REAL/FAKE** = median của 15 khung gần nhất của điểm `p_real` (MiniVision) so với ngưỡng đã chọn (`results/threshold.json`), cần tối thiểu 8 khung. Chớp mắt, xoay đầu và chuyển động **không** tham gia quyết định, vì video phát lại vẫn có thể chứa chớp mắt hay quay đầu.

### Về Google ML Kit
ML Kit chỉ có SDK cho Android/iOS, **không có SDK Python chính thức**. Dự án dùng MediaPipe Face Landmarker của Google để thay thế:

- blendshape `eyeBlinkLeft/Right` ≈ 1 − `eyeOpenProbability` (chớp mắt);
- ma trận biến đổi khuôn mặt → yaw / pitch / roll (hướng đầu);
- landmark → hộp mặt và chuyển động.

### "Khóa ánh sáng"
1. **Cổng ánh sáng:** độ sáng trung bình vùng mặt 55–205, tỉ lệ điểm cháy sáng dưới 10%, tương phản ít nhất 15. Ngoài khoảng này thì báo `BAD LIGHT` và không kết luận. Các ngưỡng do thiết kế chọn, chưa tinh chỉnh.
2. **Khóa phơi sáng camera** (phím `l`): tắt tự động phơi sáng để camera không tự bù sáng. *Thực nghiệm, phụ thuộc driver camera, chưa kiểm chứng.*

## Dataset: NUAA Photograph Imposter Database

Đặt file `dataset.zip` (bản NUAA trên Kaggle) vào `data/`. Notebook tự giải nén ra `data/raw/` nếu chưa có.

```
data/raw/
├── ClientRaw/0001 ... 0015/     ảnh người thật trước webcam         (REAL)
├── ImposterRaw/0001 ... 0016/   ảnh chụp lại ảnh in của người đó    (FAKE)
├── client_train_raw.txt   imposter_train_raw.txt    danh sách TRAIN
└── client_test_raw.txt    imposter_test_raw.txt     danh sách TEST
```

| Tập | REAL | FAKE | Số chuỗi | Số người |
|---|---|---|---|---|
| Train | 1.743 ảnh | 1.748 ảnh | 61 | 9 |
| Test | 3.362 ảnh | 5.761 ảnh | 82 | 16 |

- Mỗi thư mục con là 1 người. Ảnh là các khung hình rời của một lần quay (chuỗi), 640×480.
- Train/test theo cách chia gốc của dataset (4 file `.txt`). 9 người có mặt ở cả hai tập nhưng khác buổi quay.
- Dataset không được đưa lên GitHub (khoảng 400 MB, xem `.gitignore`).

## Kết quả đánh giá

**Quy trình:** chọn ngưỡng theo Youden's J **trên train**, giữ nguyên ngưỡng đó, đo mọi chỉ số **trên test** (positive = REAL).

| Chỉ số (trên TEST, 9.122 ảnh có thấy mặt) | Giá trị |
|---|---|
| AUC | **1.000** (khoảng tin cậy 95% bootstrap theo người: 1.000–1.000) |
| Ngưỡng chọn trên train (Youden) | 0.108 |
| Accuracy | 0.9993 (mốc "luôn đoán lớp đông nhất" = 0.631) |
| Balanced accuracy | 0.9995 |
| FAR (FAKE bị chấp nhận nhầm) | 0.10% (6/5.760 ảnh) |
| FRR (REAL bị từ chối nhầm) | 0% (0/3.362 ảnh) |
| Accuracy với ngưỡng mặc định 0.5 | 1.000 (FAR 0, FRR 0) |
| Mức chuỗi (median `p_real` mỗi chuỗi) | 82/82 chuỗi đúng |
| Giả lập realtime trên 82 chuỗi test (45 khung đầu mỗi chuỗi) | 61/61 chuỗi có kết luận đều đúng; 21 chuỗi không kết luận (`BAD LIGHT` 10, `NO FACE` 7, chưa đủ khung 4) |

Tín hiệu phụ (mức chuỗi, AUC khi dùng riêng): chớp mắt 0.86 (ảnh in không chớp được), xoay đầu 0.53 (không phân biệt), chuyển động 0.11 (**ảnh giả chuyển động nhiều hơn** vì bị cầm tay rung).

Biểu đồ (phân bố điểm, ROC train/test, đánh đổi FAR/FRR), confusion matrix và lỗi theo từng người nằm trong **mục 7** của notebook. Dữ liệu gốc nằm trong `results/`.

## Cấu trúc thư mục

```
.
├── Face_Liveness_Detection.ipynb   # Notebook chính (code + giải thích + kết quả)
├── requirements.txt                # Phiên bản thư viện đã kiểm thử
├── models/                         # MiniFASNet.py, trọng số .pth, face_landmarker.task (tự tải)
├── data/dataset.zip                # NUAA (tự giải nén ra data/raw/)
└── results/                        # frame_scores.csv (điểm từng ảnh, cache), threshold.json
```

Các mục của notebook: 1 Khả thi · 2 Cài đặt · 3 Tải model và đọc dataset · 4 Module lõi (MiniVision, phát hiện mặt, chớp mắt, xoay đầu, chuyển động, ánh sáng) · 5 Chấm điểm dataset · 6 Chớp mắt, xoay đầu, chuyển động theo chuỗi · 7 Đánh giá (ngưỡng trên train, kết quả trên test) · 8 Hệ thống realtime (8.1 giả lập trên chuỗi test) · 9 Chạy camera · 10 Kết luận và giới hạn.

## Cài đặt và chạy

Yêu cầu: **Python 3.10**. Đã thử trên Windows 11 với torch CPU.

```powershell
# 1. Tạo môi trường và cài thư viện
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

# 2. Đăng ký kernel cho Jupyter
python -m ipykernel install --user --name liveness --display-name "Python (liveness)"

# 3. Mở notebook rồi Run All
jupyter lab Face_Liveness_Detection.ipynb
```

`requirements.txt` ghi đúng phiên bản đã kiểm thử. Hãy dùng Python 3.10 và môi trường ảo riêng, đừng cài vào Python hệ thống.

- **Thời gian:** chấm điểm toàn bộ 12.614 ảnh mất khoảng 15 phút trên CPU (khoảng 0,07 giây/ảnh). Kết quả được lưu cache trong `results/frame_scores.csv`; lần sau notebook đọc lại trong vài giây, trừ khi danh sách ảnh thay đổi hoặc đặt `FORCE_RECOMPUTE = True`.
- **Máy chậm:** đặt `FRAME_STEP = 2` hoặc lớn hơn (mục 3) để chỉ lấy 1 ảnh trong mỗi `FRAME_STEP` ảnh của mỗi chuỗi.

**Chạy không cần camera** (kiểm tra tự động, ghi ra file mới):

```powershell
$env:LIVENESS_NO_CAMERA = "1"
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.kernel_name=liveness --ExecutePreprocessor.timeout=7200 Face_Liveness_Detection.ipynb --output run_check.ipynb
```

## Chạy realtime với camera

Chạy đến cell ở **mục 9** (không đặt `LIVENESS_NO_CAMERA`). Cửa sổ camera hiển thị REAL/FAKE, điểm `p_real`, số lần chớp mắt, hướng đầu, mức chuyển động và trạng thái ánh sáng.

| Phím | Chức năng |
|---|---|
| `q` / `ESC` | Thoát |
| `r` | Reset bộ làm mượt và bộ đếm |
| `l` | Khóa/mở khóa phơi sáng camera (thực nghiệm) |

**Về ngưỡng cho camera:** `results/threshold.json` hiện là **0.108**, chọn trên NUAA. Vì REAL và FAKE của NUAA tách rất xa nhau, ngưỡng này **rất lỏng**: ảnh trên điện thoại/màn hình trước webcam hiện đại có thể vượt qua. Nên thử với chính camera của bạn và chỉnh khi gọi, ví dụ `run_realtime(source=0, threshold=0.9)`.

