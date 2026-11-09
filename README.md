# Face Liveness Detection (REAL vs FAKE)

Nhận diện khuôn mặt **thật (REAL)** hay **giả (FAKE: ảnh in, ảnh/video trên màn hình)** qua camera theo thời gian thực, kèm trạng thái **mở/nhắm mắt**, **quay đầu trái/phải**, **ngẩng/cúi đầu**. Toàn bộ nằm trong một notebook ngắn: [Face_Liveness_Detection.ipynb](Face_Liveness_Detection.ipynb).

## Ngưỡng nhận diện

| Nhận diện | Số đo | Ngưỡng | Kết quả |
|---|---|---|---|
| Mở / nhắm mắt | `blink` (MediaPipe, 0 = mở … 1 = nhắm) | `≥ 0.5` | CLOSED, còn lại OPEN |
| Quay đầu | `yaw` (độ) | `> 12°` / `< −12°` | RIGHT / LEFT (theo ảnh hiển thị) |
| Ngẩng / cúi | `pitch` (độ) | `< −10°` / `> 10°` | UP / DOWN |
| **REAL / FAKE** | `p_real` (MiniVision, 0 … 1) | `≥ 0.108` (tìm bằng ROC, Youden trên train) | REAL, còn lại FAKE |

- Tất cả ngưỡng nằm ở cell đầu tiên của notebook (`EYE_CLOSED`, `YAW_TH`, `PITCH_TH`, `REAL_TH`).
- Ngưỡng mắt và góc đầu là lựa chọn thiết kế (dataset không có nhãn cho các trạng thái này) và chỉ để hiển thị. Video phát lại vẫn chớp mắt, quay đầu được, nên **quyết định REAL/FAKE dựa vào MiniVision**.
- Realtime dùng median `p_real` của 15 khung gần nhất, cần ít nhất 8 khung mới kết luận.
- ML Kit không có SDK Python nên dùng [MediaPipe Face Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker) thay thế. REAL/FAKE dùng [MiniVision Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing).

## Kết quả (NUAA, ngưỡng tìm trên train, đánh giá trên test)

| Trên TEST (9.122 ảnh) | Giá trị |
|---|---|
| Ngưỡng REAL/FAKE | 0.108 |
| Accuracy | 0.9993 |
| AUC | 1.000 |
| FAR (FAKE bị nhận là REAL) | 0.10% (6/5.760) |
| FRR (REAL bị từ chối) | 0% |

Confusion matrix, đường ROC và biểu đồ ngưỡng mắt/góc đầu nằm trong notebook (mục 4 và 5).

## Dataset

**NUAA Photograph Imposter Database** (bản trên Kaggle): đặt `dataset.zip` vào `data/`, notebook tự giải nén ra `data/raw/`. Train/test theo 4 file `.txt` gốc: train 3.491 ảnh, test 9.123 ảnh. `ClientRaw` = REAL, `ImposterRaw` = FAKE (ảnh in). Dataset không đưa lên GitHub (khoảng 400 MB).

## Cài đặt và chạy

Yêu cầu: **Python 3.10**.

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m ipykernel install --user --name liveness --display-name "Python (liveness)"
jupyter lab Face_Liveness_Detection.ipynb      # Run All; cell cuối mở camera, q / ESC để thoát
```

- Chấm điểm toàn bộ dataset mất khoảng 15 phút trên CPU. Kết quả được lưu cache ở `results/frame_scores.csv`, lần sau chỉ mất vài giây.
- Không có webcam: đặt `$env:LIVENESS_NO_CAMERA = "1"` trước khi chạy.
- Ngưỡng 0.108 khá lỏng với điện thoại/màn hình thật; muốn khắt khe hơn dùng `run_camera(thr=0.9)`.

## Tài liệu

Tóm tắt kiến thức (công thức, cách chọn ngưỡng, kết quả): [docs/tom_tat_kien_thuc.pdf](docs/tom_tat_kien_thuc.pdf) (mã LaTeX: [docs/tom_tat_kien_thuc.tex](docs/tom_tat_kien_thuc.tex)).
