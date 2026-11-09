# Description

Nhận diện khuôn mặt **thật (REAL)** hay **giả (FAKE: ảnh in, ảnh/video trên màn hình)** qua camera theo thời gian thực, kèm trạng thái **mở/nhắm mắt** (EAR), **số lần chớp mắt**, **xoay đầu trái/phải** và **ngẩng/cúi đầu** (rotation matrix). Toàn bộ nằm trong một notebook: [Face_Liveness_Detection.ipynb](Face_Liveness_Detection.ipynb).

| Thành phần | Công cụ |
|---|---|
| REAL / FAKE | [MiniVision Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing) (MiniFASNetV2 + MiniFASNetV1SE) |
| Khuôn mặt, điểm mốc, rotation matrix | [MediaPipe Face Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker) (ML Kit không có SDK Python nên dùng MediaPipe thay thế) |
| Dataset | NUAA Photograph Imposter Database (Kaggle) |

## Bảng tổng hợp các ngưỡng

Tất cả ngưỡng được **tính từ dữ liệu** ở mục 4 của notebook, lưu vào `results/thresholds.json`. Camera đọc đúng file này.

| Nhận diện | Số đo | Ngưỡng | Hiển thị trên camera | Cách tìm | Kiểm tra lại (notebook in ra) |
|---|---|---|---|---|---|
| Mở / nhắm mắt | EAR | **< 0.105** → nhắm | `EYES CLOSED` / `EYES OPEN` | Phân vị 1% của EAR trên mọi ảnh (99% ảnh có EAR lớn hơn) | 12.479 / 12.606 ảnh có EAR ≥ 0.105 (99,0%) |
| Chớp mắt | Số lần CLOSED → OPEN | dùng ngưỡng mắt 0.105 | `blinks N` | Đếm theo chuyển trạng thái mắt | Cả 127 ảnh có EAR < 0.105 đều là người thật (ảnh in không chớp được) |
| Xoay trái / phải | yaw (°) | **< −11.5° / > 11.5°** | `TURN LEFT` / `TURN RIGHT` / `FACE STRAIGHT` | Phân vị 99% của \|yaw\| trên ảnh người thật nhìn thẳng | 5.055 / 5.105 ảnh có \|yaw\| ≤ 11.5 (99,0%) |
| Ngẩng / cúi | pitch (°) | **< −12.27° / > 12.27°** | `HEAD UP` / `HEAD DOWN` / `HEAD LEVEL` | Phân vị 99% của \|pitch\| trên ảnh người thật nhìn thẳng | 5.052 / 5.105 ảnh có \|pitch\| ≤ 12.27 (99,0%) |
| **REAL / FAKE** | p_real (MiniFASNet) | **≥ 0.108** → REAL | `REAL` / `FAKE` | ROC trên tập train, chọn ngưỡng làm TPR − FPR lớn nhất (Youden) | Train: REAL thấp nhất 0.108, FAKE cao nhất 0.050 |

- **Hình chứng minh** (ảnh thật kèm số đo: vùng mắt theo EAR, cùng một khuôn mặt xếp theo yaw/pitch): [docs/chung_minh_nguong.png](docs/chung_minh_nguong.png).
- Mắt và góc đầu chỉ để hiển thị. Video phát lại vẫn chớp mắt, quay đầu được, nên **quyết định REAL/FAKE dựa vào MiniFASNet**.

Yêu cầu: **Python 3.10**.

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m ipykernel install --user --name liveness --display-name "Python (liveness)"
jupyter lab Face_Liveness_Detection.ipynb      # Run All từ đầu; mục 5 mở camera, q / ESC để thoát
```

## Tài liệu

Tóm tắt kiến thức [docs/tom_tat_kien_thuc.pdf](docs/tom_tat_kien_thuc.pdf)
