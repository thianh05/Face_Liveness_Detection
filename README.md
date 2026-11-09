# Face Liveness Detection (REAL vs FAKE)

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

## Các công thức dùng trong source

### 1. MiniFASNet: điểm người thật `p_real`
Cắt vùng quanh hộp mặt $(x, y, w, h)$ với tỉ lệ mở rộng $s \in \{2.7,\ 4.0\}$ (mỗi mạng một tỉ lệ), rồi thu về 80×80:

$$k=\min\left(s,\ \frac{H-1}{h},\ \frac{W-1}{w}\right),\qquad c_x = x+\frac{w}{2},\quad c_y = y+\frac{h}{2},\qquad \text{vùng cắt}=\left[c_x \pm \frac{kw}{2}\right]\times\left[c_y \pm \frac{kh}{2}\right]$$

Mỗi mạng cho 3 giá trị đầu ra $z_0, z_1, z_2$. Hàm softmax đổi chúng thành xác suất, lớp 1 là "người thật":

$$p_j=\frac{e^{z_j}}{\sum_{m=0}^{2} e^{z_m}},\qquad p_{\text{real}}=\frac{p_1^{(\text{V2})}+p_1^{(\text{V1SE})}}{2}$$

### 2. Hộp mặt từ 478 điểm mốc MediaPipe
$$x_0=\min_i x_i,\quad y_0=\min_i y_i,\quad w=\max_i x_i-x_0,\quad h=\max_i y_i-y_0$$

### 3. Mắt: EAR (Eye Aspect Ratio)
Mỗi mắt lấy 6 điểm mốc: $p_1, p_4$ là 2 khoé mắt, $p_2, p_3$ trên mí trên, $p_6, p_5$ trên mí dưới. Mắt trái dùng các điểm `[33, 160, 158, 133, 153, 144]`, mắt phải dùng `[362, 385, 387, 263, 373, 380]`.

$$\text{EAR}=\frac{\lVert p_2-p_6\rVert+\lVert p_3-p_5\rVert}{2\,\lVert p_1-p_4\rVert},\qquad \text{EAR}_{\text{mặt}}=\frac{\text{EAR}_{\text{trái}}+\text{EAR}_{\text{phải}}}{2}$$

Tử số là chiều cao mắt, mẫu số là chiều rộng mắt. Mắt mở thì EAR ≈ 0.3, nhắm thì EAR gần 0.

### 4. Góc đầu: rotation matrix
MediaPipe trả về ma trận biến đổi của khuôn mặt. Lấy phần xoay $R$ (3×3), chia mỗi cột cho độ dài của nó để bỏ hệ số phóng to. Ba phép xoay cơ bản:

$$
R_x(\theta)=\begin{bmatrix}1&0&0\\0&\cos\theta&-\sin\theta\\0&\sin\theta&\cos\theta\end{bmatrix},\quad
R_y(\psi)=\begin{bmatrix}\cos\psi&0&\sin\psi\\0&1&0\\-\sin\psi&0&\cos\psi\end{bmatrix},\quad
R_z(\varphi)=\begin{bmatrix}\cos\varphi&-\sin\varphi&0\\\sin\varphi&\cos\varphi&0\\0&0&1\end{bmatrix}
$$

Tư thế đầu $R=R_y(\psi)\,R_x(\theta)\,R_z(\varphi)$, trong đó $\psi$ = yaw (xoay trái/phải), $\theta$ = pitch (ngẩng/cúi), $\varphi$ = roll (nghiêng, không dùng). Nhân ra được:

$$R_{02}=\sin\psi\cos\theta,\qquad R_{22}=\cos\psi\cos\theta,\qquad R_{12}=-\sin\theta$$

$$\boxed{\text{yaw}=\operatorname{atan2}(R_{02},\,R_{22})}\qquad \boxed{\text{pitch}=\arcsin(-R_{12})}$$

- yaw > 0: mặt quay về bên phải màn hình; pitch > 0: cúi xuống. Camera được lật gương nên trái/phải trên màn hình trùng với trái/phải của người dùng.
- Notebook có phép kiểm tra: tạo $R$ từ yaw = 20°, pitch = −10°, roll = 5°, tách ngược lại ra đúng $[20, -10]$.

### 5. Áp ngưỡng → trạng thái

$$
\text{mắt}=\begin{cases}\text{CLOSED} & \text{EAR} < T_{\text{eye}}\\ \text{OPEN} & \text{ngược lại}\end{cases}\qquad
\text{xoay}=\begin{cases}\text{LEFT} & \text{yaw} < -T_{\text{yaw}}\\ \text{RIGHT} & \text{yaw} > T_{\text{yaw}}\\ \text{STRAIGHT} & \text{ngược lại}\end{cases}\qquad
\text{ngẩng/cúi}=\begin{cases}\text{UP} & \text{pitch} < -T_{\text{pitch}}\\ \text{DOWN} & \text{pitch} > T_{\text{pitch}}\\ \text{STRAIGHT} & \text{ngược lại}\end{cases}
$$

### 6. Realtime: chớp mắt và REAL/FAKE
- **Đếm chớp mắt:** mỗi lần mắt chuyển từ CLOSED sang OPEN thì $\text{blinks} \leftarrow \text{blinks}+1$.
- **REAL/FAKE:** lấy trung vị của 15 khung gần nhất, cần ít nhất 8 khung mới kết luận:

$$\tilde p=\operatorname{median}\left(p_{\text{real}}^{(t-14)},\dots,p_{\text{real}}^{(t)}\right),\qquad \text{REAL}\iff \tilde p \ge T_{\text{real}}$$

### 7. Tìm ngưỡng
- Mắt: $T_{\text{eye}} = P_{1}(\text{EAR})$, phân vị 1% trên mọi ảnh.
- Góc đầu: $T_{\text{yaw}} = P_{99}(\lvert\text{yaw}\rvert)$ và $T_{\text{pitch}} = P_{99}(\lvert\text{pitch}\rvert)$, trên ảnh người thật nhìn thẳng.
- REAL/FAKE (Youden, trên tập train):

$$J(T)=\text{TPR}(T)-\text{FPR}(T),\qquad T_{\text{real}}=\arg\max_T J(T)$$

### 8. Chỉ số đánh giá (trên tập test, lớp dương = REAL)

$$\text{TPR}=\frac{TP}{TP+FN},\quad \text{FAR}=\text{FPR}=\frac{FP}{FP+TN},\quad \text{FRR}=\frac{FN}{FN+TP},\quad \text{Accuracy}=\frac{TP+TN}{TP+TN+FP+FN}$$

AUC là diện tích dưới đường ROC, tức là các điểm $(\text{FPR}(T), \text{TPR}(T))$ khi $T$ chạy từ 1 về 0. AUC bằng xác suất một ảnh REAL có `p_real` cao hơn một ảnh FAKE.

## Kết quả (NUAA, ngưỡng tìm trên train, đánh giá trên test)

| Trên TEST (9.122 ảnh) | Giá trị |
|---|---|
| Ngưỡng REAL/FAKE | 0.108 |
| Accuracy | 0.9993 |
| AUC | 1.000 |
| FAR (FAKE bị nhận là REAL) | 0.10% (6/5.760) |
| FRR (REAL bị từ chối) | 0% (0/3.362) |

Biểu đồ ngưỡng (EAR, yaw, pitch), phân bố `p_real`, đường ROC, confusion matrix và bảng ngưỡng nằm ở **mục 6 (vẽ bảng)**, cuối notebook.

## Dataset

**NUAA Photograph Imposter Database** (bản trên Kaggle): đặt `dataset.zip` vào `data/`, notebook tự giải nén ra `data/raw/`. Train/test theo 4 file `.txt` gốc: train 3.491 ảnh, test 9.123 ảnh. `ClientRaw` = REAL, `ImposterRaw` = FAKE (ảnh in). Dataset không đưa lên GitHub (khoảng 400 MB).

## Cài đặt và chạy

Yêu cầu: **Python 3.10**.

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m ipykernel install --user --name liveness --display-name "Python (liveness)"
jupyter lab Face_Liveness_Detection.ipynb      # Run All từ đầu; mục 5 mở camera, q / ESC để thoát
```

- **Chạy các cell từ trên xuống.** Nếu chỉ muốn mở camera: chạy cell import, mục 1, mục 2, rồi cell camera ở mục 5. Ngưỡng có sẵn trong `results/thresholds.json`.
- Chấm điểm toàn bộ dataset mất khoảng 15 phút trên CPU. Kết quả được lưu cache ở `results/frame_scores.csv`, lần sau chỉ mất vài giây.
- Ngưỡng REAL 0.108 khá lỏng với ảnh/video trên điện thoại hoặc màn hình thật. Muốn khắt khe hơn, sửa `"real"` trong `results/thresholds.json` (ví dụ 0.9) rồi chạy lại cell camera.

## Tài liệu

Tóm tắt kiến thức (LaTeX): [docs/tom_tat_kien_thuc.pdf](docs/tom_tat_kien_thuc.pdf) (mã nguồn: [docs/tom_tat_kien_thuc.tex](docs/tom_tat_kien_thuc.tex)).
