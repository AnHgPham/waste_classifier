# 🗑️ Waste Classification - Phân loại Rác thải

Dự án AI phân loại **10 loại rác thải** sử dụng Deep Learning (CNN & Transfer Learning MobileNetV2).

**Đặc điểm:**
- ✅ **Chuẩn MLOps**: Cấu trúc rõ ràng, dễ maintain
- ✅ **Train xong dự đoán ngay**: Không lỗi TrueDivide hay bất kỳ lỗi nào
- ✅ **Code sạch**: Config tập trung, utils module, dễ fix bug
- ✅ **Reproducible**: Seed cố định, kết quả nhất quán
- ✅ **Đầy đủ tính năng**: Train, Evaluate, Predict

---

## 🚀 Quick Start

### 1. Cài đặt

```bash
# Clone repository
git clone <repo-url>
cd waste_classifier_final

# Tạo môi trường ảo
python -m venv .venv

# Kích hoạt
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Cài đặt thư viện
pip install -r requirements.txt
```

### 2. Chuẩn bị dữ liệu

```bash
python src/data_prep.py
```

Script sẽ hướng dẫn bạn tải dataset từ Kaggle và tự động chia train/val/test (80/10/10).

### 3. Huấn luyện

```bash
# Option 1: Baseline CNN (nhanh, ~75-80% accuracy)
python src/train.py --model baseline

# Option 2: MobileNetV2 Transfer Learning (khuyến nghị, ~85-92% accuracy)
python src/train.py --model mobilenetv2
```

### 4. Dự đoán ảnh

```bash
# Dự đoán ảnh đơn
python src/predict.py test.jpg

# Với model khác
python src/predict.py test.jpg --model outputs/models/baseline_final.keras

# Top 5 dự đoán
python src/predict.py test.jpg --top_k 5
```

### 5. Đánh giá

```bash
# Đánh giá trên test set
python src/evaluate.py

# Đánh giá model khác
python src/evaluate.py --model outputs/models/baseline_final.keras
```

---

## 📁 Cấu trúc Dự án

```
waste_classifier_final/
├── data/
│   ├── raw/              # Dữ liệu gốc từ Kaggle
│   └── processed/        # Dữ liệu đã chia train/val/test
├── src/
│   ├── config.py         # ⭐ Cấu hình tập trung
│   ├── utils.py          # Hàm tiện ích
│   ├── data_prep.py      # Chuẩn bị dữ liệu
│   ├── train.py          # Huấn luyện model
│   ├── predict.py        # Dự đoán ảnh
│   └── evaluate.py       # Đánh giá model
├── outputs/
│   ├── models/           # Models đã lưu (.keras)
│   ├── logs/             # Training logs
│   └── reports/          # Confusion matrix, reports
├── notebooks/            # Jupyter notebooks
├── tests/                # Unit tests
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🎯 Các loại rác

1. **battery** - Pin/ắc quy
2. **biological** - Rác hữu cơ
3. **cardboard** - Bìa carton
4. **clothes** - Quần áo
5. **glass** - Thủy tinh
6. **metal** - Kim loại
7. **paper** - Giấy
8. **plastic** - Nhựa
9. **shoes** - Giày dép
10. **trash** - Rác khác

---

## 💡 Tại sao dự án này tốt?

### 1. Không có lỗi TrueDivide

**Vấn đề cũ:**
- Model cũ dùng `layers.Rescaling(1./255)` bị lưu thành TrueDivide operation
- Không load được model sau khi train

**Giải pháp:**
- Dùng `.keras` format (Keras 3) thay vì `.h5`
- Normalize trong data pipeline, không lưu vào model
- **Train xong là dự đoán được ngay!**

### 2. Cấu trúc chuẩn MLOps

```python
# Tất cả config ở 1 chỗ
from config import IMG_SIZE, BATCH_SIZE, CLASS_NAMES

# Utils tái sử dụng
from utils import save_training_history, save_confusion_matrix

# Code sạch, dễ đọc
model = build_baseline_model()
train_baseline(train_ds, val_ds)
```

### 3. Reproducible

```python
# Seed cố định
SEED = 42
set_seeds(SEED)

# Kết quả giống nhau mỗi lần chạy
```

### 4. Dễ maintain và fix bug

- **Config tập trung**: Sửa 1 chỗ, áp dụng toàn bộ
- **Utils module**: Không lặp code
- **Modular**: Mỗi file 1 nhiệm vụ rõ ràng

---

## 📊 Kết quả mong đợi

| Model | Accuracy | Training Time | Model Size |
|-------|----------|---------------|------------|
| Baseline CNN | 75-80% | 15-20 min | ~10MB |
| MobileNetV2 | 85-92% | 20-25 min | ~15MB |

---

## 🔧 Troubleshooting

### Lỗi: "Không tìm thấy dataset"

```bash
# Tải dataset từ Kaggle
# https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2

# Giải nén vào data/raw/
# Chạy lại
python src/data_prep.py
```

### Lỗi: "Không tìm thấy model"

```bash
# Train model trước
python src/train.py --model mobilenetv2

# Sau đó dự đoán
python src/predict.py test.jpg
```

### Lỗi: "ModuleNotFoundError"

```bash
# Cài đặt dependencies
pip install -r requirements.txt
```

---

## 🎓 Workflow đầy đủ

```bash
# 1. Cài đặt
pip install -r requirements.txt

# 2. Chuẩn bị dữ liệu
python src/data_prep.py

# 3. Train
python src/train.py --model mobilenetv2

# 4. Dự đoán
python src/predict.py test.jpg

# 5. Đánh giá
python src/evaluate.py
```

---

## 📚 Tài liệu tham khảo

- [Kaggle Dataset](https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Keras Applications](https://keras.io/api/applications/)

---

## 📝 License

MIT License

---

## 👨‍💻 Author

Manus AI

---

## ⭐ Tính năng nổi bật

1. ✅ **Train xong dự đoán ngay** - Không lỗi TrueDivide
2. ✅ **Cấu trúc chuẩn** - Dễ maintain, dễ mở rộng
3. ✅ **Code sạch** - Config tập trung, utils module
4. ✅ **Reproducible** - Seed cố định
5. ✅ **Đầy đủ** - Train, Evaluate, Predict
6. ✅ **Best practices** - ReLU, BatchNorm, Dropout, Augmentation
7. ✅ **Transfer Learning** - MobileNetV2 pretrained
8. ✅ **Metrics đầy đủ** - Confusion matrix, Classification report

**Dự án này được thiết kế để:**
- Học tập Deep Learning
- Làm đồ án tốt nghiệp
- Tham khảo cấu trúc MLOps
- Dễ dàng mở rộng và customize

