# ⚡ Quick Start Guide

Get started with Waste Classifier in 5 minutes!

## 🎯 Prerequisites

- Python 3.8+ installed
- 5GB free disk space
- (Optional) Webcam for real-time detection
- (Optional) GPU for faster training

## 📦 Installation (2 minutes)

```bash
# 1. Clone repository
git clone https://github.com/AnHgPham/waste_classifier.git
cd waste_classifier

# 2. Create virtual environment
python -m venv .venv

# 3. Activate environment
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt
```

## 🚀 Quick Test (30 seconds)

Test with a pre-trained model (if available):

```bash
# Predict single image
python src/predict.py data/raw/battery/battery_2.jpg

# Expected output:
# 🥇 BATTERY  99.95%  ⭐
```

## 🎥 Real-time Detection (1 minute)

```bash
# Install YOLO
pip install ultralytics

# Start webcam detection
python src/predict_realtime.py

# Controls:
# - Press 'q' to quit
# - Press 's' to screenshot
```

## 🎓 Full Workflow (20-30 minutes)

### Step 1: Prepare Data (5 min)

```bash
# Download dataset from:
# https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2

# Place in data/raw/ folder, then:
python src/data_prep.py
```

### Step 2: Train Model (20-25 min)

```bash
# Train MobileNetV2 (recommended)
python src/train.py --model mobilenetv2

# Or train baseline (faster, lower accuracy)
python src/train.py --model baseline
```

### Step 3: Evaluate (1 min)

```bash
# Evaluate model
python src/evaluate.py

# Check outputs/reports/ for:
# - Confusion matrix
# - Classification report
# - Training history
```

### Step 4: Use It! 

```bash
# Single image
python src/predict.py path/to/image.jpg

# Real-time
python src/predict_realtime.py
```

## 📊 What You Get

After training, you'll have:

- ✅ Model with 85-92% accuracy
- ✅ Real-time object detection capability
- ✅ Confusion matrix and metrics
- ✅ Training history plots
- ✅ Ready-to-deploy API

## 🎯 Use Cases

### Academic/Research
```bash
# Train model
python src/train.py --model mobilenetv2

# Generate metrics for paper
python src/evaluate.py

# Results in outputs/reports/
```

### Production/Deployment
```bash
# Train model
python src/train.py --model mobilenetv2

# Test on real data
python src/predict_realtime.py

# Deploy with Docker (see DEPLOYMENT.md)
docker build -t waste-classifier .
docker run -it waste-classifier
```

### Education/Demo
```bash
# Use pre-trained model
python src/predict_realtime.py

# Show real-time classification
# Great for presentations!
```

## 🆘 Need Help?

- 📖 Full documentation: [README.md](README.md)
- 🚀 Deployment guide: [DEPLOYMENT.md](DEPLOYMENT.md)
- 🤝 Contributing: [CONTRIBUTING.md](CONTRIBUTING.md)
- 🐛 Issues: [GitHub Issues](https://github.com/AnHgPham/waste_classifier/issues)

## 🎉 Next Steps

1. ⭐ Star the repository
2. 🍴 Fork for your own experiments
3. 🤝 Contribute improvements
4. 📢 Share with others

Happy classifying! 🗑️♻️

