# 🚀 Deployment Guide

This guide covers various deployment options for the Waste Classifier project.

## 📋 Table of Contents

- [Local Deployment](#local-deployment)
- [Docker Deployment](#docker-deployment)
- [Cloud Deployment](#cloud-deployment)
  - [AWS](#aws-deployment)
  - [Google Cloud](#google-cloud-deployment)
  - [Azure](#azure-deployment)
- [Edge Deployment](#edge-deployment)
- [API Deployment](#api-deployment)

---

## 🏠 Local Deployment

### Production Setup

```bash
# Clone repository
git clone https://github.com/AnHgPham/waste_classifier.git
cd waste_classifier

# Create production environment
python -m venv venv_prod
source venv_prod/bin/activate  # or venv_prod\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Download trained model (if not included)
# Place model in outputs/models/

# Run predictions
python src/predict.py path/to/image.jpg

# Or run real-time detection
python src/predict_realtime.py
```

---

## 🐳 Docker Deployment

### Dockerfile

Create `Dockerfile`:

```dockerfile
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY outputs/models/ ./outputs/models/

# Expose port (if using API)
EXPOSE 8000

# Run application
CMD ["python", "src/predict_realtime.py"]
```

### Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  waste-classifier:
    build: .
    container_name: waste_classifier
    volumes:
      - ./outputs:/app/outputs
      - /dev/video0:/dev/video0  # For webcam access
    devices:
      - /dev/video0:/dev/video0
    environment:
      - DISPLAY=${DISPLAY}
    network_mode: host
    restart: unless-stopped
```

### Build and Run

```bash
# Build image
docker build -t waste-classifier:latest .

# Run container
docker run -it --rm \
  -v $(pwd)/outputs:/app/outputs \
  --device /dev/video0:/dev/video0 \
  waste-classifier:latest

# With docker-compose
docker-compose up
```

---

## ☁️ Cloud Deployment

### AWS Deployment

#### Option 1: EC2 Instance

```bash
# Launch EC2 instance (Ubuntu 22.04, t3.medium or better)
# Connect via SSH

# Install dependencies
sudo apt update
sudo apt install -y python3-pip git

# Clone and setup
git clone https://github.com/AnHgPham/waste_classifier.git
cd waste_classifier
pip3 install -r requirements.txt

# Run application
python3 src/predict.py image.jpg
```

#### Option 2: AWS Lambda (for API)

Create `lambda_handler.py`:

```python
import json
import base64
import numpy as np
from PIL import Image
from io import BytesIO
from tensorflow import keras

model = None

def load_model():
    global model
    if model is None:
        model = keras.models.load_model('mobilenetv2_final.keras')
    return model

def lambda_handler(event, context):
    try:
        # Decode image
        image_data = base64.b64decode(event['body'])
        image = Image.open(BytesIO(image_data))
        
        # Preprocess
        image = image.resize((224, 224))
        img_array = np.array(image, dtype='float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict
        model = load_model()
        predictions = model.predict(img_array)
        
        # Get top class
        class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][class_idx])
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'class': CLASS_NAMES[class_idx],
                'confidence': confidence
            })
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
```

#### Option 3: Amazon SageMaker

```python
# Train and deploy on SageMaker
import sagemaker
from sagemaker.tensorflow import TensorFlow

# Define estimator
estimator = TensorFlow(
    entry_point='train.py',
    source_dir='src',
    role=sagemaker.get_execution_role(),
    instance_count=1,
    instance_type='ml.p3.2xlarge',
    framework_version='2.13',
    py_version='py310'
)

# Train
estimator.fit({'training': 's3://bucket/data/train'})

# Deploy
predictor = estimator.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.xlarge'
)
```

### Google Cloud Deployment

#### Cloud Run

```bash
# Build container
gcloud builds submit --tag gcr.io/PROJECT_ID/waste-classifier

# Deploy
gcloud run deploy waste-classifier \
  --image gcr.io/PROJECT_ID/waste-classifier \
  --platform managed \
  --region us-central1 \
  --memory 2Gi \
  --cpu 2
```

#### AI Platform

```bash
# Upload model to Cloud Storage
gsutil cp outputs/models/mobilenetv2_final.keras gs://BUCKET/models/

# Create model version
gcloud ai-platform versions create v1 \
  --model waste_classifier \
  --runtime-version 2.13 \
  --python-version 3.10 \
  --framework tensorflow \
  --origin gs://BUCKET/models/
```

### Azure Deployment

#### Azure Container Instances

```bash
# Create resource group
az group create --name waste-classifier-rg --location eastus

# Create container instance
az container create \
  --resource-group waste-classifier-rg \
  --name waste-classifier \
  --image YOUR_REGISTRY/waste-classifier:latest \
  --cpu 2 --memory 4 \
  --ports 8000
```

---

## 📱 Edge Deployment

### Raspberry Pi

```bash
# Install dependencies on Raspberry Pi
sudo apt update
sudo apt install -y python3-pip python3-opencv

# Install TensorFlow Lite
pip3 install tflite-runtime

# Convert model to TFLite
python3 -c "
import tensorflow as tf

model = tf.keras.models.load_model('mobilenetv2_final.keras')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
"

# Run inference with TFLite
python3 src/predict_tflite.py
```

### NVIDIA Jetson

```bash
# Install JetPack SDK
# Then install dependencies

sudo apt install python3-pip
pip3 install tensorflow-gpu opencv-python

# Run with GPU acceleration
python3 src/predict_realtime.py
```

---

## 🌐 API Deployment

### FastAPI REST API

Create `api/main.py`:

```python
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI(title="Waste Classifier API")

# Load model at startup
model = tf.keras.models.load_model('outputs/models/mobilenetv2_final.keras')

CLASS_NAMES = ['battery', 'biological', 'cardboard', 'clothes', 'glass',
               'metal', 'paper', 'plastic', 'shoes', 'trash']

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Classify waste image"""
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Preprocess
        image = image.resize((224, 224))
        img_array = np.array(image, dtype='float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict
        predictions = model.predict(img_array)
        
        # Get results
        results = []
        for idx in np.argsort(predictions[0])[::-1][:3]:
            results.append({
                'class': CLASS_NAMES[idx],
                'confidence': float(predictions[0][idx])
            })
        
        return JSONResponse(content={
            'success': True,
            'predictions': results
        })
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={'success': False, 'error': str(e)}
        )

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {'status': 'healthy'}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
```

Run API:

```bash
# Install FastAPI
pip install fastapi uvicorn python-multipart

# Run server
python api/main.py

# Test
curl -X POST "http://localhost:8000/predict" \
  -F "file=@test_image.jpg"
```

### Flask API

Create `api/app.py`:

```python
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
model = tf.keras.models.load_model('outputs/models/mobilenetv2_final.keras')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    image = Image.open(file.stream)
    
    # Preprocess and predict
    image = image.resize((224, 224))
    img_array = np.array(image, dtype='float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions[0])
    
    return jsonify({
        'class': CLASS_NAMES[class_idx],
        'confidence': float(predictions[0][class_idx])
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

---

## 🔒 Security Considerations

### Environment Variables

Create `.env` file (never commit this):

```bash
MODEL_PATH=/path/to/model.keras
API_KEY=your_secret_api_key
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
```

### API Authentication

```python
from fastapi import Header, HTTPException

async def verify_token(x_api_key: str = Header(...)):
    if x_api_key != os.getenv('API_KEY'):
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return x_api_key

@app.post("/predict", dependencies=[Depends(verify_token)])
async def predict(...):
    # Your code here
    pass
```

### Rate Limiting

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/predict")
@limiter.limit("10/minute")
async def predict(...):
    # Your code here
    pass
```

---

## 📊 Monitoring

### Application Metrics

```python
from prometheus_client import Counter, Histogram
import time

prediction_counter = Counter('predictions_total', 'Total predictions')
prediction_duration = Histogram('prediction_duration_seconds', 'Prediction duration')

@app.post("/predict")
async def predict(...):
    start_time = time.time()
    
    # Your prediction code
    
    prediction_counter.inc()
    prediction_duration.observe(time.time() - start_time)
    
    return result
```

### Logging

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

@app.post("/predict")
async def predict(...):
    logger.info(f"Received prediction request from {request.client.host}")
    # Your code
    logger.info(f"Predicted: {result}")
```

---

## 🎯 Performance Optimization

### Model Optimization

```python
# Convert to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Quantization
converter.target_spec.supported_types = [tf.float16]
```

### Caching

```python
from functools import lru_cache

@lru_cache(maxsize=100)
def get_model():
    return tf.keras.models.load_model('model.keras')
```

---

## 📝 Deployment Checklist

- [ ] Model trained and saved
- [ ] Dependencies documented in requirements.txt
- [ ] Environment variables configured
- [ ] API authentication implemented
- [ ] Rate limiting configured
- [ ] Logging enabled
- [ ] Error handling implemented
- [ ] Health check endpoint added
- [ ] Security headers configured
- [ ] HTTPS enabled
- [ ] Monitoring dashboard setup
- [ ] Backup strategy defined
- [ ] Documentation updated

---

For more information, see [README.md](README.md) or [CONTRIBUTING.md](CONTRIBUTING.md).

