"""
Script đánh giá model trên test set

Usage:
    python src/evaluate.py
    python src/evaluate.py --model outputs/models/baseline_final.keras
"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Import config và utils
from config import *
from utils import *

def evaluate_model(model_path):
    """Đánh giá model"""
    
    print_section(f"ĐÁNH GIÁ MODEL: {os.path.basename(model_path)}")
    
    # Load model
    print(f"🔄 Đang load model...")
    model = keras.models.load_model(model_path)
    print(f"✅ Đã load model")
    
    # Load test dataset
    test_dir = os.path.join(PROCESSED_DATA_DIR, 'test')
    
    if not os.path.exists(test_dir):
        print("❌ Chưa có test set! Chạy: python src/data_prep.py")
        sys.exit(1)
    
    print(f"\n🔄 Đang load test dataset...")
    test_ds = keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical',
        shuffle=False
    )
    
    # Normalize
    normalization = keras.layers.Rescaling(1./255)
    test_ds = test_ds.map(lambda x, y: (normalization(x), y))
    
    print(f"✅ Test batches: {len(test_ds)}")
    
    # Evaluate
    print(f"\n🔄 Đang đánh giá...")
    loss, accuracy = model.evaluate(test_ds, verbose=1)
    
    print(f"\n📊 KẾT QUẢ:")
    print(f"   Loss: {loss:.4f}")
    print(f"   Accuracy: {accuracy*100:.2f}%")
    
    # Dự đoán để tạo confusion matrix
    print(f"\n🔄 Đang tạo confusion matrix...")
    y_true = []
    y_pred = []
    
    for images, labels in test_ds:
        predictions = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(predictions, axis=1))
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Lưu confusion matrix
    model_name = os.path.basename(model_path).replace('.keras', '')
    cm_path = os.path.join(REPORTS_DIR, f'{model_name}_confusion_matrix.png')
    save_confusion_matrix(y_true, y_pred, CLASS_NAMES, cm_path)
    
    # Lưu classification report
    report_path = os.path.join(REPORTS_DIR, f'{model_name}_classification_report.txt')
    report = save_classification_report(y_true, y_pred, CLASS_NAMES, report_path)
    
    print(f"\n{report}")
    
    print_section("✅ HOÀN TẤT ĐÁNH GIÁ!")

def main():
    parser = argparse.ArgumentParser(description='Đánh giá model trên test set')
    parser.add_argument('--model', type=str, 
                       default=os.path.join(MODELS_DIR, 'mobilenetv2_final.keras'),
                       help='Đường dẫn đến file model')
    
    args = parser.parse_args()
    
    evaluate_model(args.model)

if __name__ == "__main__":
    main()

