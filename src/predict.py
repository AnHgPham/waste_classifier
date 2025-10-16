"""
Script dự đoán ảnh - Đơn giản và chắc chắn chạy được

Usage:
    python src/predict.py test.jpg
    python src/predict.py test.jpg --model outputs/models/baseline_final.keras
    python src/predict.py test.jpg --top_k 5
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image

# Import config
sys.path.insert(0, os.path.dirname(__file__))
from config import CLASS_NAMES, IMG_SIZE, MODELS_DIR

# Tắt warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError:
    print("❌ Không tìm thấy TensorFlow!")
    print("💡 Cài đặt: pip install tensorflow")
    sys.exit(1)

def load_model(model_path):
    """Load model - Đơn giản, không phức tạp"""
    print(f"🔄 Đang load model: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ Không tìm thấy model: {model_path}")
        print(f"\n💡 Các model có sẵn:")
        if os.path.exists(MODELS_DIR):
            for f in os.listdir(MODELS_DIR):
                if f.endswith('.keras'):
                    print(f"   - {os.path.join(MODELS_DIR, f)}")
        return None
    
    try:
        model = keras.models.load_model(model_path)
        print(f"✅ Đã load model thành công!")
        return model
    except Exception as e:
        print(f"❌ Lỗi khi load model: {e}")
        return None

def preprocess_image(image_path):
    """Tiền xử lý ảnh"""
    try:
        # Đọc ảnh
        img = Image.open(image_path)
        
        # Convert sang RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize
        img = img.resize(IMG_SIZE)
        
        # Chuyển sang array và normalize
        img_array = np.array(img, dtype='float32') / 255.0
        
        # Thêm batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    except Exception as e:
        print(f"❌ Lỗi khi xử lý ảnh: {e}")
        return None

def predict_image(model, image_path, top_k=3):
    """Dự đoán ảnh"""
    
    print(f"\n🔍 Đang phân tích: {image_path}")
    
    if not os.path.exists(image_path):
        print(f"❌ Không tìm thấy ảnh: {image_path}")
        return None
    
    # Tiền xử lý
    img_array = preprocess_image(image_path)
    if img_array is None:
        return None
    
    # Dự đoán
    print("   Đang dự đoán...")
    predictions = model.predict(img_array, verbose=0)
    probabilities = predictions[0]
    
    # Lấy top K
    top_indices = np.argsort(probabilities)[::-1][:top_k]
    
    # Hiển thị kết quả
    print("\n" + "="*60)
    print("📊 KẾT QUẢ DỰ ĐOÁN:")
    print("="*60)
    
    results = []
    for i, idx in enumerate(top_indices):
        class_name = CLASS_NAMES[idx]
        confidence = probabilities[idx] * 100
        results.append((class_name, confidence))
        
        if i == 0:
            print(f"🥇 {class_name.upper():<15} {confidence:>6.2f}%  ⭐")
        else:
            print(f"{i+1}. {class_name:<15} {confidence:>6.2f}%")
    
    print("="*60)
    
    return results

def main():
    parser = argparse.ArgumentParser(
        description='Dự đoán loại rác thải từ ảnh',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  python src/predict.py test.jpg
  python src/predict.py test.jpg --model outputs/models/baseline_final.keras
  python src/predict.py test.jpg --top_k 5
        """
    )
    
    parser.add_argument('image', type=str, 
                       help='Đường dẫn đến ảnh cần dự đoán')
    parser.add_argument('--model', type=str, 
                       default=os.path.join(MODELS_DIR, 'mobilenetv2_final.keras'),
                       help='Đường dẫn đến file model')
    parser.add_argument('--top_k', type=int, default=3,
                       help='Số lượng dự đoán hàng đầu (mặc định: 3)')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("🗑️  GARBAGE CLASSIFIER - PHÂN LOẠI RÁC THẢI")
    print("="*60)
    
    # Load model
    model = load_model(args.model)
    if model is None:
        sys.exit(1)
    
    # Dự đoán
    results = predict_image(model, args.image, top_k=args.top_k)
    
    if results:
        print(f"\n✅ Hoàn thành! Ảnh này là: {results[0][0].upper()} ({results[0][1]:.1f}%)")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("\n❌ Thiếu đường dẫn ảnh!")
        print("\nCách dùng:")
        print("  python src/predict.py <đường_dẫn_ảnh>")
        print("\nVí dụ:")
        print("  python src/predict.py test.jpg")
        print("\nĐể xem đầy đủ options:")
        print("  python src/predict.py --help")
        sys.exit(1)
    
    main()

