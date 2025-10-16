"""
Real-time Waste Classification using Webcam + YOLO Object Detection
Phân loại rác thải real-time qua webcam với nhận diện vật thể

Usage:
    python src/predict_realtime.py
    python src/predict_realtime.py --model outputs/models/baseline_final.keras
    python src/predict_realtime.py --camera 1 --confidence 0.5
"""

import os
import sys
import argparse
import time
from datetime import datetime
import numpy as np
import cv2
from PIL import Image

# Import config
sys.path.insert(0, os.path.dirname(__file__))
from config import CLASS_NAMES, IMG_SIZE, MODELS_DIR, SCREENSHOTS_DIR

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

try:
    import cv2
except ImportError:
    print("❌ Không tìm thấy OpenCV!")
    print("💡 Cài đặt: pip install opencv-python")
    sys.exit(1)

try:
    from ultralytics import YOLO
except ImportError:
    print("❌ Không tìm thấy Ultralytics YOLO!")
    print("💡 Cài đặt: pip install ultralytics")
    sys.exit(1)


class RealtimePredictor:
    """Class xử lý dự đoán real-time từ webcam với YOLO object detection"""
    
    def __init__(self, model_path, camera_id=0, confidence_threshold=0.5):
        """
        Khởi tạo predictor
        
        Args:
            model_path: Đường dẫn đến waste classification model
            camera_id: ID của camera (default: 0)
            confidence_threshold: Ngưỡng confidence tối thiểu (default: 0.5)
        """
        self.model_path = model_path
        self.camera_id = camera_id
        self.confidence_threshold = confidence_threshold
        self.classifier_model = None
        self.yolo_model = None
        self.cap = None
        
        # Color mapping cho từng loại rác (BGR format for OpenCV)
        self.colors = {
            'battery': (0, 165, 255),      # Orange
            'biological': (0, 255, 0),     # Green
            'cardboard': (139, 69, 19),    # Brown
            'clothes': (255, 0, 255),      # Magenta
            'glass': (255, 255, 0),        # Cyan
            'metal': (128, 128, 128),      # Gray
            'paper': (255, 255, 255),      # White
            'plastic': (0, 0, 255),        # Red
            'shoes': (255, 0, 0),          # Blue
            'trash': (0, 0, 0)             # Black
        }
        
        # FPS tracking
        self.fps_history = []
        self.max_fps_history = 30
        
    def load_models(self):
        """Load cả classification model và YOLO model"""
        # Load waste classifier
        print(f"🔄 Đang load waste classification model: {self.model_path}")
        
        if not os.path.exists(self.model_path):
            print(f"❌ Không tìm thấy model: {self.model_path}")
            print(f"\n💡 Các model có sẵn:")
            if os.path.exists(MODELS_DIR):
                for f in os.listdir(MODELS_DIR):
                    if f.endswith('.keras'):
                        print(f"   - {os.path.join(MODELS_DIR, f)}")
            return False
        
        try:
            self.classifier_model = keras.models.load_model(self.model_path)
            print(f"✅ Đã load waste classifier thành công!")
        except Exception as e:
            print(f"❌ Lỗi khi load model: {e}")
            return False
        
        # Load YOLO model
        print(f"🔄 Đang load YOLO model (YOLOv8n)...")
        try:
            self.yolo_model = YOLO('yolov8n.pt')  # Nano model, nhẹ nhất
            print(f"✅ Đã load YOLO thành công!")
        except Exception as e:
            print(f"❌ Lỗi khi load YOLO: {e}")
            print("💡 YOLO sẽ tự động download lần đầu chạy (~6MB)")
            return False
        
        return True
    
    def setup_camera(self):
        """Thiết lập camera"""
        print(f"📹 Đang mở camera {self.camera_id}...")
        
        self.cap = cv2.VideoCapture(self.camera_id)
        
        if not self.cap.isOpened():
            print(f"❌ Không thể mở camera {self.camera_id}")
            print("\n💡 Thử:")
            print("   - Kiểm tra camera có hoạt động không")
            print("   - Thử camera khác: --camera 1")
            print("   - Kiểm tra quyền truy cập camera")
            return False
        
        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print(f"✅ Camera đã sẵn sàng!")
        return True
    
    def preprocess_crop(self, crop_bgr):
        """
        Tiền xử lý crop của object - CHUẨN HÓA GIỐNG predict.py
        
        Args:
            crop_bgr: Crop từ frame (BGR format)
            
        Returns:
            numpy array đã được xử lý
        """
        # Convert BGR sang RGB
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        
        # Chuyển sang PIL Image (giống predict.py)
        pil_img = Image.fromarray(crop_rgb)
        
        # Resize với PIL (giống predict.py)
        pil_img = pil_img.resize(IMG_SIZE, Image.LANCZOS)
        
        # Convert sang array và normalize
        img_array = np.array(pil_img, dtype='float32') / 255.0
        
        # Thêm batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def classify_crop(self, crop):
        """
        Classify một crop của object
        
        Args:
            crop: Crop từ frame (BGR)
            
        Returns:
            (class_name, confidence)
        """
        # Tiền xử lý với PIL (chuẩn hóa)
        processed = self.preprocess_crop(crop)
        
        # Dự đoán
        predictions = self.classifier_model.predict(processed, verbose=0)
        probabilities = predictions[0]
        
        # Lấy top 1
        top_idx = np.argmax(probabilities)
        class_name = CLASS_NAMES[top_idx]
        confidence = float(probabilities[top_idx])
        
        return class_name, confidence
    
    def detect_and_classify(self, frame):
        """
        Detect objects với YOLO và classify từng object
        
        Args:
            frame: Frame từ camera (BGR)
            
        Returns:
            List of detections: [(x1, y1, x2, y2, class_name, confidence), ...]
        """
        # YOLO detection
        results = self.yolo_model(frame, verbose=False)[0]
        
        detections = []
        
        # Xử lý từng detection
        for box in results.boxes:
            # Lấy bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Đảm bảo bounding box hợp lệ
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            # Kiểm tra kích thước tối thiểu (ít nhất 50x50 pixels)
            if (x2 - x1) < 50 or (y2 - y1) < 50:
                continue
            
            # Crop object
            crop = frame[y1:y2, x1:x2]
            
            # Classify crop
            waste_class, waste_conf = self.classify_crop(crop)
            
            # Chỉ giữ lại nếu confidence đủ cao
            if waste_conf >= self.confidence_threshold:
                detections.append((x1, y1, x2, y2, waste_class, waste_conf))
        
        return detections
    
    def draw_detections(self, frame, detections, fps):
        """
        Vẽ bounding boxes và labels lên frame
        
        Args:
            frame: Frame gốc
            detections: List of (x1, y1, x2, y2, class_name, confidence)
            fps: FPS hiện tại
            
        Returns:
            Frame đã vẽ
        """
        height, width = frame.shape[:2]
        
        # Vẽ FPS và title ở góc trên
        cv2.rectangle(frame, (10, 10), (width - 10, 80), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (width - 10, 80), (100, 100, 100), 2)
        
        cv2.putText(frame, "WASTE CLASSIFIER + YOLO", (20, 40),
                   cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"FPS: {fps:.1f} | Objects: {len(detections)}", (20, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        
        # Vẽ từng detection
        for (x1, y1, x2, y2, class_name, confidence) in detections:
            color = self.colors.get(class_name, (255, 255, 255))
            
            # Vẽ bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            
            # Vẽ label background
            label = f"{class_name.upper()}: {confidence * 100:.1f}%"
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            )
            
            # Label box
            cv2.rectangle(frame, 
                         (x1, y1 - label_h - 10), 
                         (x1 + label_w + 10, y1), 
                         color, -1)
            
            # Label text
            cv2.putText(frame, label, (x1 + 5, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            # Vẽ confidence bar bên trong box
            bar_height = 10
            bar_width = int((x2 - x1) * confidence)
            cv2.rectangle(frame, 
                         (x1, y2 - bar_height), 
                         (x1 + bar_width, y2), 
                         color, -1)
        
        # Instructions
        cv2.putText(frame, "Press 'q' to quit | 's' to screenshot", 
                   (20, height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        return frame
    
    def save_screenshot(self, frame):
        """
        Lưu screenshot
        
        Args:
            frame: Frame cần lưu
        """
        # Tạo thư mục nếu chưa có
        os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
        
        # Tạo filename với timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{timestamp}.jpg"
        filepath = os.path.join(SCREENSHOTS_DIR, filename)
        
        # Lưu
        cv2.imwrite(filepath, frame)
        print(f"\n📸 Screenshot saved: {filepath}")
    
    def run(self):
        """Main loop - chạy real-time prediction với YOLO"""
        
        # Load models
        if not self.load_models():
            return
        
        # Setup camera
        if not self.setup_camera():
            return
        
        print("\n" + "="*60)
        print("🎥 REAL-TIME WASTE CLASSIFICATION + YOLO")
        print("="*60)
        print("📹 Camera đang chạy...")
        print("🤖 YOLO đang detect objects...")
        print("⌨️  Nhấn 'q' để thoát")
        print("⌨️  Nhấn 's' để chụp ảnh")
        print("="*60 + "\n")
        
        try:
            frame_count = 0
            
            while True:
                start_time = time.time()
                
                # Đọc frame
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Không thể đọc frame từ camera")
                    break
                
                frame_count += 1
                
                # Detect và classify (mỗi 2 frames để tăng FPS)
                if frame_count % 2 == 0:
                    detections = self.detect_and_classify(frame)
                else:
                    # Giữ detections cũ
                    if frame_count == 1:
                        detections = []
                
                # Tính FPS
                elapsed = time.time() - start_time
                fps = 1.0 / elapsed if elapsed > 0 else 0
                self.fps_history.append(fps)
                if len(self.fps_history) > self.max_fps_history:
                    self.fps_history.pop(0)
                avg_fps = sum(self.fps_history) / len(self.fps_history)
                
                # Vẽ detections
                display_frame = self.draw_detections(frame, detections, avg_fps)
                
                # Hiển thị
                cv2.imshow('Waste Classifier + YOLO - Real-time', display_frame)
                
                # Xử lý keyboard
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\n👋 Thoát...")
                    break
                elif key == ord('s'):
                    self.save_screenshot(display_frame)
        
        except KeyboardInterrupt:
            print("\n\n👋 Thoát (Ctrl+C)...")
        
        finally:
            # Cleanup
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
            print("✅ Đã đóng camera và cửa sổ")


def main():
    parser = argparse.ArgumentParser(
        description='Real-time waste classification using webcam + YOLO object detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  python src/predict_realtime.py
  python src/predict_realtime.py --model outputs/models/baseline_final.keras
  python src/predict_realtime.py --camera 1 --confidence 0.6
  
Features:
  ✅ YOLO object detection - Detect nhiều vật thể
  ✅ Bounding boxes - Viền xung quanh vật thể
  ✅ Real-time classification - Phân loại từng vật thể
  ✅ High accuracy - Preprocessing chuẩn với PIL
  
Keyboard shortcuts:
  q - Quit
  s - Save screenshot
        """
    )
    
    parser.add_argument('--model', type=str,
                       default=os.path.join(MODELS_DIR, 'mobilenetv2_final.keras'),
                       help='Đường dẫn đến waste classification model')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera device ID (default: 0)')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Minimum confidence threshold (default: 0.5)')
    
    args = parser.parse_args()
    
    # Tạo predictor và chạy
    predictor = RealtimePredictor(
        model_path=args.model,
        camera_id=args.camera,
        confidence_threshold=args.confidence
    )
    
    predictor.run()


if __name__ == "__main__":
    main()
