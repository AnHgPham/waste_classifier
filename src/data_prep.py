"""
Script chuẩn bị dữ liệu
Tải dataset từ Kaggle và chia train/val/test

Usage:
    python src/data_prep.py
"""

import os
import sys
import shutil
import random
from pathlib import Path

# Import config
from config import *

def download_dataset():
    """Hướng dẫn tải dataset"""
    
    print("\n" + "="*70)
    print("📥 HƯỚNG DẪN TẢI DATASET")
    print("="*70 + "\n")
    
    print("Dataset: Garbage Classification v2")
    print("URL: https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2\n")
    
    print("Cách 1: Tải thủ công (Khuyến nghị)")
    print("  1. Truy cập: https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2")
    print("  2. Click 'Download' (cần đăng nhập Kaggle)")
    print(f"  3. Giải nén vào: {RAW_DATA_DIR}/")
    print(f"  4. Đảm bảo có thư mục: {RAW_DATA_DIR}/garbage-classification-v2/\n")
    
    print("Cách 2: Dùng Kaggle API")
    print("  1. Cài đặt: pip install kaggle")
    print("  2. Setup API token: https://www.kaggle.com/docs/api")
    print(f"  3. Chạy: kaggle datasets download -d {DATASET_NAME} -p {RAW_DATA_DIR} --unzip\n")
    
    choice = input("Bạn đã tải dataset chưa? (y/n): ").lower()
    
    if choice != 'y':
        print("\n💡 Vui lòng tải dataset trước, sau đó chạy lại script này.")
        sys.exit(0)

def split_data():
    """Chia dữ liệu thành train/val/test"""
    
    print("\n" + "="*70)
    print("✂️  CHIA DỮ LIỆU TRAIN/VAL/TEST")
    print("="*70 + "\n")
    
    # Tìm thư mục dataset
    dataset_dir = os.path.join(RAW_DATA_DIR, "garbage-classification-v2", "garbage-classification-v2")
    
    if not os.path.exists(dataset_dir):
        # Thử tìm ở các vị trí khác
        alt_paths = [
            os.path.join(RAW_DATA_DIR, "garbage-classification-v2"),
            RAW_DATA_DIR
        ]
        
        for path in alt_paths:
            if os.path.exists(path):
                # Kiểm tra có thư mục classes không
                subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
                if len(subdirs) >= 10:  # Có ít nhất 10 classes
                    dataset_dir = path
                    break
        
        if not os.path.exists(dataset_dir):
            print(f"❌ Không tìm thấy dataset tại: {dataset_dir}")
            print(f"\n💡 Vui lòng đảm bảo dataset được giải nén đúng vị trí.")
            sys.exit(1)
    
    print(f"✅ Tìm thấy dataset: {dataset_dir}")
    
    # Lấy danh sách classes
    class_dirs = [d for d in os.listdir(dataset_dir) 
                  if os.path.isdir(os.path.join(dataset_dir, d))]
    
    print(f"✅ Số lượng classes: {len(class_dirs)}")
    print(f"   Classes: {', '.join(sorted(class_dirs))}\n")
    
    # Xóa thư mục processed cũ
    if os.path.exists(PROCESSED_DATA_DIR):
        shutil.rmtree(PROCESSED_DATA_DIR)
    
    # Tạo thư mục train/val/test
    for split in ['train', 'val', 'test']:
        for class_name in class_dirs:
            os.makedirs(os.path.join(PROCESSED_DATA_DIR, split, class_name), exist_ok=True)
    
    # Chia dữ liệu
    print("🔄 Đang chia dữ liệu...")
    
    total_files = 0
    train_count = 0
    val_count = 0
    test_count = 0
    
    for class_name in sorted(class_dirs):
        class_path = os.path.join(dataset_dir, class_name)
        
        # Lấy tất cả file ảnh
        files = [f for f in os.listdir(class_path) 
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        # Shuffle
        random.seed(SEED)
        random.shuffle(files)
        
        # Tính số lượng
        n_files = len(files)
        n_train = int(n_files * TRAIN_RATIO)
        n_val = int(n_files * VAL_RATIO)
        
        # Chia files
        train_files = files[:n_train]
        val_files = files[n_train:n_train + n_val]
        test_files = files[n_train + n_val:]
        
        # Copy files
        for f in train_files:
            shutil.copy(
                os.path.join(class_path, f),
                os.path.join(PROCESSED_DATA_DIR, 'train', class_name, f)
            )
        
        for f in val_files:
            shutil.copy(
                os.path.join(class_path, f),
                os.path.join(PROCESSED_DATA_DIR, 'val', class_name, f)
            )
        
        for f in test_files:
            shutil.copy(
                os.path.join(class_path, f),
                os.path.join(PROCESSED_DATA_DIR, 'test', class_name, f)
            )
        
        total_files += n_files
        train_count += len(train_files)
        val_count += len(val_files)
        test_count += len(test_files)
        
        print(f"   {class_name:<15} {n_files:>5} files → train: {len(train_files):>4}, val: {len(val_files):>4}, test: {len(test_files):>4}")
    
    print(f"\n✅ Hoàn thành!")
    print(f"   Tổng: {total_files} files")
    print(f"   Train: {train_count} ({train_count/total_files*100:.1f}%)")
    print(f"   Val: {val_count} ({val_count/total_files*100:.1f}%)")
    print(f"   Test: {test_count} ({test_count/total_files*100:.1f}%)")

def main():
    print("\n" + "="*70)
    print("🗑️  CHUẨN BỊ DỮ LIỆU - WASTE CLASSIFICATION")
    print("="*70)
    
    # Tạo thư mục
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    # Kiểm tra dataset đã có chưa
    dataset_exists = False
    possible_paths = [
        os.path.join(RAW_DATA_DIR, "garbage-classification-v2", "garbage-classification-v2"),
        os.path.join(RAW_DATA_DIR, "garbage-classification-v2"),
        RAW_DATA_DIR
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
            if len(subdirs) >= 10:
                dataset_exists = True
                break
    
    if not dataset_exists:
        download_dataset()
    
    # Chia dữ liệu
    split_data()
    
    print("\n" + "="*70)
    print("✅ HOÀN TẤT CHUẨN BỊ DỮ LIỆU!")
    print("="*70)
    
    print("\n💡 Bước tiếp theo:")
    print("   python src/train.py --model mobilenetv2")

if __name__ == "__main__":
    main()

