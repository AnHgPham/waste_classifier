"""
Script huấn luyện model
Hỗ trợ: Baseline CNN và MobileNetV2 Transfer Learning

Usage:
    python src/train.py --model baseline
    python src/train.py --model mobilenetv2
"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

# Import config và utils
from config import *
from utils import *

# Tắt warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore')

def create_data_generators():
    """Tạo data generators với augmentation"""
    
    print_section("TẠO DATA GENERATORS")
    
    train_dir = os.path.join(PROCESSED_DATA_DIR, 'train')
    val_dir = os.path.join(PROCESSED_DATA_DIR, 'val')
    
    if not os.path.exists(train_dir):
        print("❌ Chưa có dữ liệu! Chạy: python src/data_prep.py")
        sys.exit(1)
    
    # Data augmentation cho train
    if USE_AUGMENTATION:
        data_augmentation = keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(AUGMENTATION_CONFIG['rotation_factor']),
            layers.RandomZoom(AUGMENTATION_CONFIG['zoom_factor']),
            layers.RandomContrast(AUGMENTATION_CONFIG['contrast_factor'])
        ])
    else:
        data_augmentation = None
    
    # Train dataset
    train_ds = keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical',
        shuffle=True,
        seed=SEED
    )
    
    # Validation dataset
    val_ds = keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical',
        shuffle=False,
        seed=SEED
    )
    
    # Normalize và augment
    normalization = layers.Rescaling(1./255)
    
    if data_augmentation:
        train_ds = train_ds.map(lambda x, y: (data_augmentation(normalization(x), training=True), y))
    else:
        train_ds = train_ds.map(lambda x, y: (normalization(x), y))
    
    val_ds = val_ds.map(lambda x, y: (normalization(x), y))
    
    # Prefetch để tăng tốc
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    print(f"✅ Train batches: {len(train_ds)}")
    print(f"✅ Val batches: {len(val_ds)}")
    
    return train_ds, val_ds

def build_baseline_model():
    """Tạo baseline CNN model"""
    
    print_section("XÂY DỰNG BASELINE CNN")
    
    model = models.Sequential([
        layers.Input(shape=(*IMG_SIZE, 3)),
        
        # Block 1
        layers.Conv2D(32, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Dropout(0.25),
        
        # Block 2
        layers.Conv2D(64, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Dropout(0.25),
        
        # Block 3
        layers.Conv2D(128, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Dropout(0.25),
        
        # Classifier
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    
    print(f"✅ Đã tạo baseline model")
    print(f"   Total params: {model.count_params():,}")
    
    return model

def build_mobilenetv2_model():
    """Tạo MobileNetV2 transfer learning model"""
    
    print_section("XÂY DỰNG MOBILENETV2 TRANSFER LEARNING")
    
    # Load MobileNetV2 pretrained
    base_model = keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model
    base_model.trainable = False
    
    # Tạo model
    inputs = keras.Input(shape=(*IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    
    print(f"✅ Đã tạo MobileNetV2 model")
    print(f"   Total params: {model.count_params():,}")
    print(f"   Trainable params: {sum([np.prod(v.shape) for v in model.trainable_weights]):,}")
    
    return model, base_model

def train_baseline(train_ds, val_ds):
    """Train baseline model"""
    
    print_section("HUẤN LUYỆN BASELINE CNN")
    
    # Tạo model
    model = build_baseline_model()
    
    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(LEARNING_RATE_BASELINE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=REDUCE_LR_FACTOR,
            patience=REDUCE_LR_PATIENCE,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            os.path.join(MODELS_DIR, 'best_baseline.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_BASELINE,
        callbacks=callbacks,
        verbose=1
    )
    
    # Lưu model cuối
    final_path = os.path.join(MODELS_DIR, 'baseline_final.keras')
    model.save(final_path)
    print(f"\n✅ Đã lưu model: {final_path}")
    
    # Lưu history
    history_path = os.path.join(REPORTS_DIR, 'baseline_history.png')
    save_training_history(history, history_path)
    
    # Lưu model info
    info_path = os.path.join(REPORTS_DIR, 'baseline_info.json')
    save_model_info(model, info_path)
    
    return model, history

def train_mobilenetv2(train_ds, val_ds):
    """Train MobileNetV2 với 2 phases"""
    
    # Phase 1: Train classifier head
    print_section("PHASE 1: TRAIN CLASSIFIER HEAD (FREEZE BACKBONE)")
    
    model, base_model = build_mobilenetv2_model()
    
    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(LEARNING_RATE_TRANSFER_PHASE1),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks phase 1
    callbacks_phase1 = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            os.path.join(MODELS_DIR, 'mobilenetv2_phase1.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train phase 1
    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_TRANSFER_PHASE1,
        callbacks=callbacks_phase1,
        verbose=1
    )
    
    # Phase 2: Fine-tune
    print_section("PHASE 2: FINE-TUNE TOP LAYERS")
    
    # Unfreeze top layers
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    
    print(f"✅ Unfroze top 30 layers")
    print(f"   Trainable params: {sum([np.prod(v.shape) for v in model.trainable_weights]):,}")
    
    # Compile với learning rate thấp hơn
    model.compile(
        optimizer=keras.optimizers.Adam(LEARNING_RATE_TRANSFER_PHASE2),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks phase 2
    callbacks_phase2 = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=REDUCE_LR_FACTOR,
            patience=REDUCE_LR_PATIENCE,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            os.path.join(MODELS_DIR, 'mobilenetv2_phase2.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train phase 2
    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_TRANSFER_PHASE2,
        callbacks=callbacks_phase2,
        verbose=1
    )
    
    # Lưu model cuối
    final_path = os.path.join(MODELS_DIR, 'mobilenetv2_final.keras')
    model.save(final_path)
    print(f"\n✅ Đã lưu model: {final_path}")
    
    # Lưu history
    history_path = os.path.join(REPORTS_DIR, 'mobilenetv2_history.png')
    save_training_history(history2, history_path)
    
    # Lưu model info
    info_path = os.path.join(REPORTS_DIR, 'mobilenetv2_info.json')
    save_model_info(model, info_path)
    
    return model, history1, history2

def main():
    parser = argparse.ArgumentParser(description='Huấn luyện model phân loại rác thải')
    parser.add_argument('--model', type=str, required=True, choices=['baseline', 'mobilenetv2'],
                       help='Loại model: baseline hoặc mobilenetv2')
    
    args = parser.parse_args()
    
    print_section(f"🚀 HUẤN LUYỆN MODEL: {args.model.upper()}")
    
    # Set seeds
    set_seeds(SEED)
    print(f"✅ Đã set seed: {SEED}")
    
    # Tạo thư mục outputs
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    # Tạo data generators
    train_ds, val_ds = create_data_generators()
    
    # Train
    if args.model == 'baseline':
        model, history = train_baseline(train_ds, val_ds)
    else:
        model, history1, history2 = train_mobilenetv2(train_ds, val_ds)
    
    print_section("✅ HOÀN TẤT HUẤN LUYỆN!")
    print(f"\n💡 Để dự đoán ảnh:")
    print(f"   python src/predict.py test.jpg")
    print(f"\n💡 Để đánh giá model:")
    print(f"   python src/evaluate.py")

if __name__ == "__main__":
    main()

