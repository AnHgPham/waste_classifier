"""
Các hàm tiện ích cho dự án
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import json
from datetime import datetime

def set_seeds(seed=42):
    """Set seeds cho reproducibility"""
    import random
    import numpy as np
    import tensorflow as tf
    
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def save_training_history(history, save_path):
    """Lưu biểu đồ training history"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Train', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Loss
    axes[1].plot(history.history['loss'], label='Train', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Validation', linewidth=2)
    axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Đã lưu training history: {save_path}")

def save_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Lưu confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           title='Confusion Matrix',
           ylabel='True label',
           xlabel='Predicted label')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Thêm text vào cells
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black",
                   fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Đã lưu confusion matrix: {save_path}")

def save_classification_report(y_true, y_pred, class_names, save_path):
    """Lưu classification report"""
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("CLASSIFICATION REPORT\n")
        f.write("="*70 + "\n\n")
        f.write(report)
        f.write("\n" + "="*70 + "\n")
    
    print(f"✅ Đã lưu classification report: {save_path}")
    return report

def save_model_info(model, save_path):
    """Lưu thông tin model"""
    info = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_params': int(model.count_params()),
        'trainable_params': int(sum([np.prod(v.shape) for v in model.trainable_weights])),
        'non_trainable_params': int(sum([np.prod(v.shape) for v in model.non_trainable_weights])),
        'num_layers': len(model.layers),
        'input_shape': str(model.input_shape),
        'output_shape': str(model.output_shape)
    }
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Đã lưu model info: {save_path}")
    return info

def print_section(title):
    """In section header đẹp"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")

