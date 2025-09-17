"""
빠른 학습을 위한 설정 오버라이드
이 파일이 있으면 자동으로 최적화된 설정 사용
"""

FAST_TRAINING_CONFIG = {
    'epochs': 20,
    'batch_size': 256,
    'learning_rate': 0.003,
    'hidden_sizes': [128, 64],
    'early_stopping': True,
    'early_stopping_patience': 5,
    'gradient_clip': 1.0,
    'use_amp': True  # Automatic Mixed Precision
}

# GPU 최적화
import torch
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.enabled = True
    
    # 메모리 할당 최적화
    torch.cuda.empty_cache()
    
    print(f"🚀 GPU 최적화 활성화: {torch.cuda.get_device_name(0)}")
    print(f"   메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

print("⚡ 빠른 학습 모드 활성화됨!")
print(f"   설정: {FAST_TRAINING_CONFIG}")
