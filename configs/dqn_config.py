"""
DQN 학습 시스템 설정
"""

# DQN 모델 설정 (메인 설정)
DQN_CONFIG = {
    'feature_dim': 20,
    'action_size': 3,  # 임시: 기존 모델과 호환 (0=keep, 1=add_curve, 2=delete)
    'hyperparameters': {
        'learning_rate': 0.001,
        'batch_size': 64,
        'gamma': 0.99,
        'epsilon_decay': 0.995
    },
    'reward_system': {
        'correct_intersection': 10.0,
        'correct_curve': 8.0,
        'correct_endpoint': 8.0,
        'correct_delete': 5.0,
        'correct_normal': 1.0,
        'miss_important': -10.0,
        'wrong_delete': -15.0,
        'wrong_prediction': -5.0
    }
}

# 데이터 수집 설정
DATA_CONFIG = {
    'data_dir': 'data/training_samples',
    'session_timeout': 3600,  # 1시간
    'auto_save_interval': 300,  # 5분마다 자동 저장
    'feature_vector_size': 20,
}

# 모델 학습 설정
TRAINING_CONFIG = {
    'model_dir': 'src/learning/models',
    'epochs': 100,
    'batch_size': 64,
    'learning_rate': 0.001,
    'hidden_sizes': [128, 64],
    'device': 'auto',  # 'cuda', 'cpu', 'auto'
}

# 예측 설정
PREDICTION_CONFIG = {
    'confidence_threshold': 0.7,
    'max_suggestions': 10,
    'model_filename': 'user_pattern_model.pth',
}

# 로깅 설정
LOGGING_CONFIG = {
    'level': 'INFO',
    'log_dir': 'logs',
    'log_filename': 'dqn_system.log',
}
