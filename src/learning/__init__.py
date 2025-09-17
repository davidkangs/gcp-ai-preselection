"""
Deep Q-Network Learning Modules
DQN 강화학습 모듈들
"""

from .dqn_model import DQNAgent, DQN
from .dqn_trainer import DQNTrainer
from .session_predictor import SessionPredictor
from .dqn_data_collector import DQNDataCollector

# DQN 설정 import
try:
    from configs.dqn_config import DQN_CONFIG
except ImportError:
    # 폴백 설정
    DQN_CONFIG = {
        'feature_dim': 20,
        'action_size': 5,
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

__all__ = [
    'DQNAgent',
    'DQN', 
    'DQNTrainer',
    'SessionPredictor',
    'DQNDataCollector',
    'DQN_CONFIG'
]