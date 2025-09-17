"""
DQN 모델 (완전 수정)
보상 기반 강화학습 Deep Q-Network 구현
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class DQN(nn.Module):
    """Deep Q-Network - 3-액션 통일 시스템"""
    
    def __init__(self, input_size=20, hidden_sizes=[256, 128, 64], output_size=3):
        super().__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(0.2)
            ])
            prev_size = hidden_size
        
        # Q값 출력층
        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)
        
        # 가중치 초기화
        self._initialize_weights()
    
    def _initialize_weights(self):
        """가중치 초기화"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)
    
    def forward(self, x):
        """순전파"""
        return self.network(x)


class ExperienceReplay:
    """경험 재생 버퍼"""
    
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
    
    def push(self, state, action, reward, next_state, done):
        """경험 저장"""
        experience = (
            np.array(state, dtype=np.float32),
            action,
            reward,
            np.array(next_state, dtype=np.float32) if next_state is not None else None,
            done
        )
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        """배치 샘플링"""
        batch = random.sample(self.buffer, batch_size)
        
        states = np.stack([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch], dtype=np.float32)
        next_states = []
        for e in batch:
            if e[3] is not None:
                next_states.append(e[3])
            else:
                next_states.append(np.zeros_like(e[0]))
        next_states = np.stack(next_states)
        dones = np.array([e[4] for e in batch], dtype=bool)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """DQN 에이전트 - 강화학습"""
    
    def __init__(self, state_size=20, action_size=3, lr=0.001,  # 3-액션 시스템
                 batch_size=64, gamma=0.99, epsilon_decay=0.995, device=None):
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.gamma = gamma  # 할인 인수
        self.epsilon_decay = epsilon_decay
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        logger.info(f"DQN 에이전트 초기화 - 디바이스: {self.device}")
        logger.info(f"상태 차원: {state_size}, 액션 수: {action_size}")
        
        # Q-네트워크 (메인 & 타겟)
        self.q_network = DQN(state_size, output_size=action_size).to(self.device)
        self.target_network = DQN(state_size, output_size=action_size).to(self.device)
        self.update_target_network()
        
        # 옵티마이저
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=10, factor=0.5, verbose=True
        )
        
        # 경험 재생
        self.memory = ExperienceReplay(capacity=50000)
        
        # DQN 하이퍼파라미터
        self.epsilon = 1.0  # 탐험 확률
        self.epsilon_min = 0.01
        self.update_target_every = 100  # 타겟 네트워크 업데이트 주기
        self.steps = 0
        
        logger.info(f"DQN 설정 - Gamma: {self.gamma}, Epsilon: {self.epsilon}")
    
    def update_target_network(self):
        """타겟 네트워크 업데이트"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        """경험 저장"""
        self.memory.push(state, action, reward, next_state, done)
    
    def act(self, state, training=True):
        """Epsilon-greedy 행동 선택"""
        if training and np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            action = np.argmax(q_values.cpu().data.numpy())
            
            # 삭제 행동(2)은 높은 신뢰도일 때만
            if action == 2:
                probs = torch.softmax(q_values, dim=1)
                if probs[0][action].item() < 0.8:
                    # 삭제가 아닌 다른 액션 선택 (keep=0 또는 add_curve=1)
                    sorted_q = torch.sort(q_values, descending=True)[1][0]
                    action = sorted_q[1].item() if sorted_q[0].item() == 2 else sorted_q[0].item()
            
            return action
    
    def replay(self):
        """경험 재생 학습"""
        if len(self.memory) < self.batch_size:
            return 0
        
        # 배치 샘플링
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # 텐서 변환
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # 현재 Q값
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # 다음 Q값 (타겟 네트워크 사용)
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # 손실 계산
        loss = F.mse_loss(current_q_values, target_q_values)
        
        # 역전파
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        self.steps += 1
        
        # 타겟 네트워크 업데이트
        if self.steps % self.update_target_every == 0:
            self.update_target_network()
        
        # Epsilon 감소
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()
    
    def train_on_batch(self, states, actions, rewards, next_states, dones, epochs=1):
        """배치 데이터로 직접 학습"""
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        losses = []
        
        for epoch in range(epochs):
            # 현재 Q값
            current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            
            # 타겟 Q값
            with torch.no_grad():
                next_q_values = self.target_network(next_states).max(1)[0]
                target_q_values = rewards + (self.gamma * next_q_values * ~dones)
            
            # 손실 계산
            loss = F.mse_loss(current_q_values, target_q_values)
            
            # 역전파
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
            self.optimizer.step()
            
            losses.append(loss.item())
            
            # 스케줄러 업데이트
            self.scheduler.step(loss)
        
        return np.mean(losses)
    
    def predict(self, states):
        """Q값 예측"""
        self.q_network.eval()
        with torch.no_grad():
            states_tensor = torch.FloatTensor(states).to(self.device)
            q_values = self.q_network(states_tensor)
            predictions = torch.argmax(q_values, dim=1)
        self.q_network.train()
        return predictions.cpu().numpy()
    
    def get_q_values(self, states):
        """Q값 반환"""
        self.q_network.eval()
        with torch.no_grad():
            states_tensor = torch.FloatTensor(states).to(self.device)
            q_values = self.q_network(states_tensor)
        self.q_network.train()
        return q_values.cpu().numpy()
    
    def save(self, path):
        """모델 저장"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
            'state_size': self.state_size,
            'action_size': self.action_size,
            'gamma': self.gamma
        }, path)
        logger.info(f"DQN 모델 저장 완료: {path}")
    
    def load(self, path):
        """모델 로드"""
        if not Path(path).exists():
            raise FileNotFoundError(f"DQN 모델 파일 없음: {path}")
        
        checkpoint = torch.load(path, map_location=self.device)
        
        # 네트워크 재생성 (저장된 차원 사용)
        saved_state_size = checkpoint.get('state_size', self.state_size)
        saved_action_size = checkpoint.get('action_size', self.action_size)
        
        if saved_state_size != self.state_size or saved_action_size != self.action_size:
            logger.warning(f"모델 차원 불일치: 저장({saved_state_size}, {saved_action_size}) vs 현재({self.state_size}, {self.action_size})")
            # 새로운 차원으로 네트워크 재생성
            self.state_size = saved_state_size
            self.action_size = saved_action_size
            self.q_network = DQN(saved_state_size, output_size=saved_action_size).to(self.device)
            self.target_network = DQN(saved_state_size, output_size=saved_action_size).to(self.device)
        
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', 0.01)
        self.steps = checkpoint.get('steps', 0)
        
        logger.info(f"DQN 모델 로드 완료: {path}")
    
    def get_model_info(self):
        """모델 정보 반환"""
        return {
            'algorithm': 'Deep Q-Network (DQN)',
            'state_size': self.state_size,
            'action_size': self.action_size,
            'epsilon': self.epsilon,
            'steps': self.steps,
            'gamma': self.gamma,
            'device': str(self.device),
            'memory_size': len(self.memory)
        }


def create_agent(config=None):
    """DQN 에이전트 생성 - 3-액션 시스템"""
    from . import DQN_CONFIG  # 중앙 설정 import
    
    default_config = {
        'state_size': DQN_CONFIG['feature_dim'],  # 20
        'action_size': DQN_CONFIG['action_size'], # 3
        'lr': DQN_CONFIG['hyperparameters']['learning_rate'],
        'batch_size': DQN_CONFIG['hyperparameters']['batch_size'],
        'gamma': DQN_CONFIG['hyperparameters']['gamma'],
        'epsilon_decay': DQN_CONFIG['hyperparameters']['epsilon_decay']
    }
    
    if config:
        default_config.update(config)
    
    return DQNAgent(
        state_size=default_config['state_size'],
        action_size=default_config['action_size'],
        lr=default_config['lr'],
        batch_size=default_config['batch_size'],
        gamma=default_config['gamma'],
        epsilon_decay=default_config['epsilon_decay']
    )


# 편의 함수들
def predict_with_dqn(skeleton_points, confidence_threshold=0.7):
    """DQN으로 포인트 예측"""
    from .session_predictor import get_predictor
    predictor = get_predictor()
    return predictor.predict_points(skeleton_points, confidence_threshold)

def create_dqn_experience(state, action, reward, next_state, done):
    """DQN 경험 생성"""
    return (state, action, reward, next_state, done)

def calculate_dqn_reward(predicted_action, true_action, state_features=None):
    """DQN 보상 계산"""
    from . import DQN_CONFIG
    
    if predicted_action == true_action:
        # 올바른 예측에 대한 보상
        if true_action == 1:  # 교차점
            return DQN_CONFIG['reward_system']['correct_intersection']
        elif true_action == 2:  # 커브
            return DQN_CONFIG['reward_system']['correct_curve']
        elif true_action == 3:  # 끝점
            return DQN_CONFIG['reward_system']['correct_endpoint']
        elif true_action == 4:  # 삭제
            return DQN_CONFIG['reward_system']['correct_delete']
        else:  # 일반
            return DQN_CONFIG['reward_system']['correct_normal']
    else:
        # 잘못된 예측에 대한 벌점
        if true_action in [1, 3]:  # 중요한 포인트 놓침
            return DQN_CONFIG['reward_system']['miss_important']
        elif predicted_action == 4:  # 잘못된 삭제
            return DQN_CONFIG['reward_system']['wrong_delete']
        else:
            return DQN_CONFIG['reward_system']['wrong_prediction']


# 하위 호환성을 위한 별칭
DQNNetwork = DQN

class RoadEnvironment:
    """도로망 환경 (DQN용) - 더미 클래스"""
    
    def __init__(self, skeleton_data=None):
        self.skeleton_data = skeleton_data
        self.current_step = 0
    
    def reset(self):
        """환경 리셋"""
        self.current_step = 0
        return self._get_state()
    
    def step(self, action):
        """한 스텝 진행"""
        self.current_step += 1
        next_state = self._get_state()
        reward = self._calculate_reward(action)
        done = self.current_step >= len(self.skeleton_data) if self.skeleton_data else True
        
        return next_state, reward, done, {}
    
    def _get_state(self):
        """현재 상태 반환"""
        if self.skeleton_data and self.current_step < len(self.skeleton_data):
            return self.skeleton_data[self.current_step]
        return np.zeros(20)  # 수정: 10 → 20
    
    def _calculate_reward(self, action):
        """보상 계산"""
        return 1.0  # 기본 보상