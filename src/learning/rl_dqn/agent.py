"""
진짜 강화학습 DQN 에이전트
지적측량기준점 선점을 위한 Deep Q-Network
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class DuelingDQN(nn.Module):
    """Dueling DQN 네트워크"""
    
    def __init__(self, input_size, output_size, hidden_sizes=[512, 256, 128]):
        super().__init__()
        
        # 공통 레이어
        self.common_layers = nn.ModuleList()
        prev_size = input_size
        
        for hidden_size in hidden_sizes[:-1]:
            self.common_layers.append(nn.Linear(prev_size, hidden_size))
            self.common_layers.append(nn.ReLU())
            self.common_layers.append(nn.BatchNorm1d(hidden_size))
            self.common_layers.append(nn.Dropout(0.2))
            prev_size = hidden_size
        
        # 가치(Value) 스트림
        self.value_stream = nn.Sequential(
            nn.Linear(prev_size, hidden_sizes[-1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[-1], 1)
        )
        
        # 이점(Advantage) 스트림
        self.advantage_stream = nn.Sequential(
            nn.Linear(prev_size, hidden_sizes[-1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[-1], output_size)
        )
        
        # 가중치 초기화
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)
    
    def forward(self, x):
        # 공통 특징 추출
        for layer in self.common_layers:
            x = layer(x)
        
        # 가치와 이점 계산
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        
        # Q값 = V(s) + A(s,a) - mean(A(s,a))
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        return q_values


class PrioritizedReplayBuffer:
    """우선순위 경험 재생 버퍼"""
    
    def __init__(self, capacity=100000, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.alpha = alpha  # 우선순위 지수
        self.beta = beta    # 중요도 샘플링 지수
        self.beta_increment = beta_increment
        self.epsilon = 1e-6
        
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.max_priority = 1.0
    
    def push(self, state, action, reward, next_state, done):
        """경험 추가"""
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
        self.priorities.append(self.max_priority)
    
    def sample(self, batch_size):
        """우선순위 기반 샘플링"""
        if len(self.buffer) < batch_size:
            return None
        
        # 우선순위를 확률로 변환
        priorities = np.array(self.priorities)
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # 샘플링
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        
        # 중요도 가중치 계산
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        weights = torch.FloatTensor(weights)
        
        # 배치 생성
        batch = [self.buffer[idx] for idx in indices]
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.FloatTensor([e[4] for e in batch])
        
        # 베타 증가
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return states, actions, rewards, next_states, dones, indices, weights
    
    def update_priorities(self, indices, td_errors):
        """TD 오차 기반 우선순위 업데이트"""
        for idx, td_error in zip(indices, td_errors):
            priority = abs(td_error) + self.epsilon
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self):
        return len(self.buffer)


class RLDQNAgent:
    """진짜 강화학습 DQN 에이전트"""
    
    def __init__(self, state_size, action_size, config=None):
        self.state_size = state_size
        self.action_size = action_size
        
        # 하이퍼파라미터
        config = config or {}
        self.lr = config.get('learning_rate', 0.0001)
        self.gamma = config.get('gamma', 0.99)
        self.epsilon = config.get('epsilon_start', 1.0)
        self.epsilon_min = config.get('epsilon_min', 0.01)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        self.batch_size = config.get('batch_size', 32)
        self.target_update = config.get('target_update', 100)
        self.tau = config.get('tau', 0.005)  # Soft update 파라미터
        
        # 디바이스
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"RL DQN 에이전트 초기화 - 디바이스: {self.device}")
        
        # 네트워크
        self.q_network = DuelingDQN(state_size, action_size).to(self.device)
        self.target_network = DuelingDQN(state_size, action_size).to(self.device)
        self.update_target_network()
        
        # 옵티마이저
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=10, factor=0.5, verbose=True
        )
        
        # 경험 재생
        self.memory = PrioritizedReplayBuffer()
        
        # 학습 통계
        self.steps = 0
        self.episodes = 0
        self.losses = deque(maxlen=100)
    
    def act(self, state, training=True):
        """Epsilon-greedy 행동 선택"""
        if training and random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return np.argmax(q_values.cpu().data.numpy())
    
    def remember(self, state, action, reward, next_state, done):
        """경험 저장"""
        self.memory.push(state, action, reward, next_state, done)
    
    def replay(self):
        """경험 재생 학습"""
        batch = self.memory.sample(self.batch_size)
        if batch is None:
            return 0.0
        
        states, actions, rewards, next_states, dones, indices, weights = batch
        
        # 디바이스로 이동
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        weights = weights.to(self.device)
        
        # 현재 Q값
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Double DQN: 다음 액션은 현재 네트워크로, Q값은 타겟 네트워크로
        with torch.no_grad():
            next_actions = self.q_network(next_states).max(1)[1]
            next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        
        # TD 오차
        td_errors = target_q_values - current_q_values
        
        # 가중 손실
        loss = (weights * td_errors.pow(2)).mean()
        
        # 역전파
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # 우선순위 업데이트
        self.memory.update_priorities(indices, td_errors.detach().cpu().numpy())
        
        # Soft update
        self.soft_update_target_network()
        
        # Epsilon 감소
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        self.steps += 1
        self.losses.append(loss.item())
        
        return loss.item()
    
    def update_target_network(self):
        """타겟 네트워크 업데이트 (Hard update)"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def soft_update_target_network(self):
        """타겟 네트워크 소프트 업데이트"""
        for target_param, local_param in zip(self.target_network.parameters(), 
                                           self.q_network.parameters()):
            target_param.data.copy_(self.tau * local_param.data + 
                                  (1.0 - self.tau) * target_param.data)
    
    def save(self, path):
        """모델 저장"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
            'episodes': self.episodes,
            'state_size': self.state_size,
            'action_size': self.action_size,
            'losses': list(self.losses)
        }, path)
        
        logger.info(f"RL DQN 모델 저장: {path}")
    
    def load(self, path):
        """모델 로드"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', 0.01)
        self.steps = checkpoint.get('steps', 0)
        self.episodes = checkpoint.get('episodes', 0)
        
        logger.info(f"RL DQN 모델 로드: {path}")
    
    def get_stats(self):
        """학습 통계 반환"""
        return {
            'epsilon': self.epsilon,
            'steps': self.steps,
            'episodes': self.episodes,
            'avg_loss': np.mean(self.losses) if self.losses else 0.0,
            'memory_size': len(self.memory)
        }