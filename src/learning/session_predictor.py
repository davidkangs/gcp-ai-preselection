"""
DQN Session 예측기 - 20차원 통일 버전
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class TrueDQN(nn.Module):
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
        
        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class TrueDQNPredictor:
    def __init__(self, model_path="models/true_dqn_model.pth"):
        self.model_path = Path(model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.is_loaded = False
        self.feature_dim = 20
        self.action_size = 3  # 3-액션 시스템
        
        self.epsilon = 0.01
        self.confidence_threshold = 0.7
        
        logger.info(f"DQN 예측기 초기화 - 디바이스: {self.device}")
        
        if self.model_path.exists():
            self.load_model()
    
    def load_model(self):
        try:
            if not self.model_path.exists():
                logger.warning(f"DQN 모델 파일 없음: {self.model_path}")
                return False
            
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            self.feature_dim = checkpoint.get('feature_dim', 20)
            self.action_size = checkpoint.get('num_actions', checkpoint.get('action_size', 3))  # 3-액션
            
            self.model = TrueDQN(self.feature_dim, output_size=self.action_size).to(self.device)
            self.model.load_state_dict(checkpoint['q_network_state_dict'])
            self.model.eval()
            
            self.epsilon = checkpoint.get('epsilon', 0.01)
            
            self.is_loaded = True
            logger.info(f"✅ DQN 모델 로드 성공 (상태: {self.feature_dim}차원, 액션: {self.action_size}개)")
            return True
            
        except Exception as e:
            logger.error(f"❌ DQN 모델 로드 실패: {e}")
            self.is_loaded = False
            return False
    
    def predict_points(self, skeleton_points, confidence_threshold=0.7):
        if not self.is_loaded:
            logger.warning("DQN 모델이 로드되지 않음")
            return {'curve': [], 'delete': []}  # AI가 담당하는 부분만
        
        # NumPy 배열 체크 수정
        if skeleton_points is None:
            return {'curve': [], 'delete': []}
            
        # NumPy 배열인 경우
        if isinstance(skeleton_points, np.ndarray):
            if skeleton_points.size == 0:
                return {'curve': [], 'delete': []}
        # 리스트인 경우
        elif isinstance(skeleton_points, list):
            if len(skeleton_points) == 0:
                return {'curve': [], 'delete': []}
        else:
            # 다른 타입인 경우 리스트로 변환 시도
            try:
                skeleton_points = list(skeleton_points)
                if len(skeleton_points) == 0:
                    return {'curve': [], 'delete': []}
            except:
                return {'curve': [], 'delete': []}
        
        try:
            state_vectors = []
            valid_indices = []  # 유효한 인덱스 추적
            
            for i, point in enumerate(skeleton_points):
                if len(point) < 2:
                    continue
                state_vector = self._create_state_vector(
                    float(point[0]), float(point[1]), skeleton_points, i
                )
                state_vectors.append(state_vector)
                valid_indices.append(i)
            
            if not state_vectors:
                return {'curve': [], 'delete': []}
            
            X = torch.FloatTensor(state_vectors).to(self.device)
            
            with torch.no_grad():
                q_values = self.model(X)
                predicted_actions = torch.argmax(q_values, dim=1)
                probabilities = torch.softmax(q_values, dim=1)
                max_probs = torch.max(probabilities, dim=1)[0]
            
            results = {'curve': [], 'delete': []}
            
            for idx, (action, prob) in enumerate(zip(predicted_actions, max_probs)):
                if idx >= len(valid_indices):
                    continue
                
                original_idx = valid_indices[idx]
                if original_idx >= len(skeleton_points):
                    continue
                    
                point = skeleton_points[original_idx]
                if len(point) < 2 or prob.item() < confidence_threshold:
                    continue
                
                x, y = float(point[0]), float(point[1])
                action_val = action.item()
                
                # 3-액션 시스템: 0=keep, 1=add_curve, 2=delete
                if action_val == 1:  # add_curve
                    results['curve'].append((x, y))
                elif action_val == 2:  # delete
                    results['delete'].append((x, y))
                # action_val == 0 (keep)은 추가하지 않음
            
            return results
            
        except Exception as e:
            logger.error(f"DQN 예측 중 오류: {e}")
            import traceback
            traceback.print_exc()
            return {'curve': [], 'delete': []}
    
    def _create_state_vector(self, x, y, skeleton, idx, heuristic_results=None):
        features = [x, y]
        
        # 2-3: 이전 포인트와의 거리, 각도
        if idx > 0 and len(skeleton[idx-1]) >= 2:
            prev_x, prev_y = skeleton[idx-1][0], skeleton[idx-1][1]
            dist = np.sqrt((x - prev_x)**2 + (y - prev_y)**2)
            angle = np.arctan2(y - prev_y, x - prev_x)
            features.extend([dist, angle])
        else:
            features.extend([0.0, 0.0])
        
        # 4-5: 다음 포인트와의 거리, 각도
        if idx < len(skeleton) - 1 and len(skeleton[idx+1]) >= 2:
            next_x, next_y = skeleton[idx+1][0], skeleton[idx+1][1]
            dist = np.sqrt((next_x - x)**2 + (next_y - y)**2)
            angle = np.arctan2(next_y - y, next_x - x)
            features.extend([dist, angle])
        else:
            features.extend([0.0, 0.0])
        
        # 6: 로컬 밀도
        density = sum(1 for p in skeleton if len(p) >= 2 and np.sqrt((x-p[0])**2 + (y-p[1])**2) <= 50)
        features.append(density / len(skeleton) if skeleton else 0.0)
        
        # 7: 곡률
        if idx > 0 and idx < len(skeleton) - 1:
            try:
                if len(skeleton[idx-1]) >= 2 and len(skeleton[idx+1]) >= 2:
                    p1 = np.array(skeleton[idx-1][:2])
                    p2 = np.array([x, y])
                    p3 = np.array(skeleton[idx+1][:2])
                    v1 = p2 - p1
                    v2 = p3 - p2
                    angle1 = np.arctan2(v1[1], v1[0])
                    angle2 = np.arctan2(v2[1], v2[0])
                    curvature = abs(angle2 - angle1)
                    if curvature > np.pi:
                        curvature = 2 * np.pi - curvature
                    features.append(curvature)
                else:
                    features.append(0.0)
            except:
                features.append(0.0)
        else:
            features.append(0.0)
        
        # 8-19: 휴리스틱 특징들
        if heuristic_results:
            # 8-11: 휴리스틱 클래스 원핫
            heuristic_class = self._get_heuristic_class(x, y, heuristic_results)
            heuristic_onehot = [0, 0, 0, 0]
            heuristic_onehot[heuristic_class] = 1
            features.extend(heuristic_onehot)
            
            # 12: 휴리스틱 신뢰도
            confidence = 0.9 if heuristic_class > 0 else 0.1
            features.append(confidence)
            
            # 13-15: 주변 카테고리별 포인트 수
            nearby_counts = []
            for cat in ['intersection', 'curve', 'endpoint']:
                count = sum(1 for px, py in heuristic_results.get(cat, []) 
                          if np.sqrt((x - px)**2 + (y - py)**2) <= 50)
                nearby_counts.append(float(count))
            features.extend(nearby_counts)
            
            # 16: 가장 가까운 휴리스틱 포인트까지의 거리
            min_dist = 1000.0
            for cat in ['intersection', 'curve', 'endpoint']:
                for px, py in heuristic_results.get(cat, []):
                    dist = np.sqrt((x - px)**2 + (y - py)**2)
                    min_dist = min(min_dist, dist)
            features.append(min_dist)
            
            # 17: 휴리스틱 포인트 밀도
            total_count = sum(nearby_counts)
            density = total_count / (np.pi * 50 * 50) * 1000 if total_count > 0 else 0
            features.append(density)
        else:
            # 휴리스틱 없을 때 기본값
            features.extend([0] * 12)
        
        # 18-19: 패딩
        while len(features) < 20:
            features.append(0.0)
        
        return features[:20]
    
    def _get_heuristic_class(self, x, y, heuristic_results):
        threshold = 5.0
        for i, cat in enumerate(['intersection', 'curve', 'endpoint']):
            for px, py in heuristic_results.get(cat, []):
                if np.sqrt((x - px)**2 + (y - py)**2) < threshold:
                    return i + 1
        return 0
    
    def get_model_info(self):
        return {
            'is_loaded': self.is_loaded,
            'model_path': str(self.model_path),
            'device': str(self.device),
            'feature_dim': self.feature_dim,
            'action_size': self.action_size,
            'algorithm': 'Deep Q-Network (DQN)',
            'epsilon': self.epsilon,
            'model_type': 'Reinforcement Learning'
        }
    
    def set_epsilon(self, epsilon):
        self.epsilon = max(0.0, min(1.0, epsilon))
        logger.info(f"DQN Epsilon 설정: {self.epsilon:.4f}")


_global_dqn_predictor = None

def get_predictor():
    global _global_dqn_predictor
    if _global_dqn_predictor is None:
        _global_dqn_predictor = TrueDQNPredictor()
    return _global_dqn_predictor

def predict_with_dqn(skeleton_points, confidence_threshold=0.7):
    predictor = get_predictor()
    return predictor.predict_points(skeleton_points, confidence_threshold)

# 하위 호환성을 위한 alias
SessionPredictor = TrueDQNPredictor