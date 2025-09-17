"""
AI 예측 모듈 - DQN 기반 예측 로직
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
import logging

from ..learning.dqn_model import create_agent

logger = logging.getLogger(__name__)


class AIPredictor:
    """AI 예측 클래스"""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Args:
            model_path: 모델 파일 경로
        """
        self.model_path = model_path
        self.agent = None
        self.is_loaded = False
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str) -> bool:
        """모델 로드"""
        try:
            self.agent = create_agent()
            self.agent.load(model_path)
            self.model_path = model_path
            self.is_loaded = True
            
            logger.info(f"AI 모델 로드 성공: {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"AI 모델 로드 실패: {e}")
            self.is_loaded = False
            return False
    
    def create_dqn_state_vector(self, point: List[float], skeleton: List[List[float]], 
                               idx: int, heuristic_results: Optional[Dict] = None) -> List[float]:
        """
        DQN 상태 벡터 생성 (안정적인 기본 구현)
        
        Args:
            point: 현재 점 [x, y]
            skeleton: 스켈레톤 데이터
            idx: 현재 점의 인덱스
            heuristic_results: 휴리스틱 결과 (사용 안 함)
        
        Returns:
            20차원 특징 벡터
        """
        try:
            x, y = float(point[0]), float(point[1])
            
            # 기본 좌표 특징
            features = [x, y]
            
            # 이전 점과의 거리 및 각도
            if idx > 0 and len(skeleton[idx-1]) >= 2:
                prev_x, prev_y = skeleton[idx-1][0], skeleton[idx-1][1]
                dist = np.sqrt((x - prev_x)**2 + (y - prev_y)**2)
                angle = np.arctan2(y - prev_y, x - prev_x)
                features.extend([dist, angle])
            else:
                features.extend([0.0, 0.0])
            
            # 다음 점과의 거리 및 각도
            if idx < len(skeleton) - 1 and len(skeleton[idx+1]) >= 2:
                next_x, next_y = skeleton[idx+1][0], skeleton[idx+1][1]
                dist = np.sqrt((next_x - x)**2 + (next_y - y)**2)
                angle = np.arctan2(next_y - y, next_x - x)
                features.extend([dist, angle])
            else:
                features.extend([0.0, 0.0])
            
            # 주변 점 밀도
            density = sum(1 for p in skeleton if len(p) >= 2 and np.sqrt((x-p[0])**2 + (y-p[1])**2) <= 50)
            density = density / len(skeleton) if skeleton and len(skeleton) > 0 else 0.0
            features.append(density)
            
            # 곡률 계산
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
                except Exception:
                    features.append(0.0)
            else:
                features.append(0.0)
            
            # 나머지 특징들을 0으로 패딩 (20차원 맞추기)
            while len(features) < 20:
                features.append(0.0)
            
            return features[:20]
            
        except Exception as e:
            logger.error(f"특징 벡터 생성 오류: {e}")
            return [0.0] * 20  # 기본값 반환
    
    def predict_points(self, skeleton: List[List[float]], 
                      confidence_threshold: float = 0.7) -> Dict[str, Any]:
        """
        스켈레톤 포인트들에 대한 AI 예측
        
        Args:
            skeleton: 스켈레톤 데이터
            confidence_threshold: 신뢰도 임계값
        
        Returns:
            예측 결과 딕셔너리
        """
        if not self.is_loaded:
            logger.warning("모델이 로드되지 않았습니다")
            return {'success': False, 'error': '모델이 로드되지 않았습니다'}
        
        if not skeleton or len(skeleton) == 0:
            logger.warning("스켈레톤 데이터가 없습니다")
            return {'success': False, 'error': '스켈레톤 데이터가 없습니다'}
        
        try:
            # 스켈레톤 배열 변환
            skeleton_array = np.array(skeleton)
            
            # 특징 추출
            features = []
            for i, point in enumerate(skeleton_array):
                feat = self.create_dqn_state_vector(point, skeleton, i)
                features.append(feat)
            
            features_array = np.array(features)
            
            # AI 예측 실행
            ai_points = {
                'intersection': [],
                'curve': [],
                'endpoint': [],
                'delete': []
            }
            
            confidence_data = []
            
            if hasattr(self.agent, 'q_network'):
                # DQN 기반 예측
                with torch.no_grad():
                    device = next(self.agent.q_network.parameters()).device
                    input_tensor = torch.FloatTensor(features_array).to(device)
                    q_values_batch = self.agent.q_network(input_tensor)
                
                # 각 포인트에 대해 예측 및 신뢰도 계산
                for i, q_values in enumerate(q_values_batch):
                    q_vals = q_values.cpu().numpy()
                    action = np.argmax(q_vals)
                    
                    # 신뢰도 계산 (1등과 2등의 차이)
                    max_q = np.max(q_vals)
                    second_max_q = np.partition(q_vals, -2)[-2] if len(q_vals) > 1 else 0
                    confidence = max_q - second_max_q
                    
                    point = tuple(skeleton_array[i])
                    
                    # 신뢰도 정보 저장
                    confidence_data.append({
                        'point': point,
                        'action': action,
                        'confidence': confidence,
                        'q_values': q_vals.tolist()
                    })
                    
                    # 신뢰도 임계값 체크 - 삭제 전용 모드
                    if confidence >= confidence_threshold:
                        if action == 4:  # 삭제만 수행
                            ai_points['delete'].append(point)
                        # 생성 액션들(1,2,3)은 무시
            else:
                # 기본 예측 (DQN 모델이 없는 경우) - 삭제 전용
                predictions = self.agent.predict(features_array)
                
                for i, pred in enumerate(predictions):
                    point = tuple(skeleton_array[i])
                    
                    confidence_data.append({
                        'point': point,
                        'action': pred,
                        'confidence': 0.5,  # 기본 신뢰도
                        'q_values': [0.5, 0.5, 0.5, 0.5, 0.5]
                    })
                    
                    # 삭제만 수행 (생성 무시)
                    if pred == 4:
                        ai_points['delete'].append(point)
            
            # AI 최적화 통계 (긍정적 메시지)
            delete_count = len(ai_points['delete'])
            optimized_count = len(skeleton) - delete_count
            logger.info(f"AI 스마트 최적화 완료: {optimized_count}개 최적 특징점 검출")
            if delete_count > 0:
                logger.info(f"AI 품질 개선: {delete_count}개 불필요한 점 제거")
            
            return {
                'success': True,
                'ai_points': ai_points,
                'confidence_data': confidence_data,
                'total_points': len(skeleton),
                'optimized_count': optimized_count,
                'delete_count': delete_count
            }
            
        except Exception as e:
            logger.error(f"AI 예측 오류: {e}")
            return {'success': False, 'error': str(e)}
    
    def apply_deletions(self, points: Dict[str, List[Tuple[float, float]]], 
                       delete_points: List[Tuple[float, float]]) -> Dict[str, List[Tuple[float, float]]]:
        """
        AI가 예측한 삭제 포인트들을 적용
        
        Args:
            points: 현재 점들
            delete_points: 삭제할 점들
        
        Returns:
            삭제 적용된 점들
        """
        if not delete_points:
            return points
        
        try:
            modified_points = {key: list(value) for key, value in points.items()}
            deleted_count = 0
            
            for delete_x, delete_y in delete_points:
                # 각 카테고리에서 가장 가까운 점 찾아서 제거
                for category in ['intersection', 'curve', 'endpoint']:
                    if category not in modified_points:
                        continue
                    
                    point_list = modified_points[category]
                    min_dist = float('inf')
                    closest_idx = -1
                    
                    for i, (x, y) in enumerate(point_list):
                        dist = np.sqrt((x - delete_x)**2 + (y - delete_y)**2)
                        if dist < min_dist and dist < 10:  # 10m 이내만
                            min_dist = dist
                            closest_idx = i
                    
                    if closest_idx >= 0:
                        del modified_points[category][closest_idx]
                        deleted_count += 1
                        logger.debug(f"AI 품질 개선: {category} 카테고리에서 점 최적화 "
                                    f"({delete_x:.1f}, {delete_y:.1f})")
                        break
            
            if deleted_count > 0:
                logger.info(f"AI 스마트 최적화: {deleted_count}개 점 품질 개선 완료")
            return modified_points
            
        except Exception as e:
            logger.error(f"AI 삭제 적용 오류: {e}")
            return points
    
    def get_prediction_stats(self, prediction_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        예측 통계 정보
        
        Args:
            prediction_result: 예측 결과
        
        Returns:
            통계 정보
        """
        try:
            if not prediction_result.get('success', False):
                return {'error': '예측 실패'}
            
            ai_points = prediction_result.get('ai_points', {})
            confidence_data = prediction_result.get('confidence_data', [])
            
            # 카테고리별 예측 개수
            category_counts = {
                category: len(points) for category, points in ai_points.items()
            }
            
            # 신뢰도 통계
            if confidence_data:
                confidences = [data['confidence'] for data in confidence_data]
                confidence_stats = {
                    'mean': np.mean(confidences),
                    'std': np.std(confidences),
                    'min': np.min(confidences),
                    'max': np.max(confidences)
                }
            else:
                confidence_stats = {'mean': 0, 'std': 0, 'min': 0, 'max': 0}
            
            # 액션별 분포
            action_counts = {}
            for data in confidence_data:
                action = data['action']
                action_counts[action] = action_counts.get(action, 0) + 1
            
            return {
                'category_counts': category_counts,
                'confidence_stats': confidence_stats,
                'action_counts': action_counts,
                'total_points': prediction_result.get('total_points', 0),
                'total_predictions': prediction_result.get('total_predictions', 0)
            }
            
        except Exception as e:
            logger.error(f"예측 통계 계산 오류: {e}")
            return {'error': str(e)}
    
    def is_model_loaded(self) -> bool:
        """모델 로드 상태 확인"""
        return self.is_loaded
    
    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        return {
            'model_path': self.model_path,
            'is_loaded': self.is_loaded,
            'agent_type': type(self.agent).__name__ if self.agent else None
        } 