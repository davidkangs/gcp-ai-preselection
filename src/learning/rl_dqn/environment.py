"""
지적측량기준점 선점 강화학습 환경
한 지구를 하나의 에피소드로 관리
"""

import numpy as np
import gym
from gym import spaces
from shapely.geometry import Point, LineString
import logging
from typing import List, Tuple, Dict, Optional
import sys
from pathlib import Path

# 프로젝트 경로 추가
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.core.visibility_checker import VisibilityChecker
from src.core.coverage_analyzer import CoverageAnalyzer

logger = logging.getLogger(__name__)


class SurveyPointEnvironment(gym.Env):
    """지적측량기준점 선점 환경"""
    
    def __init__(self, district_polygon, road_polygons, skeleton_points,
                 heuristic_intersections=None, config=None):
        """
        Args:
            district_polygon: 지구 경계 폴리곤
            road_polygons: 도로 폴리곤들
            skeleton_points: 스켈레톤 포인트들 [(x,y), ...]
            heuristic_intersections: 휴리스틱 교차점들
            config: 환경 설정
        """
        super().__init__()
        
        self.district_polygon = district_polygon
        self.road_polygons = road_polygons
        self.skeleton_points = skeleton_points
        self.heuristic_intersections = heuristic_intersections or []
        
        # 환경 설정
        self.config = config or {}
        self.max_points = self.config.get('max_points', 100)
        self.target_distance = self.config.get('target_distance', 50.0)
        self.coverage_radius = self.config.get('coverage_radius', 50.0)
        self.max_steps = self.config.get('max_steps', 200)
        self.coverage_target = self.config.get('coverage_target', 0.95)
        
        # 모듈 초기화
        self.visibility_checker = VisibilityChecker(road_polygons)
        self.coverage_analyzer = CoverageAnalyzer(district_polygon, self.coverage_radius)
        
        # 후보점 생성
        self.candidate_points = self._generate_candidates()
        self.num_candidates = len(self.candidate_points)
        
        # 액션/관찰 공간 정의
        self.action_space = spaces.Discrete(self.num_candidates)
        
        # 상태 차원: 각 후보점의 특징 벡터
        self.feature_dim = 20  # 후보점당 특징 개수
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.num_candidates * self.feature_dim,),
            dtype=np.float32
        )
        
        # 에피소드 상태
        self.placed_points = []
        self.current_step = 0
        self.done = False
        self.episode_reward = 0
        
        logger.info(f"환경 초기화: {self.num_candidates}개 후보점, 최대 {self.max_points}개 배치 가능")
    
    def reset(self):
        """환경 리셋 - 휴리스틱 교차점만 배치"""
        # 휴리스틱 교차점으로 초기화
        self.placed_points = self.heuristic_intersections.copy()
        self.current_step = 0
        self.done = False
        self.episode_reward = 0
        
        # 후보점 사용 가능 여부 초기화
        self.available_candidates = [True] * self.num_candidates
        
        logger.info(f"환경 리셋: {len(self.placed_points)}개 초기 교차점")
        
        return self._get_observation()
    
    def step(self, action: int):
        """한 스텝 진행 - 새로운 점 추가"""
        if self.done:
            logger.warning("에피소드가 이미 종료됨")
            return self._get_observation(), 0, True, {}
        
        # 액션 유효성 검사
        if action < 0 or action >= self.num_candidates:
            logger.warning(f"잘못된 액션: {action}")
            return self._get_observation(), -10, True, {}
        
        # 이미 사용된 후보인지 확인
        if not self.available_candidates[action]:
            # 이미 사용된 후보 선택 시 페널티
            reward = -5
            info = {'reason': 'already_used'}
        else:
            # 새 점 추가
            new_point = self.candidate_points[action]
            self.placed_points.append(new_point)
            self.available_candidates[action] = False
            
            # 보상 계산
            reward = self._calculate_reward(new_point)
            info = self._get_step_info(new_point)
        
        self.current_step += 1
        self.episode_reward += reward
        
        # 종료 조건 확인
        self.done = self._check_done()
        
        # 에피소드 종료 시 추가 보상
        if self.done:
            final_bonus = self._calculate_final_bonus()
            reward += final_bonus
            self.episode_reward += final_bonus
            info['final_bonus'] = final_bonus
        
        return self._get_observation(), reward, self.done, info
    
    def _generate_candidates(self) -> List[Tuple[float, float]]:
        """후보점 생성 - 스켈레톤을 따라 일정 간격으로"""
        candidates = []
        
        # 스켈레톤 포인트를 10m 간격으로 샘플링
        sample_interval = 10.0
        
        for i in range(len(self.skeleton_points) - 1):
            p1 = np.array(self.skeleton_points[i])
            p2 = np.array(self.skeleton_points[i + 1])
            
            # 두 점 사이 거리
            dist = np.linalg.norm(p2 - p1)
            
            if dist < sample_interval:
                # 중간점만 추가
                mid = (p1 + p2) / 2
                candidates.append(tuple(mid))
            else:
                # 일정 간격으로 보간
                num_samples = int(dist / sample_interval)
                for j in range(num_samples + 1):
                    t = j / max(num_samples, 1)
                    point = p1 + t * (p2 - p1)
                    candidates.append(tuple(point))
        
        # 중복 제거
        candidates = list(set(candidates))
        
        # 최대 개수 제한
        if len(candidates) > 1000:
            # 균등하게 샘플링
            indices = np.linspace(0, len(candidates)-1, 1000, dtype=int)
            candidates = [candidates[i] for i in indices]
        
        return candidates
    
    def _get_observation(self) -> np.ndarray:
        """현재 상태를 관찰 벡터로 변환"""
        features = []
        
        for i, candidate in enumerate(self.candidate_points):
            if self.available_candidates[i]:
                # 후보점의 특징 추출
                feat = self._extract_candidate_features(candidate)
            else:
                # 이미 사용된 후보는 0 벡터
                feat = np.zeros(self.feature_dim)
            
            features.extend(feat)
        
        return np.array(features, dtype=np.float32)
    
    def _extract_candidate_features(self, candidate: Tuple[float, float]) -> List[float]:
        """후보점의 특징 벡터 추출"""
        features = []
        
        # 1-2. 후보점 좌표 (정규화)
        x, y = candidate
        norm_x = (x - self.district_polygon.bounds[0]) / (self.district_polygon.bounds[2] - self.district_polygon.bounds[0])
        norm_y = (y - self.district_polygon.bounds[1]) / (self.district_polygon.bounds[3] - self.district_polygon.bounds[1])
        features.extend([norm_x, norm_y])
        
        # 3. 가장 가까운 기존 점까지의 거리
        if self.placed_points:
            min_dist = min(np.linalg.norm(np.array(candidate) - np.array(p)) 
                          for p in self.placed_points)
            features.append(min_dist / 100.0)  # 100m로 정규화
        else:
            features.append(1.0)
        
        # 4. 시통 가능한 기존 점의 수
        visible_count = self.visibility_checker.count_visible_points(candidate, self.placed_points)
        features.append(visible_count / max(len(self.placed_points), 1))
        
        # 5. 예상 커버리지 증가량
        if self.placed_points:
            current_coverage = self.coverage_analyzer.calculate_coverage(self.placed_points)
            test_coverage = self.coverage_analyzer.calculate_coverage(self.placed_points + [candidate])
            coverage_increase = test_coverage['coverage_ratio'] - current_coverage['coverage_ratio']
            features.append(coverage_increase * 100)  # 스케일 조정
        else:
            features.append(1.0)
        
        # 6-7. 50m 범위 내 점 개수
        nearby_count = sum(1 for p in self.placed_points 
                          if np.linalg.norm(np.array(candidate) - np.array(p)) <= 50)
        features.append(nearby_count / 5.0)  # 5개로 정규화
        
        in_range_count = sum(1 for p in self.placed_points 
                           if 40 <= np.linalg.norm(np.array(candidate) - np.array(p)) <= 60)
        features.append(in_range_count / 5.0)
        
        # 8. 로컬 밀도
        density_radius = 100.0
        local_count = sum(1 for p in self.placed_points 
                         if np.linalg.norm(np.array(candidate) - np.array(p)) <= density_radius)
        features.append(local_count / 10.0)
        
        # 9-10. 패딩
        while len(features) < self.feature_dim:
            features.append(0.0)
        
        return features[:self.feature_dim]
    
    def _calculate_reward(self, new_point: Tuple[float, float]) -> float:
        """보상 계산"""
        reward = 0.0
        
        # 1. 시통성 보상
        visible_count = self.visibility_checker.count_visible_points(new_point, self.placed_points[:-1])
        reward += visible_count * 10
        
        # 2. 거리 보상/페널티
        if len(self.placed_points) > 1:
            distances = [np.linalg.norm(np.array(new_point) - np.array(p)) 
                        for p in self.placed_points[:-1]]
            min_dist = min(distances)
            
            if 45 <= min_dist <= 55:  # 이상적 거리
                reward += 50
            elif 40 <= min_dist <= 60:  # 허용 범위
                reward += 20
            elif min_dist < 30:  # 너무 가까움
                reward -= 30
            elif min_dist > 70:  # 너무 멀음
                reward -= 20
        
        # 3. 커버리지 증가 보상
        if len(self.placed_points) > 1:
            prev_coverage = self.coverage_analyzer.calculate_coverage(self.placed_points[:-1])
            curr_coverage = self.coverage_analyzer.calculate_coverage(self.placed_points)
            coverage_increase = curr_coverage['coverage_ratio'] - prev_coverage['coverage_ratio']
            reward += coverage_increase * 1000
        
        return reward
    
    def _calculate_final_bonus(self) -> float:
        """에피소드 종료 시 추가 보상"""
        bonus = 0.0
        
        coverage_info = self.coverage_analyzer.calculate_coverage(self.placed_points)
        coverage_ratio = coverage_info['coverage_ratio']
        
        # 커버리지 목표 달성 보너스
        if coverage_ratio >= self.coverage_target:
            bonus += 500
            
            # 효율성 보너스 (적은 점으로 높은 커버리지)
            efficiency = self.coverage_analyzer.calculate_efficiency_score(self.placed_points)
            bonus += efficiency * 200
        
        # 점 개수 페널티
        num_points = len(self.placed_points) - len(self.heuristic_intersections)
        bonus -= num_points * 5
        
        return bonus
    
    def _check_done(self) -> bool:
        """종료 조건 확인"""
        # 최대 스텝 도달
        if self.current_step >= self.max_steps:
            return True
        
        # 최대 점 개수 도달
        if len(self.placed_points) >= self.max_points:
            return True
        
        # 목표 커버리지 달성
        coverage_info = self.coverage_analyzer.calculate_coverage(self.placed_points)
        if coverage_info['coverage_ratio'] >= self.coverage_target:
            return True
        
        # 더 이상 유효한 후보가 없음
        if not any(self.available_candidates):
            return True
        
        return False
    
    def _get_step_info(self, new_point: Tuple[float, float]) -> Dict:
        """스텝 정보 반환"""
        coverage_info = self.coverage_analyzer.calculate_coverage(self.placed_points)
        
        return {
            'placed_points': len(self.placed_points),
            'coverage_ratio': coverage_info['coverage_ratio'],
            'visible_count': self.visibility_checker.count_visible_points(new_point, self.placed_points[:-1]),
            'new_point': new_point,
            'step': self.current_step
        }
    
    def render(self, mode='human'):
        """환경 시각화 (선택사항)"""
        pass
    
    def close(self):
        """환경 정리"""
        pass