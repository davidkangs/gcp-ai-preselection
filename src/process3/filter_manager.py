"""
필터 매니저 - 하이브리드 필터 통합 관리
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from shapely.geometry import LineString

from ..filters.hybrid_filter import create_hybrid_filter

logger = logging.getLogger(__name__)


class FilterManager:
    """필터 관리 클래스"""
    
    def __init__(self, 
                 dbscan_eps: float = 10.0,          # 20.0 → 10.0으로 엄격하게
                 network_max_dist: float = 30.0,    # 50.0 → 30.0으로 엄격하게
                 road_buffer: float = 2.0):
        """
        Args:
            dbscan_eps: DBSCAN 클러스터링 거리 (m)
            network_max_dist: 네트워크 연결 최대 거리 (m)
            road_buffer: 도로 경계선 버퍼 (m)
        """
        self.hybrid_filter = create_hybrid_filter(
            dbscan_eps=dbscan_eps,
            network_max_dist=network_max_dist,
            road_buffer=road_buffer
        )
        
        logger.info(f"FilterManager 초기화 (엄격 모드): eps={dbscan_eps}, max_dist={network_max_dist}, buffer={road_buffer}")
    
    def apply_hybrid_filter(self, 
                          points: List[Tuple[float, float]], 
                          skeleton: List[List[float]], 
                          point_roles: Optional[Dict[Tuple[float, float], str]] = None) -> List[Tuple[float, float]]:
        """
        하이브리드 필터 적용 (테스트 코드처럼 깔끔하게)
        
        Args:
            points: 필터링할 점들 [(x, y), ...]
            skeleton: 스켈레톤 데이터 [[x, y], ...]
            point_roles: 점별 역할 {(x, y): 'intersection'|'curve'|'endpoint'}
        
        Returns:
            필터링된 점들
        """
        if not points:
            return points
        
        try:
            # 스켈레톤을 LineString으로 변환
            skeleton_coords = [(float(p[0]), float(p[1])) for p in skeleton if len(p) >= 2]
            skeleton_lines = []
            
            if len(skeleton_coords) >= 2:
                skeleton_line = LineString(skeleton_coords)
                skeleton_lines = [skeleton_line]
            
            # 하이브리드 필터 적용 - 더 엄격한 필터링
            filtered_points = self.hybrid_filter.filter_by_skeleton_connectivity(
                points=points,
                skeleton_lines=skeleton_lines,
                point_roles=point_roles or {},
                dist_thresh=15.0,          # 스켈레톤 연결 거리 증가
                curve_min_length=50.0      # 커브 최소 길이 증가 (더 엄격)
            )
            
            logger.info(f"하이브리드 필터 적용: {len(points)} → {len(filtered_points)}개")
            return filtered_points
            
        except Exception as e:
            logger.error(f"하이브리드 필터 적용 오류: {e}")
            return points  # 오류 시 원본 반환
    
    def remove_duplicate_points(self, 
                              points: Dict[str, List[Tuple[float, float]]], 
                              skeleton: List[List[float]],
                              distance_threshold: float = 5.0) -> Dict[str, List[Tuple[float, float]]]:
        """
        중복점 제거 (카테고리별)
        
        Args:
            points: 카테고리별 점들 {'intersection': [...], 'curve': [...], 'endpoint': [...]}
            skeleton: 스켈레톤 데이터
            distance_threshold: 중복 판단 거리 (m)
        
        Returns:
            중복 제거된 점들
        """
        if not points:
            return points
        
        try:
            # 모든 점을 하나의 리스트로 합치기
            all_points = []
            point_roles = {}
            
            for category, point_list in points.items():
                for point in point_list:
                    all_points.append(point)
                    point_roles[point] = category
            
            if not all_points:
                return points
            
            # 하이브리드 필터로 중복 제거
            filtered_points = self.apply_hybrid_filter(all_points, skeleton, point_roles)
            
            # 카테고리별로 다시 분류
            filtered_by_category = {'intersection': [], 'curve': [], 'endpoint': []}
            
            for point in filtered_points:
                category = point_roles.get(point, 'curve')  # 기본값은 curve
                if category in filtered_by_category:
                    filtered_by_category[category].append(point)
            
            # 로그 출력
            for category in filtered_by_category:
                original_count = len(points.get(category, []))
                filtered_count = len(filtered_by_category[category])
                logger.info(f"{category}: {original_count} → {filtered_count}개")
            
            return filtered_by_category
            
        except Exception as e:
            logger.error(f"중복점 제거 오류: {e}")
            return points  # 오류 시 원본 반환
    
    def filter_by_distance(self, 
                          points: List[Tuple[float, float]], 
                          min_distance: float = 10.0) -> List[Tuple[float, float]]:
        """
        거리 기반 필터링
        
        Args:
            points: 필터링할 점들
            min_distance: 최소 거리 (m)
        
        Returns:
            필터링된 점들
        """
        if not points or len(points) < 2:
            return points
        
        try:
            filtered_points = [points[0]]  # 첫 번째 점은 항상 포함
            
            for current_point in points[1:]:
                # 기존 필터링된 점들과의 거리 확인
                too_close = False
                for existing_point in filtered_points:
                    dist = np.sqrt((current_point[0] - existing_point[0])**2 + 
                                  (current_point[1] - existing_point[1])**2)
                    if dist < min_distance:
                        too_close = True
                        break
                
                if not too_close:
                    filtered_points.append(current_point)
            
            logger.info(f"거리 기반 필터링: {len(points)} → {len(filtered_points)}개")
            return filtered_points
            
        except Exception as e:
            logger.error(f"거리 기반 필터링 오류: {e}")
            return points
    
    def filter_by_importance(self, 
                           points: List[Tuple[float, float]], 
                           skeleton: List[List[float]], 
                           importance_threshold: float = 0.3) -> List[Tuple[float, float]]:
        """
        중요도 기반 필터링
        
        Args:
            points: 필터링할 점들
            skeleton: 스켈레톤 데이터
            importance_threshold: 중요도 임계값 (0~1)
        
        Returns:
            필터링된 점들
        """
        if not points or not skeleton:
            return points
        
        try:
            filtered_points = []
            
            for point in points:
                # 중요도 점수 계산
                importance_score = self.hybrid_filter.calculate_importance_score(
                    point, skeleton, None
                )
                
                if importance_score >= importance_threshold:
                    filtered_points.append(point)
            
            logger.info(f"중요도 기반 필터링: {len(points)} → {len(filtered_points)}개")
            return filtered_points
            
        except Exception as e:
            logger.error(f"중요도 기반 필터링 오류: {e}")
            return points
    
    def apply_custom_filter(self, 
                          points: List[Tuple[float, float]], 
                          filter_func, 
                          **kwargs) -> List[Tuple[float, float]]:
        """
        사용자 정의 필터 적용
        
        Args:
            points: 필터링할 점들
            filter_func: 필터 함수
            **kwargs: 필터 함수 인자들
        
        Returns:
            필터링된 점들
        """
        try:
            filtered_points = filter_func(points, **kwargs)
            logger.info(f"사용자 정의 필터 적용: {len(points)} → {len(filtered_points)}개")
            return filtered_points
            
        except Exception as e:
            logger.error(f"사용자 정의 필터 적용 오류: {e}")
            return points
    
    def get_filter_stats(self, 
                        original_points: List[Tuple[float, float]], 
                        filtered_points: List[Tuple[float, float]]) -> Dict[str, float]:
        """
        필터링 통계 정보
        
        Args:
            original_points: 원본 점들
            filtered_points: 필터링된 점들
        
        Returns:
            통계 정보
        """
        try:
            original_count = len(original_points)
            filtered_count = len(filtered_points)
            removed_count = original_count - filtered_count
            
            if original_count > 0:
                removal_rate = removed_count / original_count
                retention_rate = filtered_count / original_count
            else:
                removal_rate = 0.0
                retention_rate = 0.0
            
            return {
                'original_count': original_count,
                'filtered_count': filtered_count,
                'removed_count': removed_count,
                'removal_rate': removal_rate,
                'retention_rate': retention_rate
            }
            
        except Exception as e:
            logger.error(f"필터링 통계 계산 오류: {e}")
            return {
                'original_count': 0,
                'filtered_count': 0,
                'removed_count': 0,
                'removal_rate': 0.0,
                'retention_rate': 0.0
            } 