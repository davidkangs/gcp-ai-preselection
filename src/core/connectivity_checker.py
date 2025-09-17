"""
도로 연결성 검사 및 최소한의 자동 점 생성 모듈
도로 폴리곤 기반 분석으로 정말 필요한 경우에만 극소수 점 추가
"""

import numpy as np
import networkx as nx
from shapely.geometry import Point, LineString, Polygon
from shapely.ops import unary_union
from sklearn.cluster import DBSCAN
import logging
from typing import List, Tuple, Dict, Set, Optional

logger = logging.getLogger(__name__)


class ConnectivityChecker:
    """도로 연결성 검사 및 최소한의 자동 점 생성 클래스"""
    
    def __init__(self, min_connection_distance: float = 30.0, max_connection_distance: float = 100.0):
        """
        Args:
            min_connection_distance: 최소 연결 거리 (미터)
            max_connection_distance: 최대 연결 거리 (미터)
        """
        self.min_connection_distance = min_connection_distance
        self.max_connection_distance = max_connection_distance
        self.road_polygons = None
        self.skeleton_points = None
        self.road_union = None
        
    def set_road_data(self, road_polygons, skeleton_points):
        """도로 데이터 설정"""
        self.road_polygons = road_polygons
        self.skeleton_points = np.array(skeleton_points) if not isinstance(skeleton_points, np.ndarray) else skeleton_points
        
        # 도로 폴리곤 통합
        try:
            if isinstance(road_polygons, list) and len(road_polygons) > 0:
                valid_polygons = [p for p in road_polygons if p is not None and not p.is_empty]
                if valid_polygons:
                    self.road_union = unary_union(valid_polygons)
                else:
                    self.road_union = None
            else:
                self.road_union = road_polygons
        except Exception as e:
            logger.warning(f"도로 폴리곤 통합 실패: {e}")
            self.road_union = None
            
        logger.info(f"연결성 체커 초기화: {len(skeleton_points)}개 스켈레톤 점")
    
    def check_point_connectivity(self, points: Dict[str, List[Tuple[float, float]]]) -> Dict:
        """
        점들의 연결성 검사 (극히 보수적 버전)
        
        Args:
            points: {'intersection': [...], 'curve': [...], 'endpoint': [...]}
            
        Returns:
            dict: 연결성 분석 결과
        """
        if self.skeleton_points is None:
            logger.warning("스켈레톤 데이터가 설정되지 않았습니다")
            return {'isolated_points': [], 'suggested_additions': []}
        
        # 모든 점들을 하나의 리스트로 통합
        all_points = []
        point_types = {}
        
        for point_type, point_list in points.items():
            for point in point_list:
                all_points.append(point)
                point_types[len(all_points) - 1] = point_type
        
        if len(all_points) == 0:
            logger.info("검사할 점이 없습니다")
            return {'isolated_points': [], 'suggested_additions': []}
        
        logger.info(f"연결성 검사 시작: {len(all_points)}개 점")
        
        # 극히 보수적 분석: 정말 고립된 점만 찾기
        isolated_points = self._find_truly_isolated_points(all_points)
        
        # 최소한의 제안만 생성 (최대 1개)
        suggested_additions = self._generate_minimal_suggestions(isolated_points, all_points)
        
        logger.info(f"연결성 검사 완료: {len(isolated_points)}개 진짜 고립점, {len(suggested_additions)}개 최소 제안")
        
        return {
            'isolated_points': isolated_points,
            'suggested_additions': suggested_additions
        }
    
    def _find_truly_isolated_points(self, all_points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """정말로 고립된 점들만 찾기"""
        isolated = []
        
        for i, point in enumerate(all_points):
            x, y = point
            has_close_neighbor = False
            
            # 주변 50미터 이내에 다른 점이 있는지 확인
            for j, other_point in enumerate(all_points):
                if i == j:
                    continue
                    
                ox, oy = other_point
                distance = ((x - ox) ** 2 + (y - oy) ** 2) ** 0.5
                
                if distance <= 50:  # 50미터 이내
                    has_close_neighbor = True
                    break
            
            # 고립된 점인 경우, 도로 내부에 있는지 추가 확인
            if not has_close_neighbor:
                if self._is_point_inside_road(point):
                    isolated.append(point)
        
        return isolated
    
    def _is_point_inside_road(self, point: Tuple[float, float]) -> bool:
        """점이 도로 폴리곤 내부에 있는지 확인"""
        if self.road_union is None:
            return True  # 도로 정보가 없으면 내부로 가정
        
        try:
            from shapely.geometry import Polygon, MultiPolygon
            shapely_point = Point(point)
            
            # road_union이 Shapely 기하객체인지 확인
            if isinstance(self.road_union, (Polygon, MultiPolygon)):
                return self.road_union.contains(shapely_point) or self.road_union.touches(shapely_point)
            elif hasattr(self.road_union, 'contains') and hasattr(self.road_union, 'touches'):
                return self.road_union.contains(shapely_point) or self.road_union.touches(shapely_point)
            else:
                return True  # 유효하지 않은 도로 데이터인 경우
        except Exception as e:
            logger.warning(f"도로 내부 확인 실패: {e}")
            return True
    
    def _generate_minimal_suggestions(self, isolated_points: List[Tuple[float, float]], 
                                    all_points: List[Tuple[float, float]]) -> List[Dict]:
        """최소한의 연결 제안 생성 (최대 1개)"""
        if len(isolated_points) == 0:
            return []
        
        # 가장 심각한 고립점 1개만 처리
        most_isolated = isolated_points[0]
        
        # 가장 가까운 스켈레톤 점 찾기
        nearest_skeleton = self._find_nearest_skeleton_point(most_isolated)
        
        if nearest_skeleton is not None and self.skeleton_points is not None:
            if isinstance(self.skeleton_points, np.ndarray) and self.skeleton_points.dtype == bool:
                skeleton_coords = np.argwhere(self.skeleton_points)
                if nearest_skeleton < len(skeleton_coords):
                    skeleton_point = tuple(skeleton_coords[nearest_skeleton].astype(float))
                else:
                    return []
            else:
                if nearest_skeleton < len(self.skeleton_points):
                    skeleton_point = tuple(np.array(self.skeleton_points[nearest_skeleton]).astype(float))
                else:
                    return []
            
            # 기존 점들과 너무 가깝지 않은지 확인
            if self._is_far_enough_from_existing(skeleton_point, all_points, min_distance=25.0):
                return [{
                    'type': 'curve',
                    'position': skeleton_point,
                    'reason': f'극히 필요한 연결점 - 고립점 {most_isolated} 해결용'
                }]
        
        return []
    
    def _find_nearest_skeleton_point(self, point: Tuple[float, float]) -> Optional[int]:
        """가장 가까운 스켈레톤 점의 인덱스 찾기"""
        if self.skeleton_points is None or len(self.skeleton_points) == 0:
            return None
        
        try:
            # 스켈레톤 포인트 형태 확인
            if isinstance(self.skeleton_points, np.ndarray) and self.skeleton_points.dtype == bool:
                # 바이너리 마스크인 경우
                skeleton_coords = np.argwhere(self.skeleton_points)
            else:
                # 좌표 배열인 경우
                skeleton_coords = self.skeleton_points
            
            if len(skeleton_coords) == 0:
                return None
            
            distances = []
            for i, sk_point in enumerate(skeleton_coords):
                if len(sk_point) >= 2:
                    dist = ((point[0] - sk_point[0]) ** 2 + (point[1] - sk_point[1]) ** 2) ** 0.5
                    distances.append((dist, i))
            
            if distances:
                distances.sort()
                min_dist, min_idx = distances[0]
                
                # 너무 멀면 None 반환 (50픽셀 이내만)
                if min_dist <= 50:
                    return min_idx
            
        except Exception as e:
            logger.warning(f"가장 가까운 스켈레톤 점 찾기 실패: {e}")
        
        return None
    
    def _is_far_enough_from_existing(self, new_point: Tuple[float, float], 
                                   existing_points: List[Tuple[float, float]], 
                                   min_distance: float = 25.0) -> bool:
        """새 점이 기존 점들로부터 충분히 떨어져 있는지 확인"""
        for existing in existing_points:
            distance = ((new_point[0] - existing[0]) ** 2 + (new_point[1] - existing[1]) ** 2) ** 0.5
            if distance < min_distance:
                return False
        return True
    
    def auto_add_connection_points(self, points: Dict[str, List[Tuple[float, float]]], 
                                 max_additions: int = 1) -> Dict[str, List[Tuple[float, float]]]:
        """자동 연결점 추가 (최대 1개만)"""
        result = {key: list(value) for key, value in points.items()}
        
        # 연결성 검사
        connectivity_results = self.check_point_connectivity(points)
        
        # 제안된 점들 추가 (최대 1개)
        added_count = 0
        for suggestion in connectivity_results.get('suggested_additions', []):
            if added_count >= max_additions:
                break
                
            if suggestion['type'] in result:
                result[suggestion['type']].append(suggestion['position'])
                added_count += 1
                logger.info(f"최소 연결점 추가: {suggestion['type']} at {suggestion['position']}")
        
        return result 