"""
시통성 체크 모듈
두 점 사이에 도로만 있고 건물이 없으면 시통 가능
"""

import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union
import logging
from typing import List, Tuple, Union

logger = logging.getLogger(__name__)


class VisibilityChecker:
    """시통성 체크 클래스"""
    
    def __init__(self, road_polygons: Union[List[Polygon], Polygon], max_distance: float = 200.0):
        """
        Args:
            road_polygons: 도로 폴리곤(들)
            max_distance: 최대 시통 거리 (미터)
        """
        # 폴리곤 통합
        if isinstance(road_polygons, list):
            if len(road_polygons) > 0:
                self.road_union = unary_union(road_polygons)
            else:
                self.road_union = Polygon()
        else:
            self.road_union = road_polygons
            
        self.max_distance = max_distance
        logger.info(f"시통성 체커 초기화: 최대거리 {max_distance}m")
    
    def check_visibility(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> bool:
        """
        두 점 사이 시통성 체크
        
        Args:
            point1: (x, y) 튜플
            point2: (x, y) 튜플
            
        Returns:
            bool: 시통 가능 여부
        """
        # 거리 체크
        distance = np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
        if distance > self.max_distance:
            return False
        
        # 같은 점이면 시통 가능
        if distance < 0.1:
            return True
        
        # 시선 라인 생성
        line = LineString([point1, point2])
        
        # 라인이 도로 내부에 완전히 포함되는지 확인
        # (도로 밖으로 나가면 건물이 있다고 가정)
        return self.road_union.contains(line) or self.road_union.covers(line)
    
    def count_visible_points(self, new_point: Tuple[float, float], 
                           existing_points: List[Tuple[float, float]]) -> int:
        """
        새로운 점에서 기존 점들 중 몇 개가 보이는지 계산
        
        Args:
            new_point: (x, y) 튜플
            existing_points: [(x, y), ...] 리스트
            
        Returns:
            int: 시통 가능한 점의 수
        """
        visible_count = 0
        
        for point in existing_points:
            if self.check_visibility(new_point, point):
                visible_count += 1
        
        return visible_count
    
    def get_visible_points(self, new_point: Tuple[float, float], 
                          existing_points: List[Tuple[float, float]]) -> List[int]:
        """
        새로운 점에서 보이는 기존 점들의 인덱스 반환
        
        Args:
            new_point: (x, y) 튜플
            existing_points: [(x, y), ...] 리스트
            
        Returns:
            list: 시통 가능한 점들의 인덱스
        """
        visible_indices = []
        
        for i, point in enumerate(existing_points):
            if self.check_visibility(new_point, point):
                visible_indices.append(i)
        
        return visible_indices
    
    def get_visibility_matrix(self, points: List[Tuple[float, float]]) -> np.ndarray:
        """
        모든 점들 간의 시통성 매트릭스 계산
        
        Args:
            points: [(x, y), ...] 리스트
            
        Returns:
            np.ndarray: N x N 시통성 매트릭스
        """
        n = len(points)
        matrix = np.eye(n, dtype=bool)  # 대각선은 True
        
        for i in range(n):
            for j in range(i + 1, n):
                visible = self.check_visibility(points[i], points[j])
                matrix[i, j] = visible
                matrix[j, i] = visible
        
        return matrix
    
    def find_isolated_points(self, points: List[Tuple[float, float]], 
                           min_connections: int = 2) -> List[int]:
        """
        연결이 부족한 고립된 점들 찾기
        
        Args:
            points: [(x, y), ...] 리스트
            min_connections: 최소 연결 개수
            
        Returns:
            list: 고립된 점들의 인덱스
        """
        visibility_matrix = self.get_visibility_matrix(points)
        connection_counts = np.sum(visibility_matrix, axis=1) - 1  # 자기 자신 제외
        
        isolated_indices = []
        for i, count in enumerate(connection_counts):
            if count < min_connections:
                isolated_indices.append(i)
        
        return isolated_indices
    
    def calculate_visibility_score(self, point: Tuple[float, float], 
                                 existing_points: List[Tuple[float, float]]) -> float:
        """
        점의 시통성 점수 계산 (0.0 ~ 1.0)
        
        Args:
            point: (x, y) 튜플
            existing_points: [(x, y), ...] 리스트
            
        Returns:
            float: 시통성 점수
        """
        if not existing_points:
            return 1.0
        
        visible_count = self.count_visible_points(point, existing_points)
        max_possible = len(existing_points)
        
        return visible_count / max_possible
    
    def is_point_on_road(self, point: Tuple[float, float]) -> bool:
        """
        점이 도로 위에 있는지 확인
        
        Args:
            point: (x, y) 튜플
            
        Returns:
            bool: 도로 위 여부
        """
        p = Point(point)
        return self.road_union.contains(p) or self.road_union.covers(p)
    
    def get_nearest_road_point(self, point: Tuple[float, float]) -> Tuple[float, float]:
        """
        가장 가까운 도로 위의 점 찾기
        
        Args:
            point: (x, y) 튜플
            
        Returns:
            tuple: 가장 가까운 도로 위의 점
        """
        p = Point(point)
        
        # 이미 도로 위에 있으면 그대로 반환
        if self.is_point_on_road(point):
            return point
        
        # 도로 경계까지의 최단 거리 점 찾기
        nearest = self.road_union.exterior.interpolate(
            self.road_union.exterior.project(p)
        )
        
        return (nearest.x, nearest.y)


# 유틸리티 함수들
def extract_road_polygons_from_gdf(road_gdf):
    """
    GeoDataFrame에서 도로 폴리곤 추출
    
    Args:
        road_gdf: 도로 GeoDataFrame
        
    Returns:
        list: 폴리곤 리스트
    """
    polygons = []
    
    for _, row in road_gdf.iterrows():
        geom = row.geometry
        
        if geom is None:
            continue
            
        # LineString은 버퍼를 적용해서 폴리곤으로 변환
        if geom.geom_type == 'LineString':
            # 도로 폭을 고려한 버퍼 (예: 5m)
            buffered = geom.buffer(5.0, cap_style=2)  # 2 = CAP_STYLE.flat
            polygons.append(buffered)
        elif geom.geom_type == 'MultiLineString':
            for line in geom.geoms:
                buffered = line.buffer(5.0, cap_style=2)
                polygons.append(buffered)
        elif geom.geom_type == 'Polygon':
            polygons.append(geom)
        elif geom.geom_type == 'MultiPolygon':
            polygons.extend(list(geom.geoms))
    
    return polygons
