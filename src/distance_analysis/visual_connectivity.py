"""
시각적 연결성 분석 모듈
기존 VisibilityChecker를 활용하되 point_sample 방식으로 개선
"""

import numpy as np
from shapely.geometry import Point, LineString
from shapely.ops import unary_union
import logging

logger = logging.getLogger(__name__)

class VisualConnectivityChecker:
    """시각적 연결성 체크 (시통성 기반)"""
    
    def __init__(self, skeleton_points, road_polygons, max_distance=300.0, min_distance=15.0):
        """
        Args:
            skeleton_points: 스켈레톤 점들 리스트
            road_polygons: 도로 폴리곤들
            max_distance: 최대 연결 거리 (기본 300m)
            min_distance: 최소 연결 거리 (기본 15m)
        """
        self.skeleton_points = skeleton_points
        self.road_polygons = road_polygons
        self.max_distance = max_distance
        self.min_distance = min_distance
        
        # 도로 union 생성
        self.road_union = unary_union(road_polygons) if road_polygons else None
        
        # 스켈레톤 점들로 공간 인덱스 생성
        self.skeleton_spatial_index = self._build_skeleton_spatial_index()
        
        logger.info(f"VisualConnectivityChecker 초기화: 스켈레톤 점 {len(skeleton_points)}개")
    
    def _build_skeleton_spatial_index(self):
        """스켈레톤 점들의 공간 인덱스 생성"""
        spatial_index = {}
        
        for i, point in enumerate(self.skeleton_points):
            # 100m 그리드로 공간 분할
            grid_x = int(point[0] // 100)
            grid_y = int(point[1] // 100)
            
            key = (grid_x, grid_y)
            if key not in spatial_index:
                spatial_index[key] = []
            spatial_index[key].append((i, point))
        
        return spatial_index
    
    def check_visual_connectivity(self, point1, point2):
        """
        두 점 사이의 시각적 연결성 체크
        
        Args:
            point1: (x, y) 좌표
            point2: (x, y) 좌표
            
        Returns:
            bool: 시각적으로 연결되는지 여부
        """
        # 거리 체크
        distance = np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
        if not (self.min_distance <= distance <= self.max_distance):
            return False
        
        # 1. 도로 내부 연결성 체크
        if not self._check_road_containment(point1, point2):
            return False
        
        # 2. 방해점 체크
        if self._has_blocking_points(point1, point2):
            return False
        
        # 3. 스켈레톤 따라가기 체크
        if not self._follows_skeleton_path(point1, point2):
            return False
        
        return True
    
    def _check_road_containment(self, point1, point2):
        """시선 라인이 도로 내부에 포함되는지 확인 (완화된 버전)"""
        if not self.road_union:
            return True
        
        line = LineString([point1, point2])
        
        # 완화된 조건: 라인이 도로와 충분히 겹치거나, 버퍼된 도로 내부에 있으면 OK
        buffered_road = self.road_union.buffer(20.0)  # 20m 버퍼
        
        # 조건 1: 라인이 버퍼된 도로 내부에 있는지
        if buffered_road.contains(line):
            return True
        
        # 조건 2: 라인과 도로의 교집합이 라인 길이의 60% 이상인지
        intersection = line.intersection(buffered_road)
        if hasattr(intersection, 'length'):
            overlap_ratio = intersection.length / line.length
            return overlap_ratio >= 0.6
        
        return False
    
    def _has_blocking_points(self, point1, point2):
        """두 점 사이에 방해하는 점이 있는지 확인"""
        if not self.skeleton_points:
            return False
        
        # 시선 라인 생성
        line = LineString([point1, point2])
        
        # 라인 주변 8m 버퍼 생성
        buffer_zone = line.buffer(8.0)
        
        # 관련 그리드 셀들 찾기
        relevant_cells = self._get_relevant_grid_cells(point1, point2)
        
        # 방해점 개수 카운트
        blocking_count = 0
        
        for cell in relevant_cells:
            if cell in self.skeleton_spatial_index:
                for _, skeleton_point in self.skeleton_spatial_index[cell]:
                    # 시작점, 끝점과 다른 점인지 확인
                    if (self._points_equal(skeleton_point, point1) or 
                        self._points_equal(skeleton_point, point2)):
                        continue
                    
                    # 버퍼 내부에 있는지 확인
                    if buffer_zone.contains(Point(skeleton_point)):
                        blocking_count += 1
                        
                        # 방해점이 너무 많으면 연결 불가
                        if blocking_count > 3:
                            return True
        
        return False
    
    def _follows_skeleton_path(self, point1, point2):
        """
        직선이 스켈레톤을 따라 자연스럽게 이어지는지 확인
        point_sample 방식의 스켈레톤 따라가기 체크
        """
        if not self.skeleton_points:
            return True
        
        # 시선 라인 생성
        line = LineString([point1, point2])
        
        # 라인을 10개 구간으로 샘플링
        sample_points = []
        for i in range(11):
            ratio = i / 10
            sample_point = line.interpolate(ratio, normalized=True)
            sample_points.append((sample_point.x, sample_point.y))
        
        # 각 샘플 점에서 가장 가까운 스켈레톤 점과의 거리 확인
        skeleton_coverage = 0
        
        for sample_point in sample_points:
            min_distance = float('inf')
            
            # 관련 그리드 셀에서 가장 가까운 스켈레톤 점 찾기
            grid_x = int(sample_point[0] // 100)
            grid_y = int(sample_point[1] // 100)
            
            # 주변 9개 셀 확인
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    cell = (grid_x + dx, grid_y + dy)
                    if cell in self.skeleton_spatial_index:
                        for _, skeleton_point in self.skeleton_spatial_index[cell]:
                            dist = np.sqrt((sample_point[0] - skeleton_point[0])**2 + 
                                         (sample_point[1] - skeleton_point[1])**2)
                            min_distance = min(min_distance, dist)
            
            # 10m 이내에 스켈레톤 점이 있으면 커버됨
            if min_distance <= 10.0:
                skeleton_coverage += 1
        
        # 70% 이상 커버되면 스켈레톤을 따라간다고 판단
        coverage_ratio = skeleton_coverage / len(sample_points)
        return coverage_ratio >= 0.7
    
    def _get_relevant_grid_cells(self, point1, point2):
        """두 점 사이의 라인이 지나는 그리드 셀들 반환"""
        cells = set()
        
        # 시작점과 끝점의 그리드 좌표
        x1, y1 = int(point1[0] // 100), int(point1[1] // 100)
        x2, y2 = int(point2[0] // 100), int(point2[1] // 100)
        
        # 라인이 지나는 모든 그리드 셀 계산 (Bresenham's line algorithm 응용)
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        
        x, y = x1, y1
        
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        
        err = dx - dy
        
        while True:
            cells.add((x, y))
            
            if x == x2 and y == y2:
                break
                
            e2 = 2 * err
            
            if e2 > -dy:
                err -= dy
                x += sx
            
            if e2 < dx:
                err += dx
                y += sy
        
        return cells
    
    def _points_equal(self, p1, p2, tolerance=1.0):
        """두 점이 같은지 확인 (허용 오차 내에서)"""
        return (abs(p1[0] - p2[0]) < tolerance and 
                abs(p1[1] - p2[1]) < tolerance)
    
    def get_visual_connections(self, points_with_metadata):
        """
        시각적으로 연결된 점 쌍들 반환
        
        Args:
            points_with_metadata: [(x, y, category, index), ...] 형태의 점 데이터
            
        Returns:
            List[Tuple]: [(idx1, idx2, distance), ...]
        """
        visual_connections = []
        
        for i in range(len(points_with_metadata)):
            for j in range(i + 1, len(points_with_metadata)):
                p1 = points_with_metadata[i]
                p2 = points_with_metadata[j]
                
                if self.check_visual_connectivity((p1[0], p1[1]), (p2[0], p2[1])):
                    distance = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                    visual_connections.append((i, j, distance))
        
        logger.info(f"시각적 연결성 분석 완료: {len(visual_connections)}개 연결")
        return visual_connections
    
    def get_visibility_score(self, point1, point2):
        """
        두 점 사이의 시각적 연결성 점수 (0.0 ~ 1.0)
        
        Returns:
            float: 연결성 점수
        """
        # 거리 점수 (가까울수록 높음)
        distance = np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
        distance_score = max(0, 1 - (distance - self.min_distance) / (self.max_distance - self.min_distance))
        
        # 도로 연결성 점수
        road_score = 1.0 if self._check_road_containment(point1, point2) else 0.0
        
        # 방해점 점수 (방해점이 적을수록 높음)
        blocking_score = 0.0 if self._has_blocking_points(point1, point2) else 1.0
        
        # 스켈레톤 따라가기 점수
        skeleton_score = 1.0 if self._follows_skeleton_path(point1, point2) else 0.0
        
        # 가중 평균 계산
        total_score = (distance_score * 0.2 + 
                      road_score * 0.3 + 
                      blocking_score * 0.3 + 
                      skeleton_score * 0.2)
        
        return total_score 