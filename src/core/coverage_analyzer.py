"""
커버리지 분석 모듈
선점된 점들이 전체 영역을 얼마나 커버하는지 계산
"""

import numpy as np
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.ops import unary_union
from scipy.spatial import Voronoi, voronoi_plot_2d
import logging
from typing import List, Tuple, Dict, Union

logger = logging.getLogger(__name__)


class CoverageAnalyzer:
    """커버리지 분석 클래스"""
    
    def __init__(self, boundary_polygon: Union[Polygon, MultiPolygon], 
                 coverage_radius: float = 50.0,
                 grid_size: float = 10.0):
        """
        Args:
            boundary_polygon: 전체 영역 경계
            coverage_radius: 각 점의 커버리지 반경 (미터)
            grid_size: 그리드 크기 (미터)
        """
        self.boundary = boundary_polygon
        self.coverage_radius = coverage_radius
        self.grid_size = grid_size
        
        # 전체 면적 계산
        if isinstance(boundary_polygon, MultiPolygon):
            self.total_area = sum(poly.area for poly in boundary_polygon.geoms)
        else:
            self.total_area = boundary_polygon.area
        
        # 경계 상자
        self.bounds = boundary_polygon.bounds  # (minx, miny, maxx, maxy)
        
        logger.info(f"커버리지 분석기 초기화: 반경 {coverage_radius}m, 전체면적 {self.total_area:.0f}㎡")
    
    def calculate_coverage(self, points: List[Tuple[float, float]]) -> Dict:
        """
        점들의 커버리지 계산
        
        Args:
            points: [(x, y), ...] 리스트
            
        Returns:
            dict: 커버리지 정보
        """
        if len(points) == 0:
            return {
                'coverage_ratio': 0.0,
                'covered_area': 0.0,
                'uncovered_area': self.total_area,
                'num_points': 0,
                'avg_spacing': 0.0,
                'overlap_ratio': 0.0
            }
        
        # 각 점 주변의 버퍼 생성
        buffers = []
        for x, y in points:
            point = Point(x, y)
            buffer = point.buffer(self.coverage_radius)
            buffers.append(buffer)
        
        # 전체 커버 영역 계산 (union)
        covered_area = unary_union(buffers)
        
        # 경계 내부만 고려
        covered_area = covered_area.intersection(self.boundary)
        
        # 커버리지 계산
        covered_area_size = covered_area.area
        coverage_ratio = covered_area_size / self.total_area
        
        # 평균 간격 계산
        avg_spacing = self._calculate_average_spacing(points)
        
        # 중복률 계산
        overlap_ratio = self._calculate_overlap_ratio_internal(points, buffers, covered_area_size)
        
        return {
            'coverage_ratio': coverage_ratio,
            'covered_area': covered_area_size,
            'uncovered_area': self.total_area - covered_area_size,
            'num_points': len(points),
            'avg_spacing': avg_spacing,
            'overlap_ratio': overlap_ratio,
            'coverage_per_point': covered_area_size / len(points) if points else 0
        }
    
    def find_coverage_gaps(self, points: List[Tuple[float, float]], 
                          min_gap_size: float = 100.0) -> List[Tuple[float, float]]:
        """
        커버되지 않은 영역 찾기
        
        Args:
            points: [(x, y), ...] 리스트
            min_gap_size: 최소 갭 크기 (㎡)
            
        Returns:
            list: 커버되지 않은 영역들의 중심점
        """
        if not points:
            # 전체 영역의 중심 반환
            centroid = self.boundary.centroid
            return [(centroid.x, centroid.y)]
        
        # 커버된 영역 계산
        buffers = [Point(x, y).buffer(self.coverage_radius) for x, y in points]
        covered_area = unary_union(buffers).intersection(self.boundary)
        
        # 커버되지 않은 영역
        uncovered_area = self.boundary.difference(covered_area)
        
        gap_centers = []
        
        # MultiPolygon인 경우 각 폴리곤 처리
        if uncovered_area.geom_type == 'MultiPolygon':
            for poly in uncovered_area.geoms:
                if poly.area >= min_gap_size:
                    centroid = poly.centroid
                    gap_centers.append((centroid.x, centroid.y))
        elif uncovered_area.geom_type == 'Polygon':
            if uncovered_area.area >= min_gap_size:
                centroid = uncovered_area.centroid
                gap_centers.append((centroid.x, centroid.y))
        
        return gap_centers
    
    def calculate_overlap_ratio(self, points: List[Tuple[float, float]]) -> float:
        """
        점들 간 커버리지 중복률 계산
        
        Args:
            points: [(x, y), ...] 리스트
            
        Returns:
            float: 중복률 (0.0 ~ 1.0)
        """
        if len(points) <= 1:
            return 0.0
        
        # 각 점의 버퍼 생성
        buffers = [Point(x, y).buffer(self.coverage_radius) for x, y in points]
        
        # 전체 개별 면적 합
        total_individual_area = sum(b.area for b in buffers)
        
        # 실제 커버 면적 (union)
        actual_covered_area = unary_union(buffers).area
        
        # 중복률 = (개별 합 - 실제) / 개별 합
        overlap_ratio = (total_individual_area - actual_covered_area) / total_individual_area
        
        return max(0.0, min(1.0, overlap_ratio))
    
    def _calculate_overlap_ratio_internal(self, points: List[Tuple[float, float]], 
                                        buffers: List[Polygon], 
                                        covered_area_size: float) -> float:
        """내부 중복률 계산"""
        if len(points) <= 1:
            return 0.0
        
        # 전체 개별 면적 합
        total_individual_area = sum(b.area for b in buffers)
        
        # 중복률 = (개별 합 - 실제) / 개별 합
        overlap_ratio = (total_individual_area - covered_area_size) / total_individual_area
        
        return max(0.0, min(1.0, overlap_ratio))
    
    def _calculate_average_spacing(self, points: List[Tuple[float, float]]) -> float:
        """평균 점 간격 계산"""
        if len(points) < 2:
            return 0.0
        
        # 각 점에서 가장 가까운 이웃까지의 거리
        min_distances = []
        
        for i, (x1, y1) in enumerate(points):
            min_dist = float('inf')
            for j, (x2, y2) in enumerate(points):
                if i != j:
                    dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    min_dist = min(min_dist, dist)
            
            if min_dist < float('inf'):
                min_distances.append(min_dist)
        
        return np.mean(min_distances) if min_distances else 0.0
    
    def create_coverage_grid(self, points: List[Tuple[float, float]]) -> np.ndarray:
        """
        커버리지 그리드 생성 (시각화용)
        
        Args:
            points: [(x, y), ...] 리스트
            
        Returns:
            np.ndarray: 2D 커버리지 그리드 (1=커버됨, 0=안커버됨)
        """
        minx, miny, maxx, maxy = self.bounds
        
        # 그리드 크기 계산
        width = int((maxx - minx) / self.grid_size) + 1
        height = int((maxy - miny) / self.grid_size) + 1
        
        # 그리드 초기화
        grid = np.zeros((height, width), dtype=bool)
        
        # 각 그리드 셀이 커버되는지 확인
        for i in range(height):
            for j in range(width):
                cell_x = minx + j * self.grid_size
                cell_y = miny + i * self.grid_size
                cell_point = Point(cell_x, cell_y)
                
                # 경계 내부인지 확인
                if not self.boundary.contains(cell_point):
                    continue
                
                # 어떤 점으로부터 커버되는지 확인
                for px, py in points:
                    dist = np.sqrt((cell_x - px)**2 + (cell_y - py)**2)
                    if dist <= self.coverage_radius:
                        grid[i, j] = True
                        break
        
        return grid
    
    def suggest_next_point(self, existing_points: List[Tuple[float, float]], 
                          candidate_points: List[Tuple[float, float]]) -> Tuple[int, float]:
        """
        다음에 추가할 최적의 점 제안
        
        Args:
            existing_points: 기존 점들
            candidate_points: 후보 점들
            
        Returns:
            tuple: (최적 후보 인덱스, 예상 커버리지 증가량)
        """
        if not candidate_points:
            return -1, 0.0
        
        current_coverage = self.calculate_coverage(existing_points)
        current_ratio = current_coverage['coverage_ratio']
        
        best_idx = -1
        best_improvement = 0.0
        
        for i, candidate in enumerate(candidate_points):
            # 후보점 추가한 경우의 커버리지
            test_points = existing_points + [candidate]
            new_coverage = self.calculate_coverage(test_points)
            new_ratio = new_coverage['coverage_ratio']
            
            improvement = new_ratio - current_ratio
            
            if improvement > best_improvement:
                best_improvement = improvement
                best_idx = i
        
        return best_idx, best_improvement
    
    def calculate_efficiency_score(self, points: List[Tuple[float, float]]) -> float:
        """
        점 배치의 효율성 점수 계산
        
        Args:
            points: [(x, y), ...] 리스트
            
        Returns:
            float: 효율성 점수 (0.0 ~ 1.0)
        """
        if not points:
            return 0.0
        
        coverage_info = self.calculate_coverage(points)
        
        # 효율성 = 커버리지 / (점 개수 * 정규화 상수)
        # 정규화 상수 = 이상적인 점당 커버리지
        ideal_coverage_per_point = np.pi * self.coverage_radius ** 2
        expected_coverage = len(points) * ideal_coverage_per_point / self.total_area
        
        # 실제 커버리지와 기대 커버리지 비교
        efficiency = coverage_info['coverage_ratio'] / min(expected_coverage, 1.0)
        
        # 중복률 페널티
        overlap_penalty = coverage_info['overlap_ratio'] * 0.5
        efficiency = efficiency * (1 - overlap_penalty)
        
        return max(0.0, min(1.0, efficiency))
