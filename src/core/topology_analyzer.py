"""
도로 토폴로지 분석 모듈
분기 도로 길이 분석, 메인도로 vs 분기도로 구분, 교차점 밀도 계산
"""

import numpy as np
import networkx as nx
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict, deque
import logging
from shapely.geometry import Point, LineString, Polygon
from shapely.ops import unary_union

logger = logging.getLogger(__name__)


class TopologyAnalyzer:
    """도로 토폴로지 분석기"""
    
    def __init__(self, skeleton_data: Dict, transform_info: Optional[Dict] = None):
        """
        Args:
            skeleton_data: 스켈레톤 데이터 {'skeleton': array, 'transform': transform_info}
            transform_info: 좌표 변환 정보
        """
        self.skeleton = skeleton_data.get('skeleton', [])
        self.transform = transform_info or skeleton_data.get('transform')
        self.graph = None
        self.intersections = []
        self.road_segments = []
        self.main_roads = []
        self.branch_roads = []
        
        # 분석 결과 캐시
        self._topology_cache = {}
        self._distance_cache = {}
        
        logger.info(f"토폴로지 분석기 초기화: 스켈레톤 포인트 {len(self.skeleton)}개")
        
    def build_road_graph(self) -> nx.Graph:
        """스켈레톤에서 도로 그래프 구축"""
        if self.graph is not None:
            return self.graph
            
        self.graph = nx.Graph()
        
        try:
            if isinstance(self.skeleton, np.ndarray):
                # 바이너리 마스크인 경우
                if self.skeleton.dtype == bool or (self.skeleton.dtype in [np.uint8, np.int32] and self.skeleton.max() <= 1):
                    skeleton_points = np.argwhere(self.skeleton)
                    # 좌표 변환 적용
                    if self.transform and 'transform' in self.transform:
                        transform_matrix = self.transform['transform']
                        # 픽셀 좌표를 실제 좌표로 변환
                        real_coords = []
                        for pt in skeleton_points:
                            x = transform_matrix[0] + pt[1] * transform_matrix[1] + pt[0] * transform_matrix[2]
                            y = transform_matrix[3] + pt[1] * transform_matrix[4] + pt[0] * transform_matrix[5]
                            real_coords.append([x, y])
                        skeleton_points = np.array(real_coords)
                else:
                    skeleton_points = self.skeleton
            else:
                # 좌표 리스트인 경우
                skeleton_points = np.array(self.skeleton)
            
            # 점이 너무 많으면 샘플링 (성능 최적화)
            if len(skeleton_points) > 1000:
                step = len(skeleton_points) // 1000
                skeleton_points = skeleton_points[::step]
                logger.info(f"성능 최적화: {len(skeleton_points)}개 점으로 샘플링")
            
            # 노드 추가
            for i, point in enumerate(skeleton_points):
                if len(point) >= 2 and not (np.isnan(point[0]) or np.isnan(point[1])):
                    self.graph.add_node(i, pos=(float(point[0]), float(point[1])))
            
            # 엣지 추가 (인접한 점들을 연결) - 더 효율적인 방법
            node_positions = {i: np.array(data['pos']) for i, data in self.graph.nodes(data=True)}
            
            for i in range(len(node_positions)):
                for j in range(i + 1, min(i + 10, len(node_positions))):  # 인접한 10개 점만 확인
                    if i in node_positions and j in node_positions:
                        dist = np.linalg.norm(node_positions[i] - node_positions[j])
                        if dist < 20:  # 20픽셀 이내의 점들을 연결
                            self.graph.add_edge(i, j, weight=float(dist))
            
            logger.info(f"도로 그래프 구축 완료: {self.graph.number_of_nodes()}개 노드, {self.graph.number_of_edges()}개 엣지")
            
        except Exception as e:
            logger.error(f"도로 그래프 구축 실패: {e}")
            # 빈 그래프라도 반환
            self.graph = nx.Graph()
            
        return self.graph
    
    def find_intersections(self, min_degree: int = 3) -> List[Tuple[float, float]]:
        """교차점 찾기 (degree가 min_degree 이상인 노드)"""
        if not self.graph:
            self.build_road_graph()
        
        intersections = []
        for node in self.graph.nodes():
            if self.graph.degree[node] >= min_degree:
                pos = self.graph.nodes[node]['pos']
                intersections.append(pos)
        
        self.intersections = intersections
        logger.info(f"교차점 {len(intersections)}개 검출")
        return intersections
    
    def analyze_road_segments(self) -> List[Dict]:
        """도로 세그먼트 분석"""
        if not self.graph:
            self.build_road_graph()
        
        segments = []
        visited_edges = set()
        
        for edge in self.graph.edges():
            if edge in visited_edges or (edge[1], edge[0]) in visited_edges:
                continue
            
            # 세그먼트 추적
            segment_path = self._trace_segment(edge[0], edge[1], visited_edges)
            if len(segment_path) >= 2:
                segment_info = self._analyze_segment(segment_path)
                segments.append(segment_info)
        
        self.road_segments = segments
        logger.info(f"도로 세그먼트 {len(segments)}개 분석 완료")
        return segments
    
    def _trace_segment(self, start_node: int, next_node: int, visited_edges: Set) -> List[int]:
        """도로 세그먼트 추적"""
        path = [start_node, next_node]
        visited_edges.add((start_node, next_node))
        visited_edges.add((next_node, start_node))
        
        current = next_node
        while True:
            neighbors = list(self.graph.neighbors(current))
            # 이미 방문한 노드 제외
            unvisited = [n for n in neighbors if (current, n) not in visited_edges]
            
            # degree가 2인 노드만 계속 추적 (교차점이 아닌)
            if len(unvisited) == 1 and self.graph.degree[current] == 2:
                next_node = unvisited[0]
                path.append(next_node)
                visited_edges.add((current, next_node))
                visited_edges.add((next_node, current))
                current = next_node
            else:
                break
        
        return path
    
    def _analyze_segment(self, path: List[int]) -> Dict:
        """세그먼트 분석"""
        positions = [self.graph.nodes[node]['pos'] for node in path]
        
        # 길이 계산
        length = 0
        for i in range(len(positions) - 1):
            dist = np.linalg.norm(np.array(positions[i]) - np.array(positions[i+1]))
            length += dist
        
        # 시작점과 끝점의 degree 확인
        start_degree = self.graph.degree[path[0]]
        end_degree = self.graph.degree[path[-1]]
        
        # 세그먼트 타입 분류
        is_main_road = self._classify_as_main_road(length, start_degree, end_degree)
        
        return {
            'path': path,
            'positions': positions,
            'length': length,
            'start_degree': start_degree,
            'end_degree': end_degree,
            'is_main_road': is_main_road,
            'start_pos': positions[0],
            'end_pos': positions[-1]
        }
    
    def _classify_as_main_road(self, length: float, start_degree: int, end_degree: int) -> bool:
        """메인도로 여부 판단"""
        # 기본 규칙: 
        # 1. 길이가 100픽셀 이상
        # 2. 양쪽 끝이 모두 교차점 (degree >= 3)
        # 3. 또는 한쪽은 교차점이고 다른 쪽은 끝점이면서 길이가 50픽셀 이상
        
        if length >= 100 and start_degree >= 3 and end_degree >= 3:
            return True
        
        if length >= 50 and ((start_degree >= 3 and end_degree == 1) or 
                            (start_degree == 1 and end_degree >= 3)):
            return True
        
        return False
    
    def classify_roads(self) -> Tuple[List[Dict], List[Dict]]:
        """메인도로와 분기도로 분류"""
        if not self.road_segments:
            self.analyze_road_segments()
        
        main_roads = [seg for seg in self.road_segments if seg['is_main_road']]
        branch_roads = [seg for seg in self.road_segments if not seg['is_main_road']]
        
        self.main_roads = main_roads
        self.branch_roads = branch_roads
        
        logger.info(f"메인도로 {len(main_roads)}개, 분기도로 {len(branch_roads)}개 분류")
        return main_roads, branch_roads
    
    def calculate_branch_length_ratio(self, point: Tuple[float, float], radius: float = 100) -> float:
        """특정 점 주변의 분기도로 길이 비율 계산"""
        if not self.main_roads or not self.branch_roads:
            self.classify_roads()
        
        # 반경 내 세그먼트들 찾기
        nearby_main_length = 0
        nearby_branch_length = 0
        
        for segment in self.main_roads:
            if self._is_segment_nearby(segment, point, radius):
                nearby_main_length += segment['length']
        
        for segment in self.branch_roads:
            if self._is_segment_nearby(segment, point, radius):
                nearby_branch_length += segment['length']
        
        total_length = nearby_main_length + nearby_branch_length
        if total_length == 0:
            return 0.5  # 기본값
        
        return nearby_branch_length / total_length
    
    def _is_segment_nearby(self, segment: Dict, point: Tuple[float, float], radius: float) -> bool:
        """세그먼트가 점 근처에 있는지 확인"""
        for pos in segment['positions']:
            dist = np.linalg.norm(np.array(pos) - np.array(point))
            if dist <= radius:
                return True
        return False
    
    def calculate_intersection_density(self, point: Tuple[float, float], radius: float = 100) -> float:
        """특정 점 주변의 교차점 밀도 계산"""
        if not self.intersections:
            self.find_intersections()
        
        nearby_count = 0
        for intersection in self.intersections:
            dist = np.linalg.norm(np.array(intersection) - np.array(point))
            if dist <= radius:
                nearby_count += 1
        
        # 밀도 = 개수 / 면적 (단위: 개/픽셀²)
        area = np.pi * radius * radius
        density = nearby_count / area
        
        return density
    
    def get_deletion_priority_score(self, point: Tuple[float, float]) -> float:
        """점의 삭제 우선순위 점수 계산 (높을수록 삭제 우선순위 높음)"""
        # 1. 분기도로 비율 (높을수록 삭제 우선순위 높음)
        branch_ratio = self.calculate_branch_length_ratio(point)
        
        # 2. 교차점 밀도 (높을수록 삭제 우선순위 높음)
        density = self.calculate_intersection_density(point)
        
        # 3. 가장 가까운 교차점까지의 거리 (가까울수록 삭제 우선순위 높음)
        min_intersection_dist = self._get_min_intersection_distance(point)
        distance_score = 1.0 / (1.0 + min_intersection_dist / 50)  # 정규화
        
        # 4. 도로에서의 위치 (분기 끝부분일수록 삭제 우선순위 높음)
        position_score = self._get_position_score(point)
        
        # 최종 점수 계산 (가중 평균)
        total_score = (
            branch_ratio * 0.3 +
            density * 1000 * 0.3 +  # 밀도는 매우 작은 값이므로 스케일 조정
            distance_score * 0.2 +
            position_score * 0.2
        )
        
        return total_score
    
    def _get_min_intersection_distance(self, point: Tuple[float, float]) -> float:
        """가장 가까운 교차점까지의 거리"""
        if not self.intersections:
            return 1000.0  # 매우 큰 값
        
        min_dist = float('inf')
        for intersection in self.intersections:
            dist = np.linalg.norm(np.array(intersection) - np.array(point))
            min_dist = min(min_dist, dist)
        
        return min_dist if min_dist != float('inf') else 1000.0
    
    def _get_position_score(self, point: Tuple[float, float]) -> float:
        """도로상 위치 점수 (분기 끝부분일수록 높음)"""
        if not self.graph:
            self.build_road_graph()
        
        # 가장 가까운 노드 찾기
        min_dist = float('inf')
        closest_node = None
        for node in self.graph.nodes():
            pos = self.graph.nodes[node]['pos']
            dist = np.linalg.norm(np.array(pos) - np.array(point))
            if dist < min_dist:
                min_dist = dist
                closest_node = node
        
        if closest_node is None:
            return 0.0
        
        # 노드의 degree가 낮을수록 (끝부분일수록) 높은 점수
        degree = self.graph.degree[closest_node]
        if degree == 1:  # 끝점
            return 1.0
        elif degree == 2:  # 일반 도로
            return 0.5
        else:  # 교차점
            return 0.0
    
    def analyze_point_context(self, point: Tuple[float, float]) -> Dict:
        """점의 토폴로지 컨텍스트 종합 분석"""
        context = {
            'branch_ratio': self.calculate_branch_length_ratio(point),
            'intersection_density': self.calculate_intersection_density(point),
            'min_intersection_distance': self._get_min_intersection_distance(point),
            'position_score': self._get_position_score(point),
            'deletion_priority': self.get_deletion_priority_score(point),
            'nearby_main_roads': 0,
            'nearby_branch_roads': 0,
            'is_on_main_road': False,
            'is_on_branch_road': False
        }
        
        # 주변 도로 정보
        for segment in self.main_roads:
            if self._is_segment_nearby(segment, point, 30):
                context['nearby_main_roads'] += 1
                if self._is_point_on_segment(point, segment):
                    context['is_on_main_road'] = True
        
        for segment in self.branch_roads:
            if self._is_segment_nearby(segment, point, 30):
                context['nearby_branch_roads'] += 1
                if self._is_point_on_segment(point, segment):
                    context['is_on_branch_road'] = True
        
        return context
    
    def _is_point_on_segment(self, point: Tuple[float, float], segment: Dict) -> bool:
        """점이 세그먼트 위에 있는지 확인"""
        for pos in segment['positions']:
            if np.linalg.norm(np.array(pos) - np.array(point)) < 10:
                return True
        return False
    
    def get_topology_features(self, point: Tuple[float, float]) -> List[float]:
        """토폴로지 관련 특징 벡터 추출 (8차원)"""
        context = self.analyze_point_context(point)
        
        features = [
            context['branch_ratio'],                    # 1. 분기도로 비율
            context['intersection_density'] * 10000,   # 2. 교차점 밀도 (스케일 조정)
            min(context['min_intersection_distance'] / 100, 1.0),  # 3. 가장 가까운 교차점 거리 (정규화)
            context['position_score'],                  # 4. 위치 점수
            min(context['deletion_priority'], 1.0),    # 5. 삭제 우선순위 (정규화)
            min(context['nearby_main_roads'] / 5.0, 1.0),  # 6. 주변 메인도로 수 (정규화)
            min(context['nearby_branch_roads'] / 5.0, 1.0),  # 7. 주변 분기도로 수 (정규화)
            1.0 if context['is_on_branch_road'] else 0.0   # 8. 분기도로 위치 여부
        ]
        
        return features


class BoundaryDistanceCalculator:
    """지구계 경계까지의 거리 계산기"""
    
    def __init__(self, boundary_polygon: Optional[Polygon] = None):
        """
        Args:
            boundary_polygon: 지구계 경계 폴리곤 (Shapely Polygon)
        """
        self.boundary = boundary_polygon
        self._distance_cache = {}
        
    def set_boundary(self, boundary_polygon: Polygon):
        """경계 폴리곤 설정"""
        self.boundary = boundary_polygon
        self._distance_cache.clear()
        
    def calculate_distance_to_boundary(self, point: Tuple[float, float]) -> float:
        """점에서 경계까지의 최단 거리 계산"""
        if self.boundary is None:
            return 1000.0  # 경계가 없으면 큰 값 반환
        
        # 캐시 확인
        point_key = (round(point[0], 1), round(point[1], 1))
        if point_key in self._distance_cache:
            return self._distance_cache[point_key]
        
        try:
            shapely_point = Point(point)
            
            if self.boundary.contains(shapely_point):
                # 내부점인 경우 경계까지의 거리
                distance = shapely_point.distance(self.boundary.boundary)
            else:
                # 외부점인 경우 경계까지의 거리 (음수로 표현)
                distance = -shapely_point.distance(self.boundary.boundary)
            
            # 캐시 저장
            self._distance_cache[point_key] = distance
            return distance
            
        except Exception as e:
            logger.warning(f"경계 거리 계산 실패: {e}")
            return 1000.0
    
    def is_near_boundary(self, point: Tuple[float, float], threshold: float = 50.0) -> bool:
        """점이 경계 근처에 있는지 확인"""
        distance = self.calculate_distance_to_boundary(point)
        return abs(distance) <= threshold
    
    def get_boundary_score(self, point: Tuple[float, float]) -> float:
        """경계 점수 계산 (경계에 가까울수록 높음, 0~1)"""
        distance = abs(self.calculate_distance_to_boundary(point))
        # 200픽셀 이내에서 선형적으로 감소
        score = max(0.0, 1.0 - distance / 200.0)
        return score 