"""
네트워크 연결성 분석 모듈
point_sample/analyze_network_connectivity.py와 hybrid_graph_filter_test.py 기반
"""

import numpy as np
import networkx as nx
from shapely.geometry import Point, LineString
from shapely.ops import unary_union
import logging

logger = logging.getLogger(__name__)

class NetworkConnectivityAnalyzer:
    """NetworkX 기반 도로 네트워크 연결성 분석"""
    
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
        
        # 네트워크 그래프 초기화
        self.graph = nx.Graph()
        
        logger.info(f"NetworkConnectivityAnalyzer 초기화: 스켈레톤 점 {len(skeleton_points)}개")
    
    def build_road_graph(self, points_with_metadata):
        """
        도로 네트워크 그래프 생성 (point_sample 방식)
        
        Args:
            points_with_metadata: [(x, y, category, index), ...] 형태의 점 데이터
            
        Returns:
            nx.Graph: 네트워크 그래프
        """
        self.graph.clear()
        
        # 점들을 그래프에 추가
        for i, (x, y, category, idx) in enumerate(points_with_metadata):
            self.graph.add_node(i, 
                              x=x, y=y, 
                              category=category, 
                              index=idx,
                              pos=(x, y))
        
        # 연결성 분석 및 엣지 추가
        valid_connections = []
        
        for i in range(len(points_with_metadata)):
            for j in range(i + 1, len(points_with_metadata)):
                p1 = points_with_metadata[i]
                p2 = points_with_metadata[j]
                
                # 거리 계산
                distance = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                
                # 거리 필터링
                if not (self.min_distance <= distance <= self.max_distance):
                    continue
                
                # 연결성 체크
                if self._check_road_network_connectivity(p1, p2):
                    self.graph.add_edge(i, j, weight=distance, distance=distance)
                    valid_connections.append((i, j, distance))
        
        logger.info(f"네트워크 그래프 생성 완료: {len(valid_connections)}개 연결")
        return self.graph
    
    def _check_road_network_connectivity(self, point1, point2):
        """
        두 점이 도로 네트워크를 통해 연결되는지 확인 (point_sample 방식)
        매우 엄격한 연결성 체크
        """
        if not self.road_union:
            return False  # 도로 정보가 없으면 연결 안됨
        
        # 직선 경로 생성
        line = LineString([(point1[0], point1[1]), (point2[0], point2[1])])
        
        # 1. 도로 경계선과의 거리 체크 (point_sample 방식)
        road_boundary = self.road_union.boundary
        
        # 라인의 양끝점이 모두 도로 경계선 50m 이내에 있어야 함 (완화)
        p1_distance = road_boundary.distance(Point(point1))
        p2_distance = road_boundary.distance(Point(point2))
        
        if p1_distance > 50.0 or p2_distance > 50.0:
            return False
        
        # 2. 직선 경로가 도로 내부를 따라가는지 확인 (완화)
        buffered_road = self.road_union.buffer(15.0)  # 15m 버퍼 (완화)
        
        # 라인을 세밀하게 샘플링
        sample_points = self._sample_line_points(line, num_samples=10)  # 샘플링 수 감소
        road_coverage = 0
        
        for sample_point in sample_points:
            if buffered_road.contains(Point(sample_point)):
                road_coverage += 1
        
        coverage_ratio = road_coverage / len(sample_points)
        return coverage_ratio >= 0.4  # 40% 이상 도로 내부 (완화)
    
    def _sample_line_points(self, line, num_samples=10):
        """라인을 균등하게 샘플링하여 점들 반환"""
        points = []
        for i in range(num_samples + 1):
            ratio = i / num_samples
            point = line.interpolate(ratio, normalized=True)
            points.append((point.x, point.y))
        return points
    
    def get_connected_pairs(self):
        """
        연결된 점 쌍들과 거리 반환
        
        Returns:
            List[Tuple]: [(node1_idx, node2_idx, distance), ...]
        """
        connected_pairs = []
        
        for edge in self.graph.edges(data=True):
            node1, node2, data = edge
            distance = data['distance']
            connected_pairs.append((node1, node2, distance))
        
        return connected_pairs
    
    def get_network_statistics(self):
        """네트워크 통계 정보 반환"""
        if not self.graph.edges():
            return {
                'total_connections': 0,
                'avg_distance': 0,
                'min_distance': 0,
                'max_distance': 0,
                'network_density': 0
            }
        
        distances = [data['distance'] for _, _, data in self.graph.edges(data=True)]
        
        return {
            'total_connections': len(distances),
            'avg_distance': np.mean(distances),
            'min_distance': np.min(distances),
            'max_distance': np.max(distances),
            'network_density': nx.density(self.graph)
        }
    
    def find_shortest_path(self, start_idx, end_idx):
        """두 점 사이의 최단 경로 찾기"""
        try:
            path = nx.shortest_path(self.graph, start_idx, end_idx, weight='distance')
            path_length = nx.shortest_path_length(self.graph, start_idx, end_idx, weight='distance')
            return path, path_length
        except nx.NetworkXNoPath:
            return None, float('inf')
    
    def get_node_centrality(self):
        """각 노드의 중심성 계산"""
        if not self.graph.nodes():
            return {}
        
        # 다양한 중심성 지표 계산
        degree_centrality = nx.degree_centrality(self.graph)
        closeness_centrality = nx.closeness_centrality(self.graph, distance='distance')
        betweenness_centrality = nx.betweenness_centrality(self.graph, weight='distance')
        
        return {
            'degree': degree_centrality,
            'closeness': closeness_centrality,
            'betweenness': betweenness_centrality
        } 