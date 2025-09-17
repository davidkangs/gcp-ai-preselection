"""
하이브리드 점 필터링 시스템
DBSCAN(20m) + 네트워크 연결성 + 중요도 + 역할 우선순위
"""

import numpy as np
import networkx as nx
from sklearn.cluster import DBSCAN
from shapely.geometry import Point, LineString
from typing import List, Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class HybridPointFilter:
    """하이브리드 점 필터링 시스템"""
    
    def __init__(self, 
                 dbscan_eps: float = 20.0,
                 network_max_dist: float = 50.0,
                 road_buffer: float = 2.0):
        """
        Args:
            dbscan_eps: DBSCAN 클러스터링 거리 (m)
            network_max_dist: 네트워크 연결 최대 거리 (m)
            road_buffer: 도로 경계선 버퍼 (m)
        """
        self.dbscan_eps = dbscan_eps
        self.network_max_dist = network_max_dist
        self.road_buffer = road_buffer
        
        # 역할 우선순위 (높을수록 우선)
        self.role_priority = {
            'intersection': 3,
            'curve': 2, 
            'endpoint': 1
        }
    
    def calculate_importance_score(self, point: Tuple[float, float], 
                                 skeleton: List[List[float]], 
                                 road_polygon=None) -> float:
        """점의 중요도 점수 계산"""
        x, y = point
        
        # 1. 스켈레톤 밀도 (100m 반경 내 점 개수 - 완화)
        density = 0
        for skel_point in skeleton:
            if len(skel_point) >= 2:
                dist = np.sqrt((x - skel_point[0])**2 + (y - skel_point[1])**2)
                if dist <= 100:  # 50m → 100m로 완화
                    density += 1
        
        # 2. 도로 경계선과의 거리 (가까울수록 높은 점수 - 완화)
        boundary_score = 1.0
        if road_polygon:
            point_geom = Point(x, y)
            dist_to_boundary = road_polygon.distance(point_geom)
            boundary_score = max(0.3, 1.0 - dist_to_boundary / 200.0)  # 더 관대한 계산
        
        # 3. 주변 점들과의 평균 거리 (적당한 거리에 있을 때 높은 점수)
        distances = []
        for skel_point in skeleton:
            if len(skel_point) >= 2:
                dist = np.sqrt((x - skel_point[0])**2 + (y - skel_point[1])**2)
                if 5 <= dist <= 100:  # 너무 가깝거나 멀지 않은 거리
                    distances.append(dist)
        
        avg_dist_score = 1.0
        if distances:
            avg_dist = np.mean(distances)
            # 20-30m 정도가 최적
            avg_dist_score = max(0.1, 1.0 - abs(avg_dist - 25) / 25)
        
        # 종합 점수 (0-1 범위)
        total_score = (density * 0.4 + boundary_score * 0.3 + avg_dist_score * 0.3) / 10
        return min(1.0, total_score)
    
    def filter_by_dbscan(self, points: List[Tuple[float, float]], 
                        importance_scores: Dict[Tuple[float, float], float]) -> List[Tuple[float, float]]:
        """DBSCAN 클러스터링으로 중복점 제거"""
        if len(points) < 2:
            return points
        
        # NumPy 배열로 변환
        points_array = np.array(points)
        
        # DBSCAN 클러스터링
        db = DBSCAN(eps=self.dbscan_eps, min_samples=1).fit(points_array)
        labels = db.labels_
        
        filtered_points = []
        for label in set(labels):
            cluster_mask = labels == label
            cluster_points = points_array[cluster_mask]
            
            if len(cluster_points) == 1:
                # 단일 점은 그대로 유지
                filtered_points.append(tuple(cluster_points[0]))
            else:
                # 클러스터 내에서 중요도가 가장 높은 점 선택
                cluster_tuples = [tuple(p) for p in cluster_points]
                best_point = max(cluster_tuples, 
                               key=lambda p: importance_scores.get(p, 0.0))
                filtered_points.append(best_point)
        
        logger.info(f"DBSCAN 필터링: {len(points)} → {len(filtered_points)}개")
        return filtered_points
    
    def filter_by_network_connectivity(self, points: List[Tuple[float, float]], 
                                     road_polygon) -> List[Tuple[float, float]]:
        """네트워크 연결성 기반 필터링"""
        if len(points) < 2 or road_polygon is None:
            return points
        
        # 네트워크 그래프 생성
        G = nx.Graph()
        G.add_nodes_from(range(len(points)))
        
        # 도로 경계선 내에서 연결 가능한 점들 연결
        for i, p1 in enumerate(points):
            for j, p2 in enumerate(points):
                if i < j:
                    # 거리 체크
                    dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                    if dist <= self.network_max_dist:
                        # 도로 경계선 내 연결 체크
                        line = LineString([p1, p2])
                        if road_polygon.buffer(self.road_buffer).contains(line):
                            G.add_edge(i, j, weight=dist)
        
        # 연결된 컴포넌트들을 모두 유지 (가장 큰 것만 유지하는 대신)
        if G.number_of_edges() > 0:
            components = list(nx.connected_components(G))
            # 모든 컴포넌트에서 점들을 유지 (단, 너무 작은 컴포넌트는 제거)
            filtered_points = []
            for component in components:
                if len(component) >= 1:  # 1개 이상이면 유지
                    filtered_points.extend([points[i] for i in component])
        else:
            filtered_points = points
        
        logger.info(f"네트워크 필터링: {len(points)} → {len(filtered_points)}개")
        return filtered_points
    
    def filter_by_role_priority(self, points: List[Tuple[float, float]], 
                              point_roles: Dict[Tuple[float, float], str]) -> List[Tuple[float, float]]:
        """역할 우선순위 기반 필터링"""
        if not points:
            return points
        
        # 역할별로 그룹화
        role_groups = {}
        for point in points:
            role = point_roles.get(point, 'curve')  # 기본값은 curve
            if role not in role_groups:
                role_groups[role] = []
            role_groups[role].append(point)
        
        # 우선순위가 높은 역할부터 처리
        filtered_points = []
        for role in sorted(role_groups.keys(), 
                          key=lambda r: self.role_priority.get(r, 0), 
                          reverse=True):
            filtered_points.extend(role_groups[role])
        
        logger.info(f"역할 우선순위 필터링: {len(points)}개 유지")
        return filtered_points

    def build_skeleton_graph(self, points: List[Tuple[float, float]], skeleton_lines: List[LineString], dist_thresh: float = 10.0):
        """스켈레톤 기반 점 연결 그래프 생성 (각 점을 모든 스켈레톤에 투영, 인접 연결)"""
        G = nx.Graph()
        for idx, pt in enumerate(points):
            G.add_node(idx, coord=pt)
        for skel in skeleton_lines:
            # 해당 스켈레톤에 가까운 점들(임계값 이내, 중복 포함 가능)
            skel_points = [(i, pt) for i, pt in enumerate(points) if skel.distance(Point(pt)) < dist_thresh]
            if len(skel_points) < 2:
                continue
            # 투영 위치 기준 정렬
            skel_points_sorted = sorted(skel_points, key=lambda x: skel.project(Point(x[1])))
            # 인접한 점끼리 연결
            for i in range(len(skel_points_sorted) - 1):
                idx1, c1 = skel_points_sorted[i]
                idx2, c2 = skel_points_sorted[i+1]
                dist = np.linalg.norm(np.array(c1) - np.array(c2))
                G.add_edge(idx1, idx2, weight=dist)
        return G

    def filter_by_skeleton_connectivity(self, points: List[Tuple[float, float]], skeleton_lines: List[LineString], point_roles: Dict[Tuple[float, float], str], dist_thresh: float = 10.0, curve_min_length: float = 20.0) -> List[Tuple[float, float]]:
        """스켈레톤 기반 연결 그래프에서 degree/거리/곡률 기반 필터링"""
        if not points or not skeleton_lines:
            return points
        G = self.build_skeleton_graph(points, skeleton_lines, dist_thresh)
        keep_idxs = set()
        for idx, pt in enumerate(points):
            deg = G.degree[idx]
            neighbors = list(G.neighbors(idx))
            if deg == 1 or deg >= 3:
                keep_idxs.add(idx)
            elif deg == 2:
                # 양쪽 이웃 거리의 합 계산
                dists = [np.linalg.norm(np.array(pt) - np.array(G.nodes[n]['coord'])) for n in neighbors]
                total_dist = sum(dists)
                # 곡률(간이) 계산: 세 점이 거의 일직선이면 곡률 작음
                if len(neighbors) == 2:
                    p0 = np.array(G.nodes[neighbors[0]]['coord'])
                    p1 = np.array(pt)
                    p2 = np.array(G.nodes[neighbors[1]]['coord'])
                    v1 = p0 - p1
                    v2 = p2 - p1
                    angle = np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1))
                    curvature = np.pi - angle  # 0에 가까우면 직선, 크면 커브
                else:
                    curvature = 0
                if total_dist <= curve_min_length:
                    continue  # 삭제
                else:
                    keep_idxs.add(idx)
        filtered_points = [pt for i, pt in enumerate(points) if i in keep_idxs]
        logger.info(f"스켈레톤 기반 필터링: {len(points)} → {len(filtered_points)}개")
        return filtered_points

    def hybrid_filter(self, 
                     points: List[Tuple[float, float]], 
                     skeleton: List[List[float]], 
                     road_polygon=None,
                     point_roles: Optional[Dict[Tuple[float, float], str]] = None) -> List[Tuple[float, float]]:
        """
        하이브리드 필터링 메인 함수
        
        Args:
            points: 필터링할 점들 [(x, y), ...]
            skeleton: 스켈레톤 데이터
            road_polygon: 도로 폴리곤 (Shapely)
            point_roles: 점별 역할 {'intersection', 'curve', 'endpoint'}
        
        Returns:
            필터링된 점들
        """
        if not points:
            return points
        
        logger.info(f"하이브리드 필터링 시작: {len(points)}개 점")
        
        # 1. 중요도 점수 계산
        importance_scores = {}
        for point in points:
            importance_scores[point] = self.calculate_importance_score(point, skeleton, road_polygon)
        
        # 2. DBSCAN 클러스터링 필터
        filtered_points = self.filter_by_dbscan(points, importance_scores)
        
        # 3. 네트워크 연결성 필터
        filtered_points = self.filter_by_network_connectivity(filtered_points, road_polygon)
        
        # 4. 역할 우선순위 필터 (역할 정보가 있는 경우)
        if point_roles:
            filtered_points = self.filter_by_role_priority(filtered_points, point_roles)
        
        # 5. 스켈레톤 기반 연결 필터 우선 적용 (더 엄격하게)
        skeleton_lines = []
        if skeleton and len(skeleton) > 1:
            skeleton_lines.append(LineString(skeleton))
        filtered_points = self.filter_by_skeleton_connectivity(filtered_points, skeleton_lines, point_roles or {}, dist_thresh=15.0, curve_min_length=50.0)
        
        # 6. 추가 거리 기반 필터링 (너무 가까운 점들 제거)
        if len(filtered_points) > 1:
            final_points = [filtered_points[0]]
            for point in filtered_points[1:]:
                min_dist_to_existing = min([
                    np.sqrt((point[0] - existing[0])**2 + (point[1] - existing[1])**2)
                    for existing in final_points
                ])
                if min_dist_to_existing >= 8.0:  # 8m 이상 떨어진 점들만 유지
                    final_points.append(point)
            filtered_points = final_points
        
        logger.info(f"하이브리드 필터링 완료: {len(points)} → {len(filtered_points)}개")
        return filtered_points


def create_hybrid_filter(dbscan_eps: float = 20.0, 
                        network_max_dist: float = 50.0, 
                        road_buffer: float = 2.0) -> HybridPointFilter:
    """하이브리드 필터 팩토리 함수"""
    return HybridPointFilter(dbscan_eps, network_max_dist, road_buffer) 