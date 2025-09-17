"""
중요도 점수 계산 모듈
point_sample/analyze_curve_importance.py와 analyze_road_importance.py 기반
"""

import numpy as np
from shapely.geometry import Point, LineString
from shapely.ops import unary_union
import logging

logger = logging.getLogger(__name__)

class ImportanceScorer:
    """점 중요도 및 연결 우선순위 계산"""
    
    def __init__(self, skeleton_points, road_polygons):
        """
        Args:
            skeleton_points: 스켈레톤 점들 리스트
            road_polygons: 도로 폴리곤들
        """
        self.skeleton_points = skeleton_points
        self.road_polygons = road_polygons
        
        # 도로 union 생성
        self.road_union = unary_union(road_polygons) if road_polygons else None
        
        # 스켈레톤 기반 도로 중요도 분석
        self.skeleton_importance = self._calculate_skeleton_importance()
        
        logger.info(f"ImportanceScorer 초기화: 스켈레톤 점 {len(skeleton_points)}개")
    
    def _calculate_skeleton_importance(self):
        """스켈레톤 점들의 중요도 계산"""
        importance_scores = {}
        
        if not self.skeleton_points:
            return importance_scores
        
        # 각 스켈레톤 점에서 반경 30m 내 이웃 점들 찾기
        for i, point in enumerate(self.skeleton_points):
            neighbors = []
            
            for j, other_point in enumerate(self.skeleton_points):
                if i == j:
                    continue
                
                distance = np.sqrt((point[0] - other_point[0])**2 + 
                                 (point[1] - other_point[1])**2)
                
                if distance <= 30.0:  # 30m 이내 이웃
                    neighbors.append((j, distance))
            
            # 중요도 점수 계산
            # 1. 연결도 점수 (이웃 점이 많을수록 중요)
            connectivity_score = min(len(neighbors) / 10.0, 1.0)  # 최대 10개 이웃
            
            # 2. 도로 폭 점수 (도로가 넓을수록 중요)
            road_width_score = self._calculate_road_width_score(point)
            
            # 3. 중심성 점수 (도로망 중심에 가까울수록 중요)
            centrality_score = self._calculate_centrality_score(point, neighbors)
            
            # 총 중요도 점수
            total_importance = (connectivity_score * 0.4 + 
                              road_width_score * 0.3 + 
                              centrality_score * 0.3)
            
            importance_scores[i] = {
                'total': total_importance,
                'connectivity': connectivity_score,
                'road_width': road_width_score,
                'centrality': centrality_score,
                'neighbors': len(neighbors)
            }
        
        return importance_scores
    
    def _calculate_road_width_score(self, point):
        """점 주변 도로 폭 기반 점수 계산"""
        if not self.road_union:
            return 0.5
        
        point_geom = Point(point)
        
        # 점에서 반경 20m 내 도로 영역 계산
        buffer_20m = point_geom.buffer(20.0)
        
        try:
            road_intersection = self.road_union.intersection(buffer_20m)
            
            if road_intersection.is_empty:
                return 0.0
            
            # 도로 영역 비율 계산
            road_area = road_intersection.area if hasattr(road_intersection, 'area') else 0
            total_area = buffer_20m.area
            
            area_ratio = road_area / total_area
            
            # 0.0 ~ 1.0 범위로 정규화
            return min(area_ratio * 2.0, 1.0)
            
        except Exception as e:
            logger.warning(f"도로 폭 계산 중 오류: {e}")
            return 0.5
    
    def _calculate_centrality_score(self, point, neighbors):
        """점의 중심성 점수 계산"""
        if not neighbors:
            return 0.0
        
        # 이웃 점들과의 평균 거리 (가까울수록 중심성 높음)
        distances = [dist for _, dist in neighbors]
        avg_distance = np.mean(distances)
        
        # 거리 기반 중심성 점수 (0~30m 범위에서 계산)
        centrality_score = max(0, 1 - (avg_distance / 30.0))
        
        return centrality_score
    
    def calculate_point_importance(self, point, category):
        """
        특정 점의 중요도 계산
        
        Args:
            point: (x, y) 좌표
            category: 점 유형 ('intersection', 'curve', 'endpoint')
            
        Returns:
            float: 중요도 점수 (0.0 ~ 1.0)
        """
        # 카테고리별 기본 중요도
        category_weights = {
            'intersection': 1.0,  # 교차점이 가장 중요
            'curve': 0.7,         # 곡선점 중간 중요도
            'endpoint': 0.5       # 끝점 낮은 중요도
        }
        
        base_importance = category_weights.get(category, 0.5)
        
        # 가장 가까운 스켈레톤 점 찾기
        min_distance = float('inf')
        closest_skeleton_idx = None
        
        for i, skeleton_point in enumerate(self.skeleton_points):
            distance = np.sqrt((point[0] - skeleton_point[0])**2 + 
                             (point[1] - skeleton_point[1])**2)
            
            if distance < min_distance:
                min_distance = distance
                closest_skeleton_idx = i
        
        # 스켈레톤 기반 중요도 보정
        skeleton_importance = 0.5
        if closest_skeleton_idx is not None and closest_skeleton_idx in self.skeleton_importance:
            skeleton_importance = self.skeleton_importance[closest_skeleton_idx]['total']
        
        # 거리 기반 보정 (스켈레톤에 가까울수록 중요)
        distance_factor = max(0, 1 - (min_distance / 50.0))  # 50m 이내
        
        # 최종 중요도 계산
        final_importance = (base_importance * 0.4 + 
                          skeleton_importance * 0.4 + 
                          distance_factor * 0.2)
        
        return min(final_importance, 1.0)
    
    def calculate_connection_priority(self, point1, point2, category1, category2):
        """
        두 점 사이 연결의 우선순위 계산
        
        Args:
            point1, point2: (x, y) 좌표
            category1, category2: 점 유형
            
        Returns:
            float: 연결 우선순위 점수 (0.0 ~ 1.0)
        """
        # 각 점의 중요도 계산
        importance1 = self.calculate_point_importance(point1, category1)
        importance2 = self.calculate_point_importance(point2, category2)
        
        # 두 점의 평균 중요도
        avg_importance = (importance1 + importance2) / 2.0
        
        # 거리 기반 우선순위 (적정 거리일수록 높음)
        distance = np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
        
        # 50m~150m가 최적 연결 거리
        if 50 <= distance <= 150:
            distance_priority = 1.0
        elif 15 <= distance < 50:
            distance_priority = 0.8
        elif 150 < distance <= 300:
            distance_priority = 0.6
        else:
            distance_priority = 0.2
        
        # 카테고리 조합 우선순위
        category_priority = self._get_category_combination_priority(category1, category2)
        
        # 최종 우선순위 계산 (point_sample 방식: 더 엄격한 가중치)
        final_priority = (avg_importance * 0.3 + 
                        distance_priority * 0.5 +  # 거리를 더 중요하게
                        category_priority * 0.2)
        
        return min(final_priority, 1.0)
    
    def _get_category_combination_priority(self, category1, category2):
        """카테고리 조합별 우선순위 반환"""
        # 카테고리 조합 우선순위 매트릭스
        priority_matrix = {
            ('intersection', 'intersection'): 1.0,  # 교차점 간 연결 최우선
            ('intersection', 'curve'): 0.8,
            ('intersection', 'endpoint'): 0.7,
            ('curve', 'curve'): 0.6,
            ('curve', 'endpoint'): 0.5,
            ('endpoint', 'endpoint'): 0.4,
        }
        
        # 순서 무관하게 조합 찾기
        combination = tuple(sorted([category1, category2]))
        
        return priority_matrix.get(combination, 0.5)
    
    def rank_connections(self, connections_with_metadata):
        """
        연결들을 우선순위순으로 정렬
        
        Args:
            connections_with_metadata: [(idx1, idx2, distance, point1, point2, cat1, cat2), ...]
            
        Returns:
            List: 우선순위순으로 정렬된 연결 리스트
        """
        ranked_connections = []
        
        for connection in connections_with_metadata:
            idx1, idx2, distance, point1, point2, cat1, cat2 = connection
            
            # 연결 우선순위 계산
            priority = self.calculate_connection_priority(point1, point2, cat1, cat2)
            
            ranked_connections.append({
                'idx1': idx1,
                'idx2': idx2,
                'distance': distance,
                'point1': point1,
                'point2': point2,
                'category1': cat1,
                'category2': cat2,
                'priority': priority
            })
        
        # 우선순위 높은 순으로 정렬
        ranked_connections.sort(key=lambda x: x['priority'], reverse=True)
        
        logger.info(f"연결 우선순위 계산 완료: {len(ranked_connections)}개 연결")
        return ranked_connections
    
    def get_top_connections(self, connections_with_metadata, top_n=None, min_priority=0.5):
        """
        상위 우선순위 연결들만 반환
        
        Args:
            connections_with_metadata: 연결 메타데이터 리스트
            top_n: 상위 N개 연결 (None이면 모든 연결)
            min_priority: 최소 우선순위 임계값
            
        Returns:
            List: 필터링된 연결 리스트
        """
        ranked_connections = self.rank_connections(connections_with_metadata)
        
        # 최소 우선순위 필터링
        filtered_connections = [
            conn for conn in ranked_connections 
            if conn['priority'] >= min_priority
        ]
        
        # 상위 N개 선택
        if top_n is not None:
            filtered_connections = filtered_connections[:top_n]
        
        logger.info(f"상위 연결 선택: {len(filtered_connections)}개 (최소 우선순위: {min_priority})")
        return filtered_connections 