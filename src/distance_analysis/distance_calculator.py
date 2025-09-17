"""
고도화된 거리 계산기 (메인 통합 모듈)
point_sample 방식을 활용한 종합 거리 분석 시스템
"""

import numpy as np
from shapely.geometry import Point, LineString
from shapely.ops import unary_union
import logging
from typing import List, Dict, Tuple, Optional

from .network_connectivity import NetworkConnectivityAnalyzer
from .visual_connectivity import VisualConnectivityChecker
from .importance_scorer import ImportanceScorer

logger = logging.getLogger(__name__)

class AdvancedDistanceCalculator:
    """point_sample 방식 고도화 거리 계산기"""
    
    def __init__(self, skeleton_points, road_polygons, 
                 max_distance=300.0, min_distance=15.0):
        """
        Args:
            skeleton_points: 스켈레톤 점들 리스트 [(x, y), ...]
            road_polygons: 도로 폴리곤들 리스트
            max_distance: 최대 연결 거리 (기본 300m)
            min_distance: 최소 연결 거리 (기본 15m)
        """
        self.skeleton_points = skeleton_points or []
        self.road_polygons = road_polygons or []
        self.max_distance = max_distance
        self.min_distance = min_distance
        
        # 하위 분석기들 초기화
        self.network_analyzer = NetworkConnectivityAnalyzer(
            skeleton_points, road_polygons, max_distance, min_distance
        )
        
        self.visual_checker = VisualConnectivityChecker(
            skeleton_points, road_polygons, max_distance, min_distance
        )
        
        self.importance_scorer = ImportanceScorer(
            skeleton_points, road_polygons
        )
        
        # 결과 캐시
        self.last_analysis_result = None
        
        logger.info(f"AdvancedDistanceCalculator 초기화 완료")
    
    def calculate_optimal_distances(self, points_data: Dict[str, List[Tuple[float, float]]]) -> Dict:
        """
        최적화된 거리 계산 (메인 함수)
        
        Args:
            points_data: {
                'intersection': [(x, y), ...],
                'curve': [(x, y), ...],
                'endpoint': [(x, y), ...]
            }
            
        Returns:
            Dict: 종합 분석 결과
        """
        logger.info("최적화된 거리 계산 시작")
        
        # 1. 점 데이터 전처리
        points_with_metadata = self._prepare_points_data(points_data)
        
        if not points_with_metadata:
            logger.warning("분석할 점이 없습니다.")
            return self._create_empty_result()
        
        # 2. 네트워크 연결성 분석
        logger.info("네트워크 연결성 분석 중...")
        network_graph = self.network_analyzer.build_road_graph(points_with_metadata)
        network_connections = self.network_analyzer.get_connected_pairs()
        
        # 3. 시각적 연결성 분석
        logger.info("시각적 연결성 분석 중...")
        visual_connections = self.visual_checker.get_visual_connections(points_with_metadata)
        
        # 4. 연결 통합 및 중복 제거
        logger.info("연결 통합 및 우선순위 계산 중...")
        integrated_connections = self._integrate_connections(
            network_connections, visual_connections, points_with_metadata
        )
        
        # 5. 중요도 기반 우선순위 계산 (point_sample 방식: 더 엄격)
        priority_connections = self.importance_scorer.get_top_connections(
            integrated_connections, 
            top_n=20,  # 상위 20개만 선택
            min_priority=0.3  # 최소 우선순위 30% (완화)
        )
        
        # 6. 결과 생성
        result = self._create_analysis_result(
            points_with_metadata,
            priority_connections,
            network_graph
        )
        
        # 결과 캐시
        self.last_analysis_result = result
        
        logger.info(f"거리 계산 완료: {len(priority_connections)}개 연결")
        return result
    
    def _prepare_points_data(self, points_data: Dict[str, List[Tuple[float, float]]]) -> List[Tuple]:
        """점 데이터를 메타데이터와 함께 준비"""
        points_with_metadata = []
        
        for category, points in points_data.items():
            for idx, (x, y) in enumerate(points):
                points_with_metadata.append((x, y, category, idx))
        
        logger.info(f"점 데이터 준비 완료: {len(points_with_metadata)}개")
        return points_with_metadata
    
    def _integrate_connections(self, network_connections, visual_connections, points_metadata):
        """네트워크 연결과 시각적 연결을 통합"""
        connection_map = {}
        
        # 네트워크 연결 추가
        for idx1, idx2, distance in network_connections:
            key = tuple(sorted([idx1, idx2]))
            if key not in connection_map:
                connection_map[key] = {
                    'idx1': idx1,
                    'idx2': idx2,
                    'distance': distance,
                    'has_network': True,
                    'has_visual': False
                }
        
        # 시각적 연결 추가/보완
        for idx1, idx2, distance in visual_connections:
            key = tuple(sorted([idx1, idx2]))
            if key in connection_map:
                connection_map[key]['has_visual'] = True
            else:
                connection_map[key] = {
                    'idx1': idx1,
                    'idx2': idx2,
                    'distance': distance,
                    'has_network': False,
                    'has_visual': True
                }
        
        # 메타데이터 추가
        integrated_connections = []
        for connection in connection_map.values():
            idx1, idx2 = connection['idx1'], connection['idx2']
            
            point1 = points_metadata[idx1]
            point2 = points_metadata[idx2]
            
            integrated_connections.append((
                idx1, idx2, connection['distance'],
                (point1[0], point1[1]),  # point1 좌표
                (point2[0], point2[1]),  # point2 좌표
                point1[2],  # category1
                point2[2]   # category2
            ))
        
        return integrated_connections
    
    def _create_analysis_result(self, points_metadata, priority_connections, network_graph):
        """분석 결과 생성"""
        result = {
            'connections': [],
            'statistics': {},
            'points_info': {},
            'network_metrics': {}
        }
        
        # 연결 정보 생성
        for conn in priority_connections:
            result['connections'].append({
                'idx1': conn['idx1'],
                'idx2': conn['idx2'],
                'point1': conn['point1'],
                'point2': conn['point2'],
                'distance': conn['distance'],
                'category1': conn['category1'],
                'category2': conn['category2'],
                'priority': conn['priority']
            })
        
        # 통계 정보 생성
        if priority_connections:
            distances = [conn['distance'] for conn in priority_connections]
            priorities = [conn['priority'] for conn in priority_connections]
            
            result['statistics'] = {
                'total_connections': len(priority_connections),
                'distance_stats': {
                    'min': float(np.min(distances)),
                    'max': float(np.max(distances)),
                    'mean': float(np.mean(distances)),
                    'median': float(np.median(distances)),
                    'std': float(np.std(distances))
                },
                'priority_stats': {
                    'min': float(np.min(priorities)),
                    'max': float(np.max(priorities)),
                    'mean': float(np.mean(priorities)),
                    'median': float(np.median(priorities))
                },
                'distance_distribution': self._get_distance_distribution(distances)
            }
        
        # 점 정보 생성
        for i, (x, y, category, idx) in enumerate(points_metadata):
            importance = self.importance_scorer.calculate_point_importance((x, y), category)
            
            result['points_info'][i] = {
                'coordinates': (x, y),
                'category': category,
                'original_index': idx,
                'importance': importance,
                'connected_count': sum(1 for conn in priority_connections 
                                     if conn['idx1'] == i or conn['idx2'] == i)
            }
        
        # 네트워크 메트릭스
        result['network_metrics'] = self.network_analyzer.get_network_statistics()
        
        return result
    
    def _get_distance_distribution(self, distances):
        """거리 분포 계산"""
        if not distances:
            return {'short': 0, 'medium': 0, 'long': 0}
        
        short_count = sum(1 for d in distances if d <= 50)
        medium_count = sum(1 for d in distances if 50 < d <= 150)
        long_count = sum(1 for d in distances if d > 150)
        
        return {
            'short': short_count,
            'medium': medium_count,
            'long': long_count
        }
    
    def _create_empty_result(self):
        """빈 결과 생성"""
        return {
            'connections': [],
            'statistics': {
                'total_connections': 0,
                'distance_stats': {},
                'priority_stats': {},
                'distance_distribution': {'short': 0, 'medium': 0, 'long': 0}
            },
            'points_info': {},
            'network_metrics': {}
        }
    
    def get_canvas_display_data(self, result: Optional[Dict] = None) -> List[Dict]:
        """
        Canvas 표시용 데이터 반환
        
        Returns:
            List[Dict]: Canvas에서 표시할 연결 데이터
        """
        if result is None:
            result = self.last_analysis_result
        
        if not result or not result['connections']:
            return []
        
        display_data = []
        
        for conn in result['connections']:
            display_data.append({
                'point1': conn['point1'],
                'point2': conn['point2'],
                'distance': conn['distance'],
                'priority': conn['priority'],
                'category1': conn['category1'],
                'category2': conn['category2'],
                'display_text': f"{conn['distance']:.1f}m"
            })
        
        return display_data
    
    def get_statistics_text(self, result: Optional[Dict] = None) -> str:
        """
        통계 정보 텍스트 반환
        
        Returns:
            str: 통계 정보 문자열
        """
        if result is None:
            result = self.last_analysis_result
        
        if not result or not result['statistics']:
            return "분석 결과가 없습니다."
        
        stats = result['statistics']
        
        if stats['total_connections'] == 0:
            return "연결된 점이 없습니다."
        
        dist_stats = stats['distance_stats']
        priority_stats = stats['priority_stats']
        dist_dist = stats['distance_distribution']
        
        text = f"""거리 분석 결과:
총 연결: {stats['total_connections']}개
평균 거리: {dist_stats['mean']:.1f}m
거리 범위: {dist_stats['min']:.1f}m ~ {dist_stats['max']:.1f}m
평균 우선순위: {priority_stats['mean']:.2f}

거리 분포:
• 단거리 (≤50m): {dist_dist['short']}개
• 중거리 (50~150m): {dist_dist['medium']}개  
• 장거리 (>150m): {dist_dist['long']}개"""
        
        return text
    
    def update_parameters(self, max_distance: Optional[float] = None, min_distance: Optional[float] = None):
        """분석 파라미터 업데이트"""
        if max_distance is not None:
            self.max_distance = max_distance
            self.network_analyzer.max_distance = max_distance
            self.visual_checker.max_distance = max_distance
        
        if min_distance is not None:
            self.min_distance = min_distance
            self.network_analyzer.min_distance = min_distance
            self.visual_checker.min_distance = min_distance
        
        logger.info(f"파라미터 업데이트: 거리 범위 {self.min_distance}m~{self.max_distance}m") 