"""
강화된 통합 휴리스틱 검출기
- 더 정확한 교차점 검출
- 개선된 커브 검출 (각도 + 연속성 분석)
- 안정적인 끝점 검출
- 노이즈 필터링 강화
"""

import numpy as np
import geopandas as gpd
from shapely.geometry import Point, LineString, MultiPoint
from shapely.ops import unary_union
from scipy.spatial.distance import cdist
import logging

logger = logging.getLogger(__name__)


class IntegratedHeuristicDetector:
    """강화된 통합 휴리스틱 검출기"""
    
    def __init__(self):
        # 파라미터 최적화
        self.intersection_params = {
            'min_distance': 15,        # 교차점 최소 간격
            'buffer_size': 5,          # 교차점 버퍼
            'min_lines': 2            # 최소 교차 라인 수
        }
        
        self.curve_params = {
            'window_size': 5,          # 각도 계산 윈도우
            'angle_threshold': 25,     # 커브 각도 임계값 (더 민감)
            'min_curvature': 0.1,     # 최소 곡률
            'group_distance': 40,      # 커브 그룹화 거리
            'min_group_size': 3        # 최소 그룹 크기
        }
        
        self.endpoint_params = {
            'edge_distance': 20,       # 끝점 판정 거리
            'min_endpoint_distance': 25  # 끝점 간 최소 거리
        }
    
    def detect_intersections(self, gdf, skeleton):
        """향상된 교차점 검출"""
        try:
            intersections = []
            skeleton_array = np.array(skeleton)
            
            # 1. 기하학적 교차점 검출
            lines = []
            for _, row in gdf.iterrows():
                if hasattr(row.geometry, 'coords'):
                    coords = list(row.geometry.coords)
                    if len(coords) >= 2:
                        lines.append(LineString(coords))
            
            if len(lines) < 2:
                return []
            
            # 2. 라인 간 교차점 찾기
            raw_intersections = []
            for i, line1 in enumerate(lines):
                for j, line2 in enumerate(lines[i+1:], i+1):
                    try:
                        intersection = line1.intersection(line2)
                        
                        if intersection.is_empty:
                            continue
                        
                        # Point 객체 처리
                        if hasattr(intersection, 'x'):
                            raw_intersections.append((intersection.x, intersection.y))
                        # MultiPoint 객체 처리
                        elif hasattr(intersection, 'geoms'):
                            for geom in intersection.geoms:
                                if hasattr(geom, 'x'):
                                    raw_intersections.append((geom.x, geom.y))
                        # 좌표 직접 접근
                        elif hasattr(intersection, 'coords'):
                            raw_intersections.extend(list(intersection.coords))
                            
                    except Exception as e:
                        logger.warning(f"교차점 계산 오류: {e}")
                        continue
            
            if not raw_intersections:
                return []
            
            # 3. 스켈레톤과 매칭
            skeleton_intersections = []
            for intersection_point in raw_intersections:
                try:
                    # 가장 가까운 스켈레톤 포인트 찾기
                    distances = [
                        np.linalg.norm(np.array(intersection_point) - np.array(skel_point))
                        for skel_point in skeleton
                    ]
                    
                    min_distance = min(distances)
                    if min_distance < self.intersection_params['buffer_size']:
                        closest_idx = np.argmin(distances)
                        closest_point = tuple(skeleton[closest_idx])
                        skeleton_intersections.append(closest_point)
                        
                except Exception as e:
                    logger.warning(f"스켈레톤 매칭 오류: {e}")
                    continue
            
            # 4. 중복 제거 및 필터링
            filtered_intersections = []
            for point in skeleton_intersections:
                is_duplicate = False
                for existing in filtered_intersections:
                    distance = np.linalg.norm(np.array(point) - np.array(existing))
                    if distance < self.intersection_params['min_distance']:
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    filtered_intersections.append(point)
            
            logger.info(f"교차점 검출: {len(raw_intersections)} -> {len(filtered_intersections)}")
            return filtered_intersections
            
        except Exception as e:
            logger.error(f"교차점 검출 오류: {e}")
            return []
    
    def detect_curves(self, skeleton):
        """향상된 커브 검출"""
        try:
            if len(skeleton) < self.curve_params['window_size'] * 2:
                return []
            
            skeleton_array = np.array(skeleton)
            curve_indices = []
            
            # 1. 각도 기반 커브 검출
            window = self.curve_params['window_size']
            
            for i in range(window, len(skeleton_array) - window):
                try:
                    # 윈도우 포인트들
                    prev_points = skeleton_array[i-window:i]
                    next_points = skeleton_array[i:i+window]
                    current_point = skeleton_array[i]
                    
                    # 방향 벡터 계산
                    if len(prev_points) > 0 and len(next_points) > 0:
                        prev_direction = np.mean(prev_points, axis=0) - current_point
                        next_direction = np.mean(next_points, axis=0) - current_point
                        
                        # 각도 계산
                        norm_prev = np.linalg.norm(prev_direction)
                        norm_next = np.linalg.norm(next_direction)
                        
                        if norm_prev > 0 and norm_next > 0:
                            cos_angle = np.dot(prev_direction, next_direction) / (norm_prev * norm_next)
                            cos_angle = np.clip(cos_angle, -1, 1)
                            angle = np.degrees(np.arccos(cos_angle))
                            
                            # 커브 판정
                            if angle > self.curve_params['angle_threshold']:
                                curve_indices.append(i)
                                
                except Exception as e:
                    logger.warning(f"각도 계산 오류 at {i}: {e}")
                    continue
            
            if not curve_indices:
                return []
            
            # 2. 연속된 커브 포인트 그룹화
            curve_groups = []
            current_group = [curve_indices[0]]
            
            for i in range(1, len(curve_indices)):
                # 연속성 확인
                if curve_indices[i] - current_group[-1] <= self.curve_params['group_distance']:
                    current_group.append(curve_indices[i])
                else:
                    # 그룹 완료
                    if len(current_group) >= self.curve_params['min_group_size']:
                        curve_groups.append(current_group)
                    current_group = [curve_indices[i]]
            
            # 마지막 그룹 추가
            if len(current_group) >= self.curve_params['min_group_size']:
                curve_groups.append(current_group)
            
            # 3. 대표점 선택 (각 그룹의 중점)
            curve_points = []
            for group in curve_groups:
                mid_idx = group[len(group) // 2]
                curve_points.append(tuple(skeleton_array[mid_idx]))
            
            logger.info(f"커브 검출: {len(curve_indices)} 후보 -> {len(curve_points)} 그룹")
            return curve_points
            
        except Exception as e:
            logger.error(f"커브 검출 오류: {e}")
            return []
    
    def detect_endpoints(self, skeleton):
        """향상된 끝점 검출"""
        try:
            if len(skeleton) < 2:
                return []
            
            skeleton_array = np.array(skeleton)
            endpoints = []
            
            # 1. 스켈레톤 양 끝 확인
            start_candidates = skeleton_array[:self.endpoint_params['edge_distance']]
            end_candidates = skeleton_array[-self.endpoint_params['edge_distance']:]
            
            # 2. 시작점 선택
            if len(start_candidates) > 0:
                # 가장 극단적인 점 선택
                start_point = start_candidates[0]
                endpoints.append(tuple(start_point))
            
            # 3. 끝점 선택
            if len(end_candidates) > 0:
                end_point = end_candidates[-1]
                
                # 시작점과 충분히 떨어져 있는지 확인
                if len(endpoints) > 0:
                    distance = np.linalg.norm(end_point - np.array(endpoints[0]))
                    if distance >= self.endpoint_params['min_endpoint_distance']:
                        endpoints.append(tuple(end_point))
                else:
                    endpoints.append(tuple(end_point))
            
            # 4. 추가 끝점 검출 (분기점이나 고립점)
            # TODO: 필요시 구현
            
            logger.info(f"끝점 검출: {len(endpoints)}개")
            return endpoints
            
        except Exception as e:
            logger.error(f"끝점 검출 오류: {e}")
            return []
    
    def detect_all(self, gdf, skeleton, intersections=None):
        """전체 검출 실행"""
        try:
            results = {}
            
            # 교차점 검출
            if intersections is not None:
                results['intersection'] = intersections
            else:
                results['intersection'] = self.detect_intersections(gdf, skeleton)
            
            # 커브 검출
            results['curve'] = self.detect_curves(skeleton)
            
            # 끝점 검출
            results['endpoint'] = self.detect_endpoints(skeleton)
            
            # 통계 로깅
            total = sum(len(points) for points in results.values())
            logger.info(f"전체 검출 완료: {total}개 (교차:{len(results['intersection'])}, "
                       f"커브:{len(results['curve'])}, 끝점:{len(results['endpoint'])})")
            
            return results
            
        except Exception as e:
            logger.error(f"전체 검출 오류: {e}")
            return {'intersection': [], 'curve': [], 'endpoint': []}
    
    def optimize_parameters(self, test_results=None):
        """파라미터 자동 최적화 (향후 구현)"""
        # TODO: 테스트 결과를 바탕으로 파라미터 자동 조정
        pass

# 편의 함수들
def quick_detect(shapefile_path):
    """빠른 검출"""
    import geopandas as gpd
    from src.core.skeleton_extractor import SkeletonExtractor
    
    gdf = gpd.read_file(shapefile_path)
    extractor = SkeletonExtractor()
    skeleton, intersections = extractor.process_shapefile(shapefile_path)
    
    detector = IntegratedHeuristicDetector()
    results = detector.detect_all(gdf, skeleton, intersections)
    
    return results, skeleton

def benchmark_detector(test_files):
    """검출기 성능 벤치마크"""
    detector = IntegratedHeuristicDetector()
    results = []
    
    for file_path in test_files:
        try:
            import time
            start_time = time.time()
            
            detection_results, skeleton = quick_detect(file_path)
            
            elapsed = time.time() - start_time
            results.append({
                'file': file_path,
                'time': elapsed,
                'skeleton_points': len(skeleton),
                'detected_points': sum(len(points) for points in detection_results.values()),
                'results': detection_results
            })
            
        except Exception as e:
            logger.error(f"벤치마크 오류 {file_path}: {e}")
    
    return results
