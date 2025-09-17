"""
데이터 처리 모듈 - 스켈레톤 추출, 파일 로드 등
"""

import os
import tempfile
from pathlib import Path
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from typing import Dict, List, Tuple, Optional, Any
import logging

from ..core.skeleton_extractor import SkeletonExtractor
from ..core.district_road_clipper import DistrictRoadClipper

logger = logging.getLogger(__name__)


class DataProcessor:
    """데이터 처리 클래스"""
    
    def __init__(self):
        self.skeleton_extractor = SkeletonExtractor()
        self.district_clipper = DistrictRoadClipper()
        
    def extract_skeleton_from_file(self, file_path: str, target_crs: str = 'EPSG:5186') -> Dict[str, Any]:
        """파일에서 스켈레톤 추출
        
        Args:
            file_path: 파일 경로
            target_crs: 대상 좌표계 (기본값: EPSG:5186)
        """
        try:
            result = self.skeleton_extractor.extract_from_shapefile(file_path, target_crs)
            
            if not result or 'skeleton' not in result:
                raise ValueError("스켈레톤 추출 실패")
                
            return result
            
        except Exception as e:
            logger.error(f"스켈레톤 추출 오류: {e}")
            raise
    
    def process_road_file(self, file_path: str, target_crs: str = 'EPSG:5186') -> Dict[str, Any]:
        """도로망 파일 처리
        
        Args:
            file_path: 파일 경로
            target_crs: 대상 좌표계 (기본값: EPSG:5186)
        """
        try:
            # 스켈레톤 추출
            skeleton, intersections = self.skeleton_extractor.process_shapefile(file_path, target_crs)
            
            # 도로망 데이터 로드
            road_gdf = gpd.read_file(file_path)
            
            return {
                'skeleton': skeleton,
                'intersections': intersections,
                'road_gdf': road_gdf,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"도로망 파일 처리 오류: {e}")
            return {'success': False, 'error': str(e)}
    
    def process_district_file(self, district_file: str, target_crs: str = 'EPSG:5186') -> Dict[str, Any]:
        """지구계 파일 처리"""
        try:
            results = self.district_clipper.process_district_file(
                district_file,
                target_crs=target_crs,
                auto_find_road=True
            )
            
            return results
            
        except Exception as e:
            logger.error(f"지구계 파일 처리 오류: {e}")
            return {'success': False, 'error': str(e)}
    
    def detect_heuristic_endpoints(self, skeleton: List[List[float]], 
                                 road_bounds: Optional[Tuple[float, float, float, float]] = None) -> List[Tuple[float, float]]:
        """휴리스틱 끝점 검출"""
        if not skeleton:
            return []
        
        endpoints = []
        
        # 스켈레톤 포인트들의 경계 계산
        if skeleton:
            x_coords = [pt[0] for pt in skeleton if len(pt) >= 2]
            y_coords = [pt[1] for pt in skeleton if len(pt) >= 2]
            
            if x_coords and y_coords:
                min_x, max_x = min(x_coords), max(x_coords)
                min_y, max_y = min(y_coords), max(y_coords)
                
                # 경계로부터의 거리 임계값 (30m)
                threshold = 30.0
                
                for i, point in enumerate(skeleton):
                    if len(point) < 2:
                        continue
                    
                    x, y = float(point[0]), float(point[1])
                    
                    # 경계와의 거리 계산
                    dist_to_boundary = min(
                        x - min_x, max_x - x,  # 좌우 경계
                        y - min_y, max_y - y   # 상하 경계
                    )
                    
                    # 경계 근처이고 연결된 점이 적으면 끝점
                    if dist_to_boundary < threshold:
                        # 주변 연결점 개수 확인
                        connected_count = 0
                        for other_point in skeleton:
                            if len(other_point) < 2:
                                continue
                            other_x, other_y = float(other_point[0]), float(other_point[1])
                            if (x, y) != (other_x, other_y):
                                dist = np.sqrt((x - other_x)**2 + (y - other_y)**2)
                                if dist < 50:  # 50m 이내 연결점
                                    connected_count += 1
                        
                        # 연결점이 2개 이하면 끝점으로 판단
                        if connected_count <= 2:
                            endpoints.append((x, y))
        
        logger.info(f"🔚 휴리스틱 끝점 검출: {len(endpoints)}개")
        return endpoints
    
    def detect_boundary_based_curves(self, skeleton: List[List[float]], 
                                   sample_distance: float = 15.0,
                                   curvature_threshold: float = 0.20,
                                   road_buffer: float = 3.0,
                                   cluster_radius: float = 20.0) -> List[Tuple[float, float]]:
        """도로 경계선 기반 커브점 검출"""
        if not skeleton or len(skeleton) < 5:
            logger.info("스켈레톤이 너무 짧아 경계선 기반 커브 검출 불가")
            return []
        
        try:
            from shapely.geometry import LineString
            from sklearn.cluster import DBSCAN
            
            # 전체 스켈레톤을 하나의 도로망으로 통합
            skeleton_coords = [(float(p[0]), float(p[1])) for p in skeleton if len(p) >= 2]
            if len(skeleton_coords) < 2:
                return []
            
            # 연속된 좌표들을 LineString으로 변환
            skeleton_line = LineString(skeleton_coords)
            
            # 통합된 도로에 버퍼 적용
            road_polygon = skeleton_line.buffer(road_buffer)
            
            # 복잡한 도로 형태 처리
            if road_polygon.geom_type == 'Polygon':
                boundaries = [road_polygon.exterior]
            elif road_polygon.geom_type == 'MultiPolygon':
                boundaries = [poly.exterior for poly in road_polygon.geoms]
            else:
                logger.warning(f"예상치 못한 geometry 타입: {road_polygon.geom_type}")
                return []
            
            # 모든 경계선에서 커브점 검출
            all_curvature_points = []
            
            for boundary in boundaries:
                total_length = boundary.length
                if total_length < sample_distance:
                    continue
            
                # 각 경계선을 따라 샘플링
                num_samples = max(10, int(total_length / sample_distance))
                
                for i in range(num_samples):
                    distance = (i * sample_distance) % total_length
                    
                    # 곡률 계산
                    curvature = self._calculate_curvature_at_distance(boundary, distance, sample_distance)
                    
                    if curvature > curvature_threshold:
                        point = boundary.interpolate(distance)
                        all_curvature_points.append({
                            'point': (point.x, point.y),
                            'curvature': curvature
                        })
            
            # 전체 커브점에 대해 군집화
            if len(all_curvature_points) < 2:
                final_curves = [cp['point'] for cp in all_curvature_points]
            else:
                points = np.array([cp['point'] for cp in all_curvature_points])
                clustering = DBSCAN(eps=cluster_radius, min_samples=2)
                labels = clustering.fit_predict(points)
                
                # 군집별 중심점 계산
                final_curves = []
                unique_labels = set(labels)
                
                for label in unique_labels:
                    if label == -1:  # 노이즈 제외
                        continue
                    
                    cluster_mask = labels == label
                    cluster_points = points[cluster_mask]
                    cluster_curvatures = [all_curvature_points[i]['curvature'] 
                                        for i, mask in enumerate(cluster_mask) if mask]
                    
                    # 곡률 가중 평균으로 중심점 계산
                    weights = np.array(cluster_curvatures)
                    center_x = np.average(cluster_points[:, 0], weights=weights)
                    center_y = np.average(cluster_points[:, 1], weights=weights)
                    final_curves.append((center_x, center_y))
            
            # 커브점을 가장 가까운 스켈레톤 점으로 이동
            corrected_curves = []
            for curve_point in final_curves:
                closest_skeleton_point = self._find_closest_skeleton_point(curve_point, skeleton)
                if closest_skeleton_point:
                    corrected_curves.append(closest_skeleton_point)
            
            logger.info(f"도로 경계선 기반 커브점 검출: {len(corrected_curves)}개")
            return corrected_curves
            
        except Exception as e:
            logger.error(f"경계선 기반 커브점 검출 실패: {e}")
            return []
    
    def _calculate_curvature_at_distance(self, boundary, distance: float, window: float = 20.0) -> float:
        """특정 거리에서의 곡률 계산"""
        try:
            # 앞뒤 점들 구하기
            d1 = max(0, distance - window)
            d2 = min(boundary.length, distance + window)
            
            if d2 - d1 < window * 0.5:
                return 0.0
            
            p1 = boundary.interpolate(d1)
            p2 = boundary.interpolate(distance)
            p3 = boundary.interpolate(d2)
            
            # 벡터 계산
            v1 = np.array([p2.x - p1.x, p2.y - p1.y])
            v2 = np.array([p3.x - p2.x, p3.y - p2.y])
            
            # 각도 변화 계산
            len1 = np.linalg.norm(v1)
            len2 = np.linalg.norm(v2)
            
            if len1 > 0 and len2 > 0:
                v1_norm = v1 / len1
                v2_norm = v2 / len2
                
                cos_angle = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
                angle = np.arccos(cos_angle)
                
                return angle
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def _find_closest_skeleton_point(self, curve_point: Tuple[float, float], 
                                   skeleton: List[List[float]]) -> Optional[Tuple[float, float]]:
        """커브점에서 가장 가까운 스켈레톤 점 찾기"""
        if not skeleton:
            return None
        
        min_dist = float('inf')
        closest_point = None
        
        for skel_point in skeleton:
            if len(skel_point) < 2:
                continue
            
            dist = np.sqrt((curve_point[0] - skel_point[0])**2 + 
                          (curve_point[1] - skel_point[1])**2)
            
            if dist < min_dist:
                min_dist = dist
                closest_point = (float(skel_point[0]), float(skel_point[1]))
        
        return closest_point
    
    def remove_curves_near_intersections(self, curves: List[Tuple[float, float]], 
                                       intersections: List[Tuple[float, float]], 
                                       threshold: float = 10.0) -> List[Tuple[float, float]]:
        """교차점 근처 커브점 제거"""
        if not curves or not intersections:
            return curves
        
        filtered_curves = []
        
        for curve in curves:
            near_intersection = False
            
            for intersection in intersections:
                if len(intersection) < 2:
                    continue
                
                dist = np.sqrt((curve[0] - intersection[0])**2 + 
                              (curve[1] - intersection[1])**2)
                
                if dist <= threshold:
                    near_intersection = True
                    break
            
            if not near_intersection:
                filtered_curves.append(curve)
        
        logger.info(f"교차점 근처 커브점 제거: {len(curves)} → {len(filtered_curves)}개")
        return filtered_curves
    
    def create_temporary_file(self, gdf: gpd.GeoDataFrame) -> str:
        """임시 파일 생성"""
        try:
            # 임시 디렉토리 생성
            tmp_dir = tempfile.mkdtemp()
            # GPKG 형식으로 임시 파일 저장 (여러 geometry 타입 지원)
            temp_path = os.path.join(tmp_dir, "temp_road.gpkg")
            
            # GeometryCollection 처리
            from shapely.geometry import GeometryCollection
            processed_gdf = gdf.copy()
            
            # GeometryCollection을 개별 geometry로 분해
            if any(processed_gdf.geometry.geom_type == 'GeometryCollection'):
                new_rows = []
                for idx, row in processed_gdf.iterrows():
                    if isinstance(row.geometry, GeometryCollection):
                        for geom in row.geometry.geoms:
                            new_row = row.copy()
                            new_row['geometry'] = geom
                            new_rows.append(new_row)
                    else:
                        new_rows.append(row)
                processed_gdf = gpd.GeoDataFrame(new_rows, crs=gdf.crs)
            
            # GeoDataFrame을 파일로 저장
            processed_gdf.to_file(temp_path, driver='GPKG')
            
            return temp_path
            
        except Exception as e:
            logger.error(f"임시 파일 생성 오류: {e}")
            raise 