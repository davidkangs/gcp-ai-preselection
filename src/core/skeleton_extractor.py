# src/core/skeleton_extractor.py

"""도로 스켈레톤 추출 모듈"""
import numpy as np
import geopandas as gpd
from shapely.ops import unary_union
from rasterio.features import rasterize
from rasterio.transform import from_origin
from skimage.morphology import skeletonize
from sklearn.cluster import DBSCAN
import time
from typing import Tuple, List
import gc
import logging

# Numba 선택적 import - 없어도 작동하도록
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Numba가 없을 때 대체 함수
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range

logger = logging.getLogger(__name__)

if NUMBA_AVAILABLE:
    @jit(nopython=True, parallel=True, cache=True)
    def fast_neighbor_count(skeleton, height, width):
        """Numba JIT로 최적화된 이웃 개수 계산"""
        neighbor_count = np.zeros((height, width), dtype=np.int32)
        
        for i in prange(1, height-1):
            for j in prange(1, width-1):
                if skeleton[i, j]:
                    count = 0
                    for di in range(-1, 2):
                        for dj in range(-1, 2):
                            if di == 0 and dj == 0:
                                continue
                            if skeleton[i+di, j+dj]:
                                count += 1
                    neighbor_count[i, j] = count
        
        return neighbor_count
else:
    def fast_neighbor_count(skeleton, height, width):
        """순수 Python 버전의 이웃 개수 계산"""
        neighbor_count = np.zeros((height, width), dtype=np.int32)
        
        for i in range(1, height-1):
            for j in range(1, width-1):
                if skeleton[i, j]:
                    count = 0
                    for di in range(-1, 2):
                        for dj in range(-1, 2):
                            if di == 0 and dj == 0:
                                continue
                            if skeleton[i+di, j+dj]:
                                count += 1
                    neighbor_count[i, j] = count
        
        return neighbor_count

class SkeletonExtractor:
    """도로 스켈레톤 추출 클래스"""
    
    def __init__(self, width=1200, intersect_threshold=3, cluster_distance=5.0):
        self.width = width
        self.intersect_threshold = intersect_threshold
        self.cluster_distance = cluster_distance
        self.simplify_tolerance = 1.0  
        
    def process_shapefile(self, file_path: str, target_crs: str = 'EPSG:5186') -> Tuple[List[List[float]], List[Tuple[float, float]]]:
        """
        Shapefile 처리하여 스켈레톤과 교차점 추출
        
        Args:
            file_path: Shapefile 경로
            target_crs: 대상 좌표계 (기본값: EPSG:5186)
            
        Returns:
            tuple: (skeleton_points, intersections)
                - skeleton_points: [[x, y], ...] 형태의 스켈레톤 포인트 리스트
                - intersections: [(x, y), ...] 형태의 교차점 리스트
        """
        result = self.extract_from_shapefile(file_path, target_crs)
        
        # 스켈레톤 포인트를 좌표 리스트로 변환
        skeleton_points = self._skeleton_to_points(
            result['skeleton'], 
            result['transform']
        )
        
        # 교차점 리스트
        intersections = result['intersections']
        
        return skeleton_points, intersections
        
    def extract_from_shapefile(self, shp_path: str, target_crs: str = 'EPSG:5186') -> dict:
        """Shapefile에서 스켈레톤과 교차점 추출
        
        Args:
            shp_path: Shapefile 경로
            target_crs: 대상 좌표계 (기본값: EPSG:5186)
        """
        logger.info(f"도로망 분석 시작: {shp_path} (대상 좌표계: {target_crs})")
        start_time = time.time()
        
        # 1. Shapefile 로딩
        gdf = gpd.read_file(shp_path)
        
        # CRS 확인 및 설정
        if gdf.crs is None:
            logger.warning(f"CRS가 없습니다. {target_crs}로 가정합니다.")
            gdf.set_crs(target_crs, inplace=True)
        
        # 2. 좌표계 확인 (변환하지 않음 - 스켈레톤 추출은 좌표계와 무관)
        if gdf.crs:
            try:
                epsg_code = gdf.crs.to_epsg()
                if epsg_code:
                    current_crs = f'EPSG:{epsg_code}'
                else:
                    current_crs = str(gdf.crs)
            except:
                current_crs = str(gdf.crs)
            logger.info(f"현재 좌표계: {current_crs} (변환 없이 그대로 사용)")
        else:
            logger.info(f"좌표계 정보 없음 (그대로 진행)")
        
        # 3. Geometry 처리
        geoms = [geom for geom in gdf.geometry if geom is not None and not geom.is_empty]
        
        # 4. Union 계산
        union_poly = unary_union(geoms)
        
        # 5. 래스터화
        skeleton, transform_info = self._rasterize_polygon(union_poly)
        
        # 6. 교차점 검출
        intersections = self._find_intersections(skeleton, transform_info)
        
        # 메모리 정리
        gc.collect()
        
        logger.info(f"처리 완료: {time.time() - start_time:.1f}초")
        logger.info(f"스켈레톤 포인트 수: {np.sum(skeleton)}")
        logger.info(f"검출된 교차점: {len(intersections)}개")
        
        return {
            'skeleton': skeleton,
            'intersections': intersections,
            'transform': transform_info,
            'union_poly': union_poly,
            'geoms': geoms,
            'gdf': gdf  # 원본 GeoDataFrame 추가
        }
    
    def _rasterize_polygon(self, union_poly) -> Tuple[np.ndarray, dict]:
        """폴리곤을 래스터화하고 스켈레톤 추출"""
        minx, miny, maxx, maxy = union_poly.bounds
        
        # 픽셀 크기 계산
        pixel_size = (maxx - minx) / self.width
        height = int((maxy - miny) / pixel_size)
        
        # 최소 높이 보장
        if height < 10:
            height = 10
            pixel_size = (maxy - miny) / height
            self.width = int((maxx - minx) / pixel_size)
        
        transform = from_origin(minx, maxy, pixel_size, pixel_size)
        
        # 래스터화
        if union_poly.geom_type == 'Polygon':
            shapes = [(union_poly, 1)]
        else:
            shapes = [(geom, 1) for geom in union_poly.geoms if geom.geom_type == 'Polygon']
        
        # LineString 처리
        if union_poly.geom_type == 'LineString':
            shapes = [(union_poly.buffer(pixel_size * 2), 1)]
        elif union_poly.geom_type == 'MultiLineString':
            shapes = [(line.buffer(pixel_size * 2), 1) for line in union_poly.geoms]
        
        mask = rasterize(shapes, out_shape=(height, self.width),
                        transform=transform, fill=0, dtype=np.uint8)
        
        # 스켈레톤화
        skeleton = skeletonize(mask > 0)
        
        transform_info = {
            'minx': minx,
            'maxy': maxy,
            'pixel_size': pixel_size,
            'width': self.width,
            'height': height
        }
        
        return skeleton, transform_info
    
    def _find_intersections(self, skeleton: np.ndarray, transform_info: dict) -> List[Tuple[float, float]]:
        """교차점 검출 및 클러스터링"""
        height, width = skeleton.shape
        
        # 이웃 개수 계산
        neighbor_count = fast_neighbor_count(skeleton.astype(np.int32), height, width)
        
        # 교차점 마스크
        intersect_mask = skeleton & (neighbor_count >= self.intersect_threshold)
        
        # 픽셀 좌표를 실제 좌표로 변환
        rows, cols = np.where(intersect_mask)
        if len(rows) == 0:
            return []
        
        x_coords = transform_info['minx'] + (cols + 0.5) * transform_info['pixel_size']
        y_coords = transform_info['maxy'] - (rows + 0.5) * transform_info['pixel_size']
        
        # DBSCAN 클러스터링
        coords = np.column_stack((x_coords, y_coords))
        clustering = DBSCAN(eps=self.cluster_distance, min_samples=1).fit(coords)
        
        # 클러스터 중심점 계산
        unique_labels = np.unique(clustering.labels_)
        cluster_centers = []
        
        for label in unique_labels:
            cluster_mask = clustering.labels_ == label
            cluster_points = coords[cluster_mask]
            center = cluster_points.mean(axis=0)
            cluster_centers.append(tuple(center))
        
        return cluster_centers
    
    def _skeleton_to_points(self, skeleton: np.ndarray, transform_info: dict) -> List[List[float]]:
        """
        스켈레톤 이미지를 좌표 포인트 리스트로 변환
        
        Returns:
            List[List[float]]: [[x, y], ...] 형태의 좌표 리스트
        """
        # 스켈레톤 픽셀 찾기
        rows, cols = np.where(skeleton)
        
        if len(rows) == 0:
            return []
        
        # 픽셀 좌표를 실제 좌표로 변환
        x_coords = transform_info['minx'] + (cols + 0.5) * transform_info['pixel_size']
        y_coords = transform_info['maxy'] - (rows + 0.5) * transform_info['pixel_size']
        
        # [[x, y], ...] 형태로 변환
        points = []
        for x, y in zip(x_coords, y_coords):
            points.append([float(x), float(y)])
        
        return points
    
    def get_skeleton_as_linestring(self, skeleton: np.ndarray, transform_info: dict):
        """
        스켈레톤을 LineString 객체로 변환 (옵션)
        """
        from shapely.geometry import LineString, MultiLineString
        from skimage.measure import find_contours
        
        # 연결된 성분 찾기
        contours = find_contours(skeleton.astype(float), 0.5)
        
        lines = []
        for contour in contours:
            # 픽셀 좌표를 실제 좌표로 변환
            coords = []
            for row, col in contour:
                x = transform_info['minx'] + col * transform_info['pixel_size']
                y = transform_info['maxy'] - row * transform_info['pixel_size']
                coords.append((x, y))
            
            if len(coords) >= 2:
                lines.append(LineString(coords))
        
        if len(lines) == 0:
            return None
        elif len(lines) == 1:
            return lines[0]
        else:
            return MultiLineString(lines)