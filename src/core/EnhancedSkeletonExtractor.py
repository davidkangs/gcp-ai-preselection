"""
향상된 스켈레톤 추출기
- NetworkX 기반 교차점 검출 통합
- 면적 필터링 지원
"""

import numpy as np
import geopandas as gpd
from shapely.ops import unary_union
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.validation import make_valid
from rasterio.features import rasterize
from rasterio.transform import from_origin
from skimage.morphology import skeletonize
import networkx as nx
import time
from typing import Tuple, List
import gc
import logging

logger = logging.getLogger(__name__)


class EnhancedSkeletonExtractor:
    """향상된 도로 스켈레톤 추출 클래스"""
    
    def __init__(self, width=1200, resolution=1.0):
        self.width = width
        self.resolution = resolution
        self.area_percentile = 5  # 면적 하위 퍼센타일
        
    def process_shapefile(self, file_path: str) -> Tuple[List[List[float]], List[Tuple[float, float]]]:
        """
        Shapefile 처리하여 스켈레톤과 교차점 추출
        
        Returns:
            tuple: (skeleton_points, intersections)
        """
        logger.info(f"향상된 도로망 분석 시작: {file_path}")
        start_time = time.time()
        
        # 1. Shapefile 로딩
        gdf = gpd.read_file(file_path)
        
        # CRS 확인 및 설정
        if gdf.crs is None:
            logger.warning("CRS가 없습니다. EPSG:5186으로 가정합니다.")
            gdf.set_crs('EPSG:5186', inplace=True)
        
        # 2. 폴리곤 추출 및 면적 필터링
        polygons = self._extract_and_filter_polygons(gdf)
        
        # 3. 스켈레톤 추출
        skeleton_data = self._extract_skeleton_from_polygons(polygons)
        
        # 4. NetworkX 기반 교차점 검출
        intersections = self._detect_intersections_networkx(
            skeleton_data['skeleton'],
            skeleton_data['transform']
        )
        
        # 5. 결과 정리
        skeleton_points = skeleton_data['skeleton_points']
        
        logger.info(f"처리 완료: {time.time() - start_time:.1f}초")
        logger.info(f"스켈레톤 포인트: {len(skeleton_points)}개")
        logger.info(f"검출된 교차점: {len(intersections)}개")
        
        return skeleton_points, intersections
    
    def _extract_and_filter_polygons(self, gdf):
        """폴리곤 추출 및 면적 기반 필터링"""
        polygons = []
        
        for _, row in gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            
            # 유효성 검사
            if not geom.is_valid:
                geom = make_valid(geom)
            
            # 폴리곤 추출
            if geom.geom_type == 'Polygon':
                polygons.append(geom)
            elif geom.geom_type == 'MultiPolygon':
                polygons.extend(list(geom.geoms))
            elif geom.geom_type in ['LineString', 'MultiLineString']:
                # LineString은 버퍼 적용
                buffered = geom.buffer(self.resolution * 2)
                if buffered.geom_type == 'Polygon':
                    polygons.append(buffered)
                elif buffered.geom_type == 'MultiPolygon':
                    polygons.extend(list(buffered.geoms))
        
        if not polygons:
            return polygons
        
        # 면적 편차 계산
        areas = np.array([poly.area for poly in polygons])
        std_dev = np.std(areas)
        mean_area = np.mean(areas)
        
        # 편차가 평균의 50% 이상이면 필터링
        if std_dev >= 0.5 * mean_area:
            threshold_area = np.percentile(areas, self.area_percentile)
            filtered = [poly for poly in polygons if poly.area >= threshold_area]
            logger.info(f"면적 필터링: {len(polygons)} → {len(filtered)} (하위 {self.area_percentile}% 제거)")
            return filtered
        else:
            logger.info(f"편차가 작아 필터링 생략 ({len(polygons)}개 유지)")
            return polygons
    
    def _extract_skeleton_from_polygons(self, polygons):
        """폴리곤에서 스켈레톤 추출"""
        if not polygons:
            return {
                'skeleton': None,
                'skeleton_points': [],
                'transform': None
            }
        
        # 전체 경계 계산
        all_geoms = unary_union(polygons)
        bounds = all_geoms.bounds
        minx, miny, maxx, maxy = bounds
        
        # 픽셀 크기 계산
        pixel_size = (maxx - minx) / self.width
        height = int((maxy - miny) / pixel_size)
        
        # 최소 크기 보장
        if height < 10:
            height = 10
            pixel_size = (maxy - miny) / height
            self.width = int((maxx - minx) / pixel_size)
        
        transform = from_origin(minx, maxy, pixel_size, pixel_size)
        
        # 래스터화
        shapes = [(geom, 1) for geom in polygons if geom.is_valid]
        mask = rasterize(
            shapes,
            out_shape=(height, self.width),
            transform=transform,
            fill=0,
            dtype=np.uint8
        )
        
        # 스켈레톤화
        skeleton = skeletonize(mask > 0)
        
        # 스켈레톤 포인트 추출
        rows, cols = np.where(skeleton)
        skeleton_points = []
        
        for row, col in zip(rows, cols):
            x = minx + (col + 0.5) * pixel_size
            y = maxy - (row + 0.5) * pixel_size
            skeleton_points.append([float(x), float(y)])
        
        return {
            'skeleton': skeleton,
            'skeleton_points': skeleton_points,
            'transform': transform,
            'pixel_size': pixel_size,
            'bounds': (minx, miny, maxx, maxy)
        }
    
    def _detect_intersections_networkx(self, skeleton, transform):
        """NetworkX를 사용한 교차점 검출"""
        if skeleton is None:
            return []
        
        # 스켈레톤을 그래프로 변환
        G = self._skeleton_to_graph(skeleton)
        
        # 차수가 3 이상인 노드 찾기
        intersection_nodes = [node for node, degree in G.degree() if degree >= 3]
        
        # 픽셀 좌표를 실제 좌표로 변환
        intersections = []
        for node in intersection_nodes:
            x, y = transform * node
            intersections.append((float(x), float(y)))
        
        return intersections
    
    def _skeleton_to_graph(self, skeleton):
        """스켈레톤을 NetworkX 그래프로 변환"""
        G = nx.Graph()
        rows, cols = np.where(skeleton)
        height, width = skeleton.shape
        
        # 노드 추가
        for y, x in zip(rows, cols):
            G.add_node((x, y))
        
        # 8방향 연결
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), 
                     (0, 1), (1, -1), (1, 0), (1, 1)]
        
        for y, x in zip(rows, cols):
            for dy, dx in directions:
                ny, nx = y + dy, x + dx
                if 0 <= nx < width and 0 <= ny < height and skeleton[ny, nx]:
                    G.add_edge((x, y), (nx, ny))
        
        return G
    
    def get_skeleton_as_linestring(self, skeleton_points):
        """스켈레톤 포인트를 LineString으로 변환"""
        from shapely.geometry import LineString, MultiLineString
        
        if not skeleton_points:
            return None
        
        # 간단한 연결 (순서대로)
        if len(skeleton_points) >= 2:
            return LineString(skeleton_points)
        else:
            return None


# 기존 시스템과의 호환성을 위한 별칭
SkeletonExtractor = EnhancedSkeletonExtractor