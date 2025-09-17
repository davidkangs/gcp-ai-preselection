# src/core/district_road_clipper.py
"""지구계-도로망 자동 클리핑 모듈"""

import re
import gc
import logging
import sys
import numpy as np
import geopandas as gpd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from shapely.geometry import Polygon, MultiPolygon
from shapely.validation import make_valid
from shapely import errors as shapely_errors

logger = logging.getLogger(__name__)

def get_executable_dir():
    """실행 파일이 위치한 디렉토리 반환 (PyInstaller 호환)"""
    if getattr(sys, 'frozen', False):
        # PyInstaller로 빌드된 실행파일인 경우
        return Path(sys.executable).parent
    else:
        # 개발 환경에서 실행하는 경우
        return Path(__file__).parent.parent.parent


class DistrictRoadClipper:
    """지구계 기반 도로망 자동 클리핑 클래스"""
    
    def __init__(self, road_base_path: str = None):
        # 실행파일 기준으로 도로망 경로 설정
        if road_base_path is None:
            executable_dir = get_executable_dir()
            self.road_base_path = executable_dir / "road_by_sigungu"
        else:
            self.road_base_path = Path(road_base_path)
        
        # 좌표계 매핑
        self.CRS_MAPPING = {
            # 중부원점 (EPSG:5186)
            '서울특별시': 'EPSG:5186',
            '인천광역시': 'EPSG:5186',
            '대전광역시': 'EPSG:5186',
            '광주광역시': 'EPSG:5186',
            '세종특별자치시': 'EPSG:5186',
            '경기도': 'EPSG:5186',
            '충청남도': 'EPSG:5186',
            '충청북도': 'EPSG:5186',
            '전라남도': 'EPSG:5186',
            '전북특별자치도': 'EPSG:5186',
            '제주특별자치도': 'EPSG:5186',
            
            # 동부원점 (EPSG:5187)
            '부산광역시': 'EPSG:5187',
            '대구광역시': 'EPSG:5187',
            '울산광역시': 'EPSG:5187',
            '경상남도': 'EPSG:5187',
            '경상북도': 'EPSG:5187',
        }
        
        # 강원도 동부/서부 구분
        self.GANGWON_EAST_SIGUNGU = {
            '속초시', '고성군', '양양군', '강릉시', '동해시', '삼척시', '태백시'
        }
        
        # 울릉도
        self.ULLEUNG_SIGUNGU = {'울릉군'}
        
        # 시도명 변형 매핑 (파일명에서 찾을 수 있는 다양한 형태)
        self.SIDO_VARIANTS = {
            '서울': '서울특별시',
            '부산': '부산광역시',
            '대구': '대구광역시',
            '인천': '인천광역시',
            '광주': '광주광역시',
            '대전': '대전광역시',
            '울산': '울산광역시',
            '세종': '세종특별자치시',
            '경기': '경기도',
            '강원': '강원특별자치도',
            '충북': '충청북도',
            '충남': '충청남도',
            '전북': '전북특별자치도',
            '전남': '전라남도',
            '경북': '경상북도',
            '경남': '경상남도',
            '제주': '제주특별자치도'
        }
        
        # 도로망 인덱스 구축
        self.road_index = self._build_road_index()
        
    def _build_road_index(self) -> Dict[str, Dict[str, str]]:
        """도로망 파일 인덱스 구축"""
        road_index = {}
        
        if not self.road_base_path.exists():
            logger.warning(f"도로망 경로가 존재하지 않습니다: {self.road_base_path}")
            return road_index
            
        for sido_folder in self.road_base_path.iterdir():
            if sido_folder.is_dir():
                sido_name = sido_folder.name
                road_index[sido_name] = {}
                
                for sigungu_folder in sido_folder.iterdir():
                    if sigungu_folder.is_dir():
                        sigungu_name = sigungu_folder.name
                        road_file = sigungu_folder / f"{sigungu_name}_도로망.shp"
                        
                        if road_file.exists():
                            road_index[sido_name][sigungu_name] = str(road_file)
        
        logger.info(f"도로망 인덱스 구축 완료: {len(road_index)}개 시도")
        return road_index
    
    def validate_and_fix_geometry(self, geometry):
        """geometry 유효성 검사 및 수정"""
        try:
            if geometry is None or geometry.is_empty:
                logger.warning("빈 geometry 발견")
                return None
            
            # 좌표 개수 확인
            if hasattr(geometry, 'exterior'):
                coords = list(geometry.exterior.coords)
                if len(coords) < 4:
                    logger.warning(f"좌표 개수 부족: {len(coords)}개 (최소 4개 필요)")
                    return None
            
            # geometry 유효성 검사
            if not geometry.is_valid:
                logger.warning("유효하지 않은 geometry 발견, 수정 시도")
                geometry = make_valid(geometry)
                
                # 수정 후에도 유효하지 않으면 None 반환
                if not geometry.is_valid:
                    logger.error("geometry 수정 실패")
                    return None
            
            # 면적이 너무 작으면 제외
            if hasattr(geometry, 'area') and geometry.area < 1e-10:
                logger.warning("geometry 면적이 너무 작음")
                return None
            
            return geometry
            
        except Exception as e:
            logger.error(f"geometry 유효성 검사 중 오류: {e}")
            return None
    
    def split_multipolygon(self, district_gdf: gpd.GeoDataFrame) -> List[gpd.GeoDataFrame]:
        """멀티폴리곤을 개별 폴리곤으로 분리 (개선된 버전)"""
        split_gdfs = []
        
        for idx, row in district_gdf.iterrows():
            geom = row.geometry
            
            # geometry 유효성 검사 및 수정
            geom = self.validate_and_fix_geometry(geom)
            if geom is None:
                logger.warning(f"인덱스 {idx}의 geometry를 건너뜁니다")
                continue
            
            try:
                if isinstance(geom, MultiPolygon):
                    # 멀티폴리곤을 개별 폴리곤으로 분리
                    for i, poly in enumerate(geom.geoms):
                        # 각 폴리곤도 유효성 검사
                        poly = self.validate_and_fix_geometry(poly)
                        if poly is None:
                            continue
                        
                        new_gdf = gpd.GeoDataFrame(
                            [row.to_dict()], 
                            geometry=[poly],
                            crs=district_gdf.crs
                        )
                        new_gdf['polygon_index'] = i + 1
                        split_gdfs.append(new_gdf)
                        
                elif isinstance(geom, Polygon):
                    # 단일 폴리곤
                    new_gdf = gpd.GeoDataFrame(
                        [row.to_dict()],
                        geometry=[geom],
                        crs=district_gdf.crs
                    )
                    new_gdf['polygon_index'] = 1
                    split_gdfs.append(new_gdf)
                else:
                    logger.warning(f"지원하지 않는 geometry 타입: {type(geom)}")
                    
            except Exception as e:
                logger.error(f"폴리곤 분리 중 오류 (인덱스 {idx}): {e}")
                continue
        
        logger.info(f"총 {len(split_gdfs)}개의 유효한 폴리곤으로 분리됨")
        return split_gdfs
    
    def extract_location_info(self, filename: str) -> Tuple[Optional[str], Optional[str]]:
        """파일명에서 시도, 시군구 정보 추출"""
        basename = Path(filename).stem
        
        # 특수문자 제거
        clean_name = basename.replace("'", "").replace("`", "").replace("_", " ")
        
        # 1. 시도명 찾기
        found_sido = None
        for variant, full_sido in self.SIDO_VARIANTS.items():
            if variant in clean_name:
                found_sido = full_sido
                break
        
        # 2. 시군구 찾기
        patterns = [
            r'([가-힣]+시)(?!도)',  # ~시 (시도 제외)
            r'([가-힣]+군)',        # ~군
            r'([가-힣]+구)',        # ~구
        ]
        
        found_sigungu = []
        for pattern in patterns:
            matches = re.findall(pattern, clean_name)
            found_sigungu.extend(matches)
        
        # 중복 제거하면서 순서 유지
        unique_sigungu = []
        for name in found_sigungu:
            if name not in unique_sigungu:
                unique_sigungu.append(name)
        
        # 3. 시도가 있으면 해당 시도 내에서만 검색
        if found_sido and unique_sigungu:
            if found_sido in self.road_index:
                for sigungu in unique_sigungu:
                    if sigungu in self.road_index[found_sido]:
                        return found_sido, sigungu
        
        # 4. 시도가 없으면 전체에서 검색
        if unique_sigungu:
            for sido, sigungu_dict in self.road_index.items():
                for sigungu in unique_sigungu:
                    if sigungu in sigungu_dict:
                        return sido, sigungu
        
        return None, None
    
    def find_road_file(self, district_file: str) -> Optional[str]:
        """지구계 파일명에서 도로망 파일 찾기"""
        sido, sigungu = self.extract_location_info(district_file)
        
        if sido and sigungu:
            road_file = self.road_index.get(sido, {}).get(sigungu)
            if road_file:
                logger.info(f"도로망 매칭 성공: {sido}/{sigungu}")
                return road_file
        
        logger.warning(f"도로망 매칭 실패: {Path(district_file).name}")
        return None
    
    def get_optimal_crs(self, sido: str, sigungu: str) -> str:
        """최적 좌표계 결정"""
        # 울릉도
        if sigungu in self.ULLEUNG_SIGUNGU:
            return 'EPSG:5188'
        
        # 강원도 특별 처리
        if sido == '강원특별자치도':
            if sigungu in self.GANGWON_EAST_SIGUNGU:
                return 'EPSG:5187'
            else:
                return 'EPSG:5186'
        
        # 기본 매핑
        return self.CRS_MAPPING.get(sido, 'EPSG:5186')
    
    def safe_buffer(self, geometry, buffer_size: float):
        """안전한 버퍼 생성"""
        try:
            # geometry 유효성 먼저 확인
            geometry = self.validate_and_fix_geometry(geometry)
            if geometry is None:
                return None
            
            # 버퍼 생성
            buffered = geometry.buffer(buffer_size)
            
            # 버퍼 결과 유효성 검사
            buffered = self.validate_and_fix_geometry(buffered)
            return buffered
            
        except Exception as e:
            logger.error(f"버퍼 생성 실패: {e}")
            return None
    
    def safe_clip_roads(self, road_gdf: gpd.GeoDataFrame, clip_boundary: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """안전한 도로망 클리핑 (다단계 fallback 방식)"""
        try:
            # 방법 1: 표준 gpd.clip 시도
            logger.info("방법 1: 표준 클리핑 시도")
            try:
                clipped = gpd.clip(road_gdf, clip_boundary)
                if not clipped.empty:
                    logger.info(f"표준 클리핑 성공: {len(clipped)}개 세그먼트")
                    return clipped
            except Exception as e:
                logger.warning(f"표준 클리핑 실패: {e}")
            
            # 방법 2: geometry 수정 후 재시도
            logger.info("방법 2: geometry 수정 후 클리핑 시도")
            try:
                # 클리핑 경계 수정
                fixed_boundary = clip_boundary.copy()
                fixed_geometries = []
                for geom in fixed_boundary.geometry:
                    fixed_geom = self.validate_and_fix_geometry(geom)
                    if fixed_geom is not None:
                        fixed_geometries.append(fixed_geom)
                
                if fixed_geometries:
                    fixed_boundary = gpd.GeoDataFrame(geometry=fixed_geometries, crs=clip_boundary.crs)
                    clipped = gpd.clip(road_gdf, fixed_boundary)
                    if not clipped.empty:
                        logger.info(f"수정된 geometry 클리핑 성공: {len(clipped)}개 세그먼트")
                        return clipped
            except Exception as e:
                logger.warning(f"수정된 geometry 클리핑 실패: {e}")
            
            # 방법 3: 교집합 기반 클리핑
            logger.info("방법 3: 교집합 기반 클리핑 시도")
            try:
                clipped_roads = []
                clip_geom = clip_boundary.geometry.iloc[0] if not clip_boundary.empty else None
                
                if clip_geom is not None:
                    clip_geom = self.validate_and_fix_geometry(clip_geom)
                    
                    for idx, road_row in road_gdf.iterrows():
                        road_geom = self.validate_and_fix_geometry(road_row.geometry)
                        if road_geom is None:
                            continue
                            
                        try:
                            # 교집합 확인
                            if road_geom.intersects(clip_geom):
                                # 교집합 계산
                                intersection = road_geom.intersection(clip_geom)
                                intersection = self.validate_and_fix_geometry(intersection)
                                
                                if intersection is not None and not intersection.is_empty:
                                    # 새 행 생성
                                    new_row = road_row.copy()
                                    new_row.geometry = intersection
                                    clipped_roads.append(new_row)
                        except Exception as road_e:
                            logger.debug(f"도로 {idx} 교집합 계산 실패: {road_e}")
                            continue
                
                if clipped_roads:
                    clipped = gpd.GeoDataFrame(clipped_roads, crs=road_gdf.crs)
                    logger.info(f"교집합 기반 클리핑 성공: {len(clipped)}개 세그먼트")
                    return clipped
                    
            except Exception as e:
                logger.warning(f"교집합 기반 클리핑 실패: {e}")
            
            # 방법 4: 단순 공간 필터링 (contains/within)
            logger.info("방법 4: 공간 필터링 시도")
            try:
                clip_geom = clip_boundary.geometry.iloc[0] if not clip_boundary.empty else None
                if clip_geom is not None:
                    clip_geom = self.validate_and_fix_geometry(clip_geom)
                    
                    # 경계 내부 또는 교차하는 도로 찾기
                    mask = road_gdf.geometry.apply(lambda x: self._safe_spatial_check(x, clip_geom))
                    filtered_roads = road_gdf[mask].copy()
                    
                    if not filtered_roads.empty:
                        logger.info(f"공간 필터링 성공: {len(filtered_roads)}개 세그먼트")
                        return filtered_roads
                        
            except Exception as e:
                logger.warning(f"공간 필터링 실패: {e}")
            
            # 모든 방법 실패
            logger.error("모든 클리핑 방법 실패")
            return gpd.GeoDataFrame(columns=road_gdf.columns, crs=road_gdf.crs)
            
        except Exception as e:
            logger.error(f"안전한 클리핑 전체 실패: {e}")
            return gpd.GeoDataFrame(columns=road_gdf.columns, crs=road_gdf.crs)
    
    def _safe_spatial_check(self, road_geom, clip_geom):
        """안전한 공간 관계 확인"""
        try:
            road_geom = self.validate_and_fix_geometry(road_geom)
            if road_geom is None:
                return False
            
            # 교집합 또는 포함 관계 확인
            return road_geom.intersects(clip_geom) or clip_geom.contains(road_geom)
        except Exception:
            return False
    
    def clip_road_network(
        self, 
        district_gdf: gpd.GeoDataFrame, 
        road_file: str,
        target_crs: str = 'EPSG:5186',
        buffer_size: float = 10.0
    ) -> Optional[gpd.GeoDataFrame]:
        """도로망 클리핑 (개선된 버전)"""
        try:
            # 도로망 로드
            road_gdf = gpd.read_file(road_file)
            if road_gdf.empty:
                logger.warning("빈 도로망 파일")
                return None
            
            # 지구계 geometry 유효성 검사 및 수정
            logger.info("지구계 geometry 유효성 검사 중...")
            valid_geometries = []
            for idx, row in district_gdf.iterrows():
                geom = self.validate_and_fix_geometry(row.geometry)
                if geom is not None:
                    valid_geometries.append(geom)
                else:
                    logger.warning(f"인덱스 {idx}의 지구계 geometry를 건너뜁니다")
            
            if not valid_geometries:
                logger.error("유효한 지구계 geometry가 없습니다")
                return None
            
            # 유효한 geometry만으로 새 GeoDataFrame 생성
            district_gdf = gpd.GeoDataFrame(
                geometry=valid_geometries,
                crs=district_gdf.crs
            )
                     
            # CRS 설정 및 변환
            if road_gdf.crs is None:
                logger.info(f"도로망 좌표계 설정: {target_crs}")
                road_gdf.set_crs(target_crs, inplace=True, allow_override=True)
            elif str(road_gdf.crs) != target_crs:
                logger.info(f"도로망 좌표계 변환: {road_gdf.crs} → {target_crs}")
                road_gdf = road_gdf.to_crs(target_crs)
            
            if district_gdf.crs is None:
                logger.info(f"지구계 좌표계 설정: {target_crs}")
                district_gdf.set_crs(target_crs, inplace=True, allow_override=True)
            elif str(district_gdf.crs) != target_crs:
                logger.info(f"지구계 좌표계 변환: {district_gdf.crs} → {target_crs}")
                district_gdf = district_gdf.to_crs(target_crs)
            
            # 안전한 버퍼 적용
            logger.info(f"버퍼 {buffer_size}m 적용 중...")
            buffered_geometries = []
            for geom in district_gdf.geometry:
                buffered = self.safe_buffer(geom, buffer_size)
                if buffered is not None:
                    buffered_geometries.append(buffered)
            
            if not buffered_geometries:
                logger.error("버퍼 생성 실패")
                return None
            
            # 유니온 생성
            try:
                from shapely.ops import unary_union
                union_geom = unary_union(buffered_geometries)
                union_geom = self.validate_and_fix_geometry(union_geom)
                
                if union_geom is None:
                    logger.error("유니온 생성 실패")
                    return None
                    
            except Exception as e:
                logger.error(f"유니온 생성 중 오류: {e}")
                return None
            
            clip_boundary = gpd.GeoDataFrame(geometry=[union_geom], crs=target_crs)
            
            # 안전한 클리핑 수행 (다단계 fallback)
            logger.info("도로망 클리핑 중...")
            clipped = self.safe_clip_roads(road_gdf, clip_boundary)
            
            # 결과가 없으면 더 큰 버퍼로 재시도
            if clipped is not None and clipped.empty and buffer_size < 100:
                logger.info(f"버퍼 {buffer_size}m로 결과 없음, 100m로 재시도")
                return self.clip_road_network(district_gdf, road_file, target_crs, 100.0)
            elif clipped is None:
                logger.warning("클리핑 결과가 None입니다")
                return None
            
            if not clipped.empty:
                # 작은 조각 제거 (면적 10㎡ 미만)
                try:
                    areas = clipped.geometry.area
                    clipped = clipped[areas >= 10].copy()
                    logger.info(f"클리핑 성공: {len(clipped)}개 도로 세그먼트")
                except Exception as e:
                    logger.warning(f"면적 필터링 중 오류: {e}")
            
            return clipped if not clipped.empty else None
            
        except Exception as e:
            logger.error(f"클리핑 중 오류: {str(e)}")
            return None
        finally:
            gc.collect()
    
    def process_district_file(
        self, 
        district_file: str,
        target_crs: Optional[str] = None,
        auto_find_road: bool = True
    ) -> Dict[str, any]:
        """지구계 파일 처리 - 멀티폴리곤 분리 및 도로망 클리핑 (개선된 버전)"""
        results = {
            'success': False,
            'polygons': [],
            'total_polygons': 0,
            'sido': None,
            'sigungu': None,
            'road_file': None,
            'error': None
        }
        
        try:
            # 지구계 파일 로드
            logger.info(f"지구계 파일 로드 중: {district_file}")
            district_gdf = gpd.read_file(district_file)
            if district_gdf.empty:
                results['error'] = "빈 지구계 파일"
                return results
            
            # 멀티폴리곤 분리 (개선된 버전 사용)
            logger.info("멀티폴리곤 분리 중...")
            split_gdfs = self.split_multipolygon(district_gdf)
            
            if not split_gdfs:
                results['error'] = "유효한 폴리곤이 없습니다"
                return results
                
            results['total_polygons'] = len(split_gdfs)
            
            # 도로망 파일 찾기
            if auto_find_road:
                road_file = self.find_road_file(district_file)
                if road_file:
                    sido, sigungu = self.extract_location_info(district_file)
                    results['sido'] = sido
                    results['sigungu'] = sigungu
                    results['road_file'] = road_file
                    
                    # 좌표계 결정 - target_crs가 명시적으로 지정되지 않은 경우만 자동 결정
                    if not target_crs and sido and sigungu:
                        target_crs = self.get_optimal_crs(sido, sigungu)
                    # target_crs가 지정된 경우 그대로 사용
                else:
                    results['error'] = "도로망 파일을 찾을 수 없음"
                    results['polygons'] = split_gdfs  # 폴리곤은 반환
                    return results
            
            # 각 폴리곤에 대한 정보 저장
            for i, poly_gdf in enumerate(split_gdfs):
                # 좌표계 변환 (target_crs가 지정된 경우)
                if target_crs and poly_gdf.crs and str(poly_gdf.crs) != target_crs:
                    logger.info(f"폴리곤 {i + 1} 좌표계 변환: {poly_gdf.crs} → {target_crs}")
                    poly_gdf = poly_gdf.to_crs(target_crs)
                elif target_crs and poly_gdf.crs is None:
                    logger.info(f"폴리곤 {i + 1} 좌표계 설정: {target_crs}")
                    poly_gdf = poly_gdf.set_crs(target_crs, allow_override=True)
                
                # GeoDataFrame에서 첫 번째 geometry 추출
                # (split_multipolygon은 각 폴리곤을 별도 GeoDataFrame로 반환)
                polygon_geom = poly_gdf.geometry.iloc[0] if not poly_gdf.empty else None
                
                polygon_info = {
                    'index': i + 1,
                    'geometry': polygon_geom,  # 단일 geometry 저장
                    'geometry_gdf': poly_gdf,   # GeoDataFrame도 별도 저장 (클리핑용)
                    'clipped_road': None
                }
                
                # 도로망 클리핑 (도로망이 있는 경우만)
                if road_file and target_crs:
                    logger.info(f"폴리곤 {i + 1}/{len(split_gdfs)} 도로망 클리핑 중... (좌표계: {target_crs})")
                    clipped = self.clip_road_network(
                        poly_gdf,  # GeoDataFrame 전달
                        road_file,
                        target_crs
                    )
                    polygon_info['clipped_road'] = clipped
                
                results['polygons'].append(polygon_info)
            
            results['success'] = True
            results['target_crs'] = target_crs
            
        except Exception as e:
            results['error'] = str(e)
            logger.error(f"지구계 처리 오류: {str(e)}")
        
        return results
    
    def clip_with_manual_road(
        self, 
        district_gdf: gpd.GeoDataFrame,
        road_folder: str,
        target_crs: str = 'EPSG:5186'
    ) -> Optional[gpd.GeoDataFrame]:
        """수동 선택한 폴더의 도로망으로 클리핑"""
        road_folder_path = Path(road_folder)
        
        # 도로망 파일 찾기
        road_files = list(road_folder_path.glob("*_도로망.shp"))
        if not road_files:
            logger.error(f"도로망 파일을 찾을 수 없음: {road_folder}")
            return None
        
        road_file = str(road_files[0])
        return self.clip_road_network(district_gdf, road_file, target_crs)