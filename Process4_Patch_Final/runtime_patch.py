#!/usr/bin/env python3
"""
프로세스4 런타임 패치
실행 시점에 메모리에서 함수를 교체하여 오류 해결
"""

import sys
import types
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def patch_district_road_clipper():
    """district_road_clipper 모듈 패치"""
    try:
        # 모듈 import
        from src.core.district_road_clipper import DistrictRoadClipper
        import geopandas as gpd
        from shapely.validation import make_valid
        
        # 안전한 클리핑 함수 정의
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
        
        # 기존 clip_road_network 함수 패치
        def patched_clip_road_network(self, district_gdf, road_file, target_crs='EPSG:5186', buffer_size=10.0):
            """패치된 도로망 클리핑 함수"""
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
                
                # 안전한 클리핑 수행
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
                import gc
                gc.collect()
        
        # 함수 교체
        DistrictRoadClipper.safe_clip_roads = safe_clip_roads
        DistrictRoadClipper._safe_spatial_check = _safe_spatial_check
        DistrictRoadClipper.clip_road_network = patched_clip_road_network
        
        logger.info("✅ DistrictRoadClipper 패치 적용 완료")
        return True
        
    except Exception as e:
        logger.error(f"❌ DistrictRoadClipper 패치 실패: {e}")
        return False

def patch_coordinate_conversion():
    """좌표계 변환 함수 패치"""
    try:
        # process4_inference 모듈에서 InferenceTool 클래스 찾기
        import sys
        inference_module = None
        for name, module in sys.modules.items():
            if hasattr(module, 'InferenceTool'):
                inference_module = module
                break
        
        if inference_module is None:
            logger.warning("InferenceTool 클래스를 찾을 수 없습니다")
            return False
        
        InferenceTool = inference_module.InferenceTool
        
        # 안전한 좌표계 변환 함수
        def patched_reload_with_new_crs(self):
            """패치된 좌표계 변환 함수"""
            try:
                # 안전장치: 필수 데이터 존재 확인
                if not self.current_polygon_data:
                    logger.warning("current_polygon_data가 없습니다")
                    return
                
                if 'polygons' not in self.current_polygon_data:
                    logger.warning("polygons 데이터가 없습니다")
                    return
                
                if not self.current_polygon_data['polygons']:
                    logger.warning("polygons 리스트가 비어있습니다")
                    return
                
                if self.current_polygon_index >= len(self.current_polygon_data['polygons']):
                    logger.warning(f"잘못된 polygon 인덱스: {self.current_polygon_index}")
                    return
                
                # 진행 표시
                from PyQt5.QtWidgets import QProgressDialog
                from PyQt5.QtCore import Qt
                progress = QProgressDialog("좌표계 변환 중...", "취소", 0, 100, self)
                progress.setWindowModality(Qt.WindowModal)
                progress.show()
                
                # 현재 폴리곤 데이터 가져오기
                current_polygon = self.current_polygon_data['polygons'][self.current_polygon_index]
                
                # 좌표계 변환
                target_crs = self.get_target_crs()
                if not target_crs:
                    logger.error("target_crs를 가져올 수 없습니다")
                    return
                    
                logger.info(f"좌표계 변환: {self.current_polygon_data.get('original_crs', 'Unknown')} → {target_crs}")
                
                # 원본 좌표계 정보 저장 (처음 로드 시 한 번만)
                if 'original_crs' not in self.current_polygon_data:
                    self.current_polygon_data['original_crs'] = self.current_polygon_data.get('target_crs', 'EPSG:5186')
                    # 각 폴리곤의 원본 geometry도 저장
                    for poly in self.current_polygon_data['polygons']:
                        if 'geometry_gdf' in poly and poly['geometry_gdf'] is not None:
                            poly['original_geometry_gdf'] = poly['geometry_gdf'].copy()
                        elif 'geometry' in poly:
                            poly['original_geometry'] = poly['geometry']
                        if 'clipped_road' in poly and poly['clipped_road'] is not None:
                            poly['original_clipped_road'] = poly['clipped_road'].copy()
                
                # 지구계 폴리곤 좌표계 변환 (원본에서 변환)
                if 'original_geometry_gdf' in current_polygon and current_polygon['original_geometry_gdf'] is not None:
                    try:
                        # GeoDataFrame가 있는 경우
                        poly_gdf = current_polygon['original_geometry_gdf'].copy()
                        
                        # 원본 좌표계로 설정
                        if poly_gdf.crs is None:
                            poly_gdf = poly_gdf.set_crs(self.current_polygon_data['original_crs'], allow_override=True)
                            
                        # 타겟 좌표계로 변환
                        if str(poly_gdf.crs) != target_crs:
                            logger.info(f"지구계 GeoDataFrame 변환: {poly_gdf.crs} → {target_crs}")
                            poly_gdf = poly_gdf.to_crs(target_crs)
                            
                        current_polygon['geometry_gdf'] = poly_gdf
                        # geometry도 업데이트
                        if not poly_gdf.empty:
                            current_polygon['geometry'] = poly_gdf.geometry.iloc[0]
                        
                        # 캔버스에 표시
                        geom_data = poly_gdf.geometry.iloc[0] if not poly_gdf.empty else None
                        if geom_data:
                            logger.info(f"지구계 폴리곤 표시")
                            self.canvas_widget.set_background_data(geom_data)
                    except Exception as gdf_e:
                        logger.error(f"GeoDataFrame 좌표계 변환 오류: {gdf_e}")
                        # 실패해도 계속 진행
                        
                elif 'original_geometry' in current_polygon:
                    # 단일 geometry만 있는 경우
                    try:
                        from shapely.geometry import shape, mapping
                        import pyproj
                        from shapely.ops import transform
                        
                        geom_data = current_polygon['original_geometry']
                        original_crs = self.current_polygon_data['original_crs']
                        
                        if original_crs != target_crs:
                            # 좌표계 변환
                            logger.info(f"지구계 geometry 변환: {original_crs} → {target_crs}")
                            project = pyproj.Transformer.from_crs(original_crs, target_crs, always_xy=True).transform
                            transformed_geom = transform(project, geom_data)
                            current_polygon['geometry'] = transformed_geom
                            geom_data = transformed_geom
                        else:
                            current_polygon['geometry'] = geom_data
                        
                        # 캔버스에 표시
                        logger.info(f"지구계 폴리곤 표시")
                        self.canvas_widget.set_background_data(geom_data)
                    except Exception as geom_e:
                        logger.error(f"Geometry 좌표계 변환 오류: {geom_e}")
                
                # 도로망 좌표계 변환 (원본에서 변환)
                if 'original_clipped_road' in current_polygon and current_polygon['original_clipped_road'] is not None:
                    try:
                        road_gdf = current_polygon['original_clipped_road'].copy()
                        
                        # 원본 좌표계로 설정
                        if road_gdf.crs is None:
                            road_gdf = road_gdf.set_crs(self.current_polygon_data['original_crs'], allow_override=True)
                        
                        # 좌표계 변환
                        if str(road_gdf.crs) != target_crs:
                            logger.info(f"도로망 GeoDataFrame 변환: {road_gdf.crs} → {target_crs}")
                            road_gdf = road_gdf.to_crs(target_crs)
                            
                        current_polygon['clipped_road'] = road_gdf
                        
                        # 캔버스에 표시
                        logger.info(f"도로망 표시")
                        self.canvas_widget.set_road_data(road_gdf)
                    except Exception as road_e:
                        logger.error(f"도로망 좌표계 변환 오류: {road_e}")
                        # 실패해도 계속 진행
                
                # 현재 target_crs 업데이트 (원본은 유지)
                self.current_polygon_data['target_crs'] = target_crs
                
                # 위성영상 갱신을 위해 캔버스 CRS 재설정
                self.canvas_widget.canvas.crs = target_crs
                
                progress.setValue(100)
                progress.close()
                
                # 화면 업데이트
                self.canvas_widget.canvas.update_display()
                
                logger.info(f"좌표계 변환 완료: {target_crs}")
                
            except Exception as e:
                if 'progress' in locals():
                    progress.close()
                logger.error(f"좌표계 변환 오류: {e}")
                from PyQt5.QtWidgets import QMessageBox
                QMessageBox.warning(self, "경고", f"좌표계 변환 중 오류:\n{str(e)}")
        
        # 함수 교체
        InferenceTool.reload_with_new_crs = patched_reload_with_new_crs
        
        logger.info("✅ 좌표계 변환 패치 적용 완료")
        return True
        
    except Exception as e:
        logger.error(f"❌ 좌표계 변환 패치 실패: {e}")
        return False

def apply_all_patches():
    """모든 패치 적용"""
    logger.info("🔧 프로세스4 런타임 패치 시작...")
    
    success_count = 0
    
    # 1. DistrictRoadClipper 패치
    if patch_district_road_clipper():
        success_count += 1
    
    # 2. 좌표계 변환 패치  
    if patch_coordinate_conversion():
        success_count += 1
    
    logger.info(f"🎉 패치 완료: {success_count}/2개 성공")
    return success_count == 2

# 자동 패치 적용 (import 시 실행)
if __name__ != "__main__":
    apply_all_patches()
