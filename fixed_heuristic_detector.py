"""
Component rings 오류 해결된 휴리스틱 검출기
- geometry 유효성 검사 강화
- polygon ring 순서 수정
- 자기 교차 제거
- 안전한 좌표계 처리
"""

import numpy as np
import geopandas as gpd
from shapely.geometry import Point, LineString, MultiPoint, Polygon
from shapely.ops import unary_union
from shapely import validation, make_valid
import warnings
import logging

logger = logging.getLogger(__name__)


class EnhancedHeuristicDetector:
    """Component rings 오류 해결된 휴리스틱 검출기"""
    
    def __init__(self):
        self.min_intersection_distance = 20
        self.curve_angle_threshold = 30
        self.endpoint_distance = 30
    
    def fix_geometry_properly(self, geom):
        """geometry 제대로 수정"""
        try:
            if geom is None or geom.is_empty:
                return None
            
            # 1. 기본 유효성 검사
            if geom.is_valid:
                return geom
            
            logger.debug(f"Invalid geometry detected: {validation.explain_validity(geom)}")
            
            # 2. make_valid 사용 (Shapely 1.8+)
            try:
                fixed_geom = make_valid(geom)
                if fixed_geom.is_valid and not fixed_geom.is_empty:
                    logger.debug("Geometry fixed with make_valid")
                    return fixed_geom
            except Exception as e:
                logger.debug(f"make_valid failed: {e}")
            
            # 3. buffer(0) 트릭
            try:
                fixed_geom = geom.buffer(0)
                if fixed_geom.is_valid and not fixed_geom.is_empty:
                    logger.debug("Geometry fixed with buffer(0)")
                    return fixed_geom
            except Exception as e:
                logger.debug(f"buffer(0) failed: {e}")
            
            # 4. 단순화 시도
            try:
                tolerance = 0.1
                while tolerance <= 10.0:
                    fixed_geom = geom.simplify(tolerance)
                    if fixed_geom.is_valid and not fixed_geom.is_empty:
                        logger.debug(f"Geometry fixed with simplify({tolerance})")
                        return fixed_geom
                    tolerance *= 2
            except Exception as e:
                logger.debug(f"simplify failed: {e}")
            
            # 5. Polygon인 경우 특별 처리
            if hasattr(geom, 'geom_type') and geom.geom_type == 'Polygon':
                try:
                    # exterior만 사용
                    exterior_coords = list(geom.exterior.coords)
                    if len(exterior_coords) >= 4:  # 최소 4점 필요 (닫힌 polygon)
                        # 중복 제거
                        unique_coords = []
                        for coord in exterior_coords:
                            if not unique_coords or np.linalg.norm(np.array(coord) - np.array(unique_coords[-1])) > 0.1:
                                unique_coords.append(coord)
                        
                        # 닫혀있는지 확인
                        if len(unique_coords) >= 3:
                            if unique_coords[0] != unique_coords[-1]:
                                unique_coords.append(unique_coords[0])
                            
                            new_polygon = Polygon(unique_coords)
                            if new_polygon.is_valid:
                                logger.debug("Polygon fixed by reconstructing exterior")
                                return new_polygon
                except Exception as e:
                    logger.debug(f"Polygon reconstruction failed: {e}")
            
            logger.warning("Could not fix geometry, returning None")
            return None
            
        except Exception as e:
            logger.warning(f"Geometry fixing failed: {e}")
            return None
    
    def safe_extract_linestrings(self, geom):
        """안전한 LineString 추출"""
        linestrings = []
        
        try:
            # 먼저 geometry 수정
            fixed_geom = self.fix_geometry_properly(geom)
            if fixed_geom is None:
                return []
            
            geom_type = fixed_geom.geom_type
            
            if geom_type == 'LineString':
                if len(fixed_geom.coords) >= 2:
                    linestrings.append(fixed_geom)
                    
            elif geom_type == 'MultiLineString':
                for line in fixed_geom.geoms:
                    if line.is_valid and len(line.coords) >= 2:
                        linestrings.append(line)
                        
            elif geom_type == 'Polygon':
                # Polygon의 경계를 LineString으로
                try:
                    exterior = fixed_geom.exterior
                    if exterior.is_valid and len(exterior.coords) >= 2:
                        linestrings.append(LineString(exterior.coords))
                    
                    # interior rings도 처리
                    for interior in fixed_geom.interiors:
                        if len(interior.coords) >= 2:
                            linestrings.append(LineString(interior.coords))
                except Exception as e:
                    logger.debug(f"Polygon boundary extraction failed: {e}")
                    
            elif geom_type == 'MultiPolygon':
                for poly in fixed_geom.geoms:
                    linestrings.extend(self.safe_extract_linestrings(poly))
                    
            elif geom_type in ['Point', 'MultiPoint']:
                # Point는 LineString으로 변환 불가
                pass
                
        except Exception as e:
            logger.debug(f"LineString extraction failed: {e}")
            
        return linestrings
    
    def detect_intersections(self, gdf, skeleton):
        """개선된 교차점 검출"""
        try:
            intersections = []
            
            # 1. CRS 확인 및 변환
            target_crs = 'EPSG:5186'
            if gdf.crs != target_crs:
                try:
                    gdf = gdf.to_crs(target_crs)
                    logger.info(f"CRS 변환 완료: -> {target_crs}")
                except Exception as e:
                    logger.warning(f"CRS 변환 실패: {e}")
            
            # 2. 모든 geometry에서 LineString 추출
            all_lines = []
            
            for idx, row in gdf.iterrows():
                try:
                    lines = self.safe_extract_linestrings(row.geometry)
                    all_lines.extend(lines)
                    
                    if idx % 100 == 0:
                        logger.debug(f"Processed {idx}/{len(gdf)} features")
                        
                except Exception as e:
                    logger.debug(f"Failed to process feature {idx}: {e}")
                    continue
            
            if len(all_lines) < 2:
                logger.warning("Not enough valid lines for intersection detection")
                return []
            
            logger.info(f"Extracted {len(all_lines)} valid lines")
            
            # 3. 라인 간 교차점 검출 (안전하게)
            raw_intersections = []
            
            for i, line1 in enumerate(all_lines):
                for j, line2 in enumerate(all_lines[i+1:], i+1):
                    try:
                        # 교차점 계산
                        intersection = line1.intersection(line2)
                        
                        if intersection.is_empty:
                            continue
                        
                        # 교차점 좌표 추출
                        if hasattr(intersection, 'x') and hasattr(intersection, 'y'):
                            # Single point
                            raw_intersections.append((intersection.x, intersection.y))
                        elif hasattr(intersection, 'coords'):
                            # Point with coords
                            raw_intersections.extend(list(intersection.coords))
                        elif hasattr(intersection, 'geoms'):
                            # MultiPoint
                            for geom in intersection.geoms:
                                if hasattr(geom, 'x') and hasattr(geom, 'y'):
                                    raw_intersections.append((geom.x, geom.y))
                                elif hasattr(geom, 'coords'):
                                    raw_intersections.extend(list(geom.coords))
                        
                    except Exception as e:
                        logger.debug(f"Intersection calculation failed for lines {i}-{j}: {e}")
                        continue
            
            if not raw_intersections:
                logger.info("No intersections found")
                return []
            
            logger.info(f"Found {len(raw_intersections)} raw intersections")
            
            # 4. 스켈레톤과 매칭
            if not skeleton:
                logger.warning("No skeleton provided, using raw intersections")
                # 중복 제거만 수행
                filtered_intersections = []
                for intersection in raw_intersections:
                    is_duplicate = False
                    for existing in filtered_intersections:
                        dist = np.linalg.norm(np.array(intersection[:2]) - np.array(existing[:2]))
                        if dist < self.min_intersection_distance:
                            is_duplicate = True
                            break
                    if not is_duplicate:
                        filtered_intersections.append(tuple(intersection[:2]))
                return filtered_intersections
            
            # 스켈레톤과 매칭
            skeleton_intersections = []
            skeleton_array = np.array(skeleton)
            
            for intersection_point in raw_intersections:
                try:
                    # 가장 가까운 스켈레톤 포인트 찾기
                    distances = []
                    for skel_point in skeleton:
                        dist = np.linalg.norm(
                            np.array(intersection_point[:2]) - np.array(skel_point[:2])
                        )
                        distances.append(dist)
                    
                    if distances:
                        min_distance = min(distances)
                        if min_distance < 50:  # 50미터 이내
                            closest_idx = np.argmin(distances)
                            closest_point = tuple(skeleton[closest_idx])
                            skeleton_intersections.append(closest_point)
                        
                except Exception as e:
                    logger.debug(f"Skeleton matching failed: {e}")
                    continue
            
            # 5. 중복 제거
            filtered_intersections = []
            for point in skeleton_intersections:
                is_duplicate = False
                for existing in filtered_intersections:
                    distance = np.linalg.norm(np.array(point[:2]) - np.array(existing[:2]))
                    if distance < self.min_intersection_distance:
                        is_duplicate = True
                        break
                if not is_duplicate:
                    filtered_intersections.append(point)
            
            logger.info(f"Final intersections: {len(filtered_intersections)}")
            return filtered_intersections
            
        except Exception as e:
            logger.error(f"Intersection detection failed: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def detect_curves(self, skeleton):
        """커브 검출 (각도 분석)"""
        try:
            if len(skeleton) < 3:
                return []
            
            skeleton_array = np.array(skeleton)
            curve_indices = []
            
            # 각도 기반 커브 검출
            for i in range(1, len(skeleton_array) - 1):
                try:
                    prev_point = skeleton_array[i-1]
                    current_point = skeleton_array[i]
                    next_point = skeleton_array[i+1]
                    
                    # 벡터 계산 (2D만 사용)
                    v1 = prev_point[:2] - current_point[:2]
                    v2 = next_point[:2] - current_point[:2]
                    
                    # 벡터 크기 확인
                    norm1 = np.linalg.norm(v1)
                    norm2 = np.linalg.norm(v2)
                    
                    if norm1 > 0 and norm2 > 0:
                        # 각도 계산
                        cos_angle = np.dot(v1, v2) / (norm1 * norm2)
                        cos_angle = np.clip(cos_angle, -1, 1)
                        angle = np.degrees(np.arccos(cos_angle))
                        
                        # 커브 판정
                        if angle > self.curve_angle_threshold:
                            curve_indices.append(i)
                            
                except Exception as e:
                    logger.debug(f"Curve calculation failed at {i}: {e}")
                    continue
            
            if not curve_indices:
                return []
            
            # 연속된 커브 포인트 그룹화
            curve_groups = []
            current_group = [curve_indices[0]]
            
            for i in range(1, len(curve_indices)):
                if curve_indices[i] - current_group[-1] <= 10:
                    current_group.append(curve_indices[i])
                else:
                    if len(current_group) >= 3:
                        curve_groups.append(current_group)
                    current_group = [curve_indices[i]]
            
            # 마지막 그룹 추가
            if len(current_group) >= 3:
                curve_groups.append(current_group)
            
            # 대표점 선택
            curve_points = []
            for group in curve_groups:
                mid_idx = group[len(group) // 2]
                curve_points.append(tuple(skeleton_array[mid_idx]))
            
            logger.info(f"Curve detection: {len(curve_indices)} candidates -> {len(curve_points)} groups")
            return curve_points
            
        except Exception as e:
            logger.error(f"Curve detection failed: {e}")
            return []
    
    def detect_endpoints(self, skeleton):
        """끝점 검출 (스켈레톤 양 끝)"""
        try:
            if len(skeleton) < 2:
                return []
            
            skeleton_array = np.array(skeleton)
            endpoints = []
            
            # 시작점과 끝점
            start_point = skeleton_array[0]
            end_point = skeleton_array[-1]
            
            # 거리 확인
            distance = np.linalg.norm(start_point[:2] - end_point[:2])
            
            if distance > self.endpoint_distance:
                endpoints.extend([tuple(start_point), tuple(end_point)])
            else:
                # 중점 사용
                midpoint = (start_point + end_point) / 2
                endpoints.append(tuple(midpoint))
            
            logger.info(f"Endpoint detection: {len(endpoints)} points")
            return endpoints
            
        except Exception as e:
            logger.error(f"Endpoint detection failed: {e}")
            return []
    
    def detect_all(self, gdf, skeleton):
        """전체 검출 실행"""
        try:
            # 경고 억제
            warnings.filterwarnings('ignore', category=UserWarning)
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            
            results = {}
            
            # 교차점 검출
            logger.info("Starting intersection detection...")
            results['intersection'] = self.detect_intersections(gdf, skeleton)
            
            # 커브 검출
            logger.info("Starting curve detection...")
            results['curve'] = self.detect_curves(skeleton)
            
            # 끝점 검출
            logger.info("Starting endpoint detection...")
            results['endpoint'] = self.detect_endpoints(skeleton)
            
            # 통계 출력
            total = sum(len(points) for points in results.values())
            logger.info(f"Detection completed: {total} total points "
                       f"(intersections: {len(results['intersection'])}, "
                       f"curves: {len(results['curve'])}, "
                       f"endpoints: {len(results['endpoint'])})")
            
            return results
            
        except Exception as e:
            logger.error(f"Complete detection failed: {e}")
            import traceback
            traceback.print_exc()
            return {'intersection': [], 'curve': [], 'endpoint': []}


# 테스트 함수
def test_fixed_detector(shapefile_path):
    """수정된 검출기 테스트"""
    if not Path(shapefile_path).exists():
        print(f"❌ File not found: {shapefile_path}")
        return
    
    print(f"🧪 Testing fixed detector with: {shapefile_path}")
    
    try:
        # GeoDataFrame 로드
        gdf = gpd.read_file(shapefile_path)
        print(f"✅ Loaded GeoDataFrame: {len(gdf)} features")
        print(f"📊 Geometry types: {gdf.geometry.geom_type.value_counts().to_dict()}")
        print(f"🗺️ CRS: {gdf.crs}")
        
        # 검출기 생성 및 실행
        detector = EnhancedHeuristicDetector()
        
        # 임시 skeleton (실제로는 skeleton extractor에서 가져와야 함)
        sample_skeleton = []
        for idx, row in gdf.head(10).iterrows():  # 처음 10개만 테스트
            try:
                geom = row.geometry
                if hasattr(geom, 'coords'):
                    sample_skeleton.extend(list(geom.coords))
                elif hasattr(geom, 'exterior'):
                    sample_skeleton.extend(list(geom.exterior.coords))
            except:
                continue
        
        print(f"📍 Sample skeleton: {len(sample_skeleton)} points")
        
        # 검출 실행
        results = detector.detect_all(gdf, sample_skeleton)
        
        print(f"🎯 Detection results:")
        print(f"  - Intersections: {len(results['intersection'])}")
        print(f"  - Curves: {len(results['curve'])}")
        print(f"  - Endpoints: {len(results['endpoint'])}")
        
        return results
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("🔧 Component rings 오류 해결 테스트")
    
    # 기본 geometry 테스트
    from shapely.geometry import Polygon
    
    # 문제가 있는 polygon 생성 (자기 교차)
    coords = [(0, 0), (1, 1), (1, 0), (0, 1), (0, 0)]  # 자기 교차
    problem_polygon = Polygon(coords)
    
    print(f"문제 polygon 유효성: {problem_polygon.is_valid}")
    if not problem_polygon.is_valid:
        print(f"문제점: {validation.explain_validity(problem_polygon)}")
    
    # 수정 시도
    detector = EnhancedHeuristicDetector()
    fixed_polygon = detector.fix_geometry_properly(problem_polygon)
    
    if fixed_polygon:
        print(f"수정 후 유효성: {fixed_polygon.is_valid}")
    else:
        print("수정 실패")
    
    print("✅ 기본 테스트 완료")
