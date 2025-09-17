import geopandas as gpd
import numpy as np
from shapely.geometry import Point, LineString
from shapely.ops import unary_union
from skimage.morphology import skeletonize
from skimage.measure import find_contours
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from scipy.spatial.distance import cdist

def extract_skeleton_from_polygons(road_gdf, resolution=2.0):
    """폴리곤에서 스켈레톤 라인 추출"""
    print("📍 도로 폴리곤에서 스켈레톤 추출 중...")
    
    # 모든 폴리곤 합치기
    union_geom = unary_union(road_gdf.geometry)
    
    # 바운딩 박스 계산
    bounds = union_geom.bounds
    width = int((bounds[2] - bounds[0]) / resolution)
    height = int((bounds[3] - bounds[1]) / resolution)
    
    # 변환 행렬 생성
    transform = from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], width, height)
    
    # 래스터화
    if hasattr(union_geom, 'geoms'):
        geoms = list(union_geom.geoms)
    else:
        geoms = [union_geom]
    
    # 폴리곤을 래스터로 변환
    raster = rasterize(geoms, out_shape=(height, width), transform=transform, fill=0, default_value=1)
    
    # 스켈레톤 추출
    skeleton = skeletonize(raster.astype(bool))
    
    # 스켈레톤을 라인으로 변환
    lines = []
    contours = find_contours(skeleton.astype(float), 0.5)
    
    for contour in contours:
        if len(contour) >= 2:
            # 픽셀 좌표를 실제 좌표로 변환
            coords = []
            for i, j in contour:
                x, y = rasterio.transform.xy(transform, i, j)
                coords.append((x, y))
            
            if len(coords) >= 2:
                lines.append(LineString(coords))
    
    print(f"✅ {len(lines)}개의 스켈레톤 라인 추출 완료")
    return lines

def extract_all_skeleton_points(skeleton_lines):
    """모든 스켈레톤 라인에서 점들 추출"""
    print("📍 스켈레톤 포인트 추출 중...")
    
    all_points = []
    for line in skeleton_lines:
        for coord in line.coords:
            all_points.append(coord)
    
    # 중복 제거 (소수점 3자리까지 반올림해서)
    unique_points = []
    seen = set()
    
    for point in all_points:
        rounded = (round(point[0], 3), round(point[1], 3))
        if rounded not in seen:
            seen.add(rounded)
            unique_points.append(point)
    
    print(f"✅ {len(unique_points)}개의 고유 스켈레톤 포인트 추출 완료")
    return unique_points

def analyze_skeleton_point_distances(skeleton_points):
    """스켈레톤 포인트들 사이의 거리 분석"""
    print("📏 스켈레톤 포인트 거리 분석 중...")
    
    # 거리 행렬 계산
    coords_array = np.array(skeleton_points)
    distances = cdist(coords_array, coords_array)
    
    # 자기 자신과의 거리(0) 제외
    non_zero_distances = distances[distances > 0]
    
    # 각 점에서 가장 가까운 다른 점까지의 거리
    min_distances = []
    for i in range(len(skeleton_points)):
        row = distances[i]
        non_self = row[row > 0]  # 자기 자신 제외
        if len(non_self) > 0:
            min_distances.append(np.min(non_self))
    
    print(f"\n📊 스켈레톤 포인트 거리 통계:")
    print(f"  총 포인트 수: {len(skeleton_points)}")
    print(f"  전체 거리 범위: {np.min(non_zero_distances):.3f}m ~ {np.max(non_zero_distances):.3f}m")
    print(f"  가장 가까운 이웃 거리:")
    print(f"    최소: {np.min(min_distances):.3f}m")
    print(f"    최대: {np.max(min_distances):.3f}m")
    print(f"    평균: {np.mean(min_distances):.3f}m")
    print(f"    중간값: {np.median(min_distances):.3f}m")
    
    # 1m, 2m, 5m 이내 연결 가능한 점 쌍 수 계산
    thresholds = [1.0, 2.0, 5.0, 10.0]
    for threshold in thresholds:
        count = np.sum(non_zero_distances <= threshold) // 2  # 대칭이므로 2로 나눔
        print(f"  {threshold}m 이내 연결 가능 쌍: {count}개")
    
    return min_distances

def analyze_individual_lines(skeleton_lines):
    """개별 스켈레톤 라인들의 점간 거리 분석"""
    print("\n📋 개별 라인별 점간 거리 분석:")
    
    line_stats = []
    for i, line in enumerate(skeleton_lines[:10]):  # 처음 10개만 체크
        coords = list(line.coords)
        if len(coords) >= 2:
            distances_in_line = []
            for j in range(len(coords) - 1):
                p1 = np.array(coords[j])
                p2 = np.array(coords[j + 1])
                dist = np.linalg.norm(p2 - p1)
                distances_in_line.append(dist)
            
            if distances_in_line:
                avg_dist = np.mean(distances_in_line)
                max_dist = np.max(distances_in_line)
                min_dist = np.min(distances_in_line)
                
                print(f"  라인 {i}: {len(coords)}개 점, 연속 거리 {min_dist:.3f}~{max_dist:.3f}m (평균: {avg_dist:.3f}m)")
                line_stats.append({
                    'line_idx': i,
                    'num_points': len(coords),
                    'avg_distance': avg_dist,
                    'min_distance': min_dist,
                    'max_distance': max_dist
                })
    
    return line_stats

def main():
    print("🔍 스켈레톤 포인트 거리 분석 시작")
    
    # 데이터 로드
    road_gdf = gpd.read_file('point_sample/road.geojson')
    
    print(f"🛣️ 도로 개수: {len(road_gdf)}")
    
    # 스켈레톤 추출
    skeleton_lines = extract_skeleton_from_polygons(road_gdf, resolution=2.0)
    
    # 개별 라인 분석
    line_stats = analyze_individual_lines(skeleton_lines)
    
    # 모든 스켈레톤 포인트 추출
    skeleton_points = extract_all_skeleton_points(skeleton_lines)
    
    # 포인트 거리 분석
    min_distances = analyze_skeleton_point_distances(skeleton_points)
    
    print(f"\n💡 결론:")
    print(f"  - 스켈레톤이 중심선으로 제대로 추출되려면 연속 점간 거리가 1-2m 내외여야 함")
    print(f"  - 현재 가장 가까운 이웃 평균 거리: {np.mean(min_distances):.1f}m")
    
    if np.mean(min_distances) > 5:
        print(f"  ⚠️  너무 성기게 추출됨 → 해상도 개선 필요")
    elif np.mean(min_distances) < 0.5:
        print(f"  ⚠️  너무 조밀하게 추출됨 → 연결 임계값 증가 필요")
    else:
        print(f"  ✅ 적절한 밀도로 추출됨 → 연결 임계값 조정으로 해결 가능")

if __name__ == "__main__":
    main() 