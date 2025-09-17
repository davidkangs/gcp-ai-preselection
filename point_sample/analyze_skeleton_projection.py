import geopandas as gpd
import numpy as np
from shapely.geometry import Point, LineString, MultiLineString
from shapely.ops import unary_union, nearest_points
from skimage.morphology import skeletonize
from skimage.measure import find_contours
import matplotlib.pyplot as plt
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
import json

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
    
    # 폴리곤을 래스터로 변환
    shapes = [(geom, 1) for geom in road_gdf.geometry if geom is not None]
    raster = rasterize(shapes, out_shape=(height, width), transform=transform, fill=0, dtype='uint8')
    
    # 스켈레톤 추출
    skeleton = skeletonize(raster > 0)
    
    # 스켈레톤을 라인으로 변환
    skeleton_lines = []
    contours = find_contours(skeleton.astype(float), 0.5)
    
    for contour in contours:
        if len(contour) > 1:
            # 픽셀 좌표를 실제 좌표로 변환
            coords = []
            for point in contour:
                # row, col을 x, y로 변환
                x = bounds[0] + point[1] * resolution
                y = bounds[3] - point[0] * resolution  # Y축 뒤집기
                coords.append((x, y))
            
            if len(coords) > 1:
                skeleton_lines.append(LineString(coords))
    
    print(f"✅ {len(skeleton_lines)}개의 스켈레톤 라인 추출 완료")
    return skeleton_lines

def project_point_to_skeleton(point, skeleton_lines):
    """점을 스켈레톤에 투영하여 가장 가까운 위치 찾기"""
    min_distance = float('inf')
    projected_point = None
    closest_line = None
    
    for line in skeleton_lines:
        # 점과 라인의 가장 가까운 점들 찾기
        nearest_geoms = nearest_points(point, line)
        nearest_on_line = nearest_geoms[1]  # 라인 상의 가장 가까운 점
        
        distance = point.distance(nearest_on_line)
        if distance < min_distance:
            min_distance = distance
            projected_point = nearest_on_line
            closest_line = line
    
    return projected_point, min_distance, closest_line

def analyze_skeleton_projection_connectivity(points_gdf, skeleton_lines):
    """스켈레톤 투영 기반 점간 연결성 분석"""
    print("🔍 스켈레톤 투영 기반 연결성 분석 중...")
    
    # 각 점을 스켈레톤에 투영
    projected_points = {}
    
    for idx, row in points_gdf.iterrows():
        point_id = row['id']
        point_geom = row.geometry
        
        projected_point, distance, closest_line = project_point_to_skeleton(point_geom, skeleton_lines)
        projected_points[point_id] = {
            'original': point_geom,
            'projected': projected_point,
            'distance_to_skeleton': distance,
            'closest_line': closest_line
        }
        
        print(f"점 {point_id}: 스켈레톤까지 거리 {distance:.2f}m")
    
    # 투영된 점들 사이의 연결성 확인
    connected_pairs = []
    blocked_pairs = []
    
    point_ids = list(projected_points.keys())
    
    for i, point1_id in enumerate(point_ids):
        for point2_id in point_ids[i+1:]:
            
            proj1 = projected_points[point1_id]['projected']
            proj2 = projected_points[point2_id]['projected']
            
            # 투영된 점들 사이의 직선 거리
            direct_distance = proj1.distance(proj2)
            
            # 중간에 다른 투영점이 있는지 확인
            line_between = LineString([proj1.coords[0], proj2.coords[0]])
            has_intermediate = False
            blocking_point = None
            
            for other_id, other_data in projected_points.items():
                if other_id in [point1_id, point2_id]:
                    continue
                    
                other_proj = other_data['projected']
                distance_to_line = line_between.distance(other_proj)
                
                # 1m 이내에 있으면 중간에 끼어있다고 판단
                if distance_to_line <= 1.0:
                    has_intermediate = True
                    blocking_point = other_id
                    break
            
            if not has_intermediate:
                connected_pairs.append({
                    'point1': point1_id,
                    'point2': point2_id,
                    'distance': direct_distance,
                    'proj1': proj1,
                    'proj2': proj2
                })
                print(f"✅ 점 {point1_id} - 점 {point2_id}: 연결됨 (거리: {direct_distance:.2f}m)")
            else:
                blocked_pairs.append({
                    'point1': point1_id,
                    'point2': point2_id,
                    'distance': direct_distance,
                    'blocking_point': blocking_point
                })
                print(f"❌ 점 {point1_id} - 점 {point2_id}: 차단됨 (중간점: {blocking_point})")
    
    return connected_pairs, blocked_pairs, projected_points

def visualize_skeleton_projection(points_gdf, skeleton_lines, connected_pairs, blocked_pairs, projected_points):
    """스켈레톤 투영 결과 시각화"""
    print("🎨 시각화 생성 중...")
    
    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    
    # 스켈레톤 라인 그리기
    for line in skeleton_lines:
        x, y = line.xy
        ax.plot(x, y, 'gray', alpha=0.5, linewidth=2, label='Skeleton' if line == skeleton_lines[0] else "")
    
    # 원본 점들 그리기
    for idx, row in points_gdf.iterrows():
        point = row.geometry
        point_id = row['id']
        ax.scatter(point.x, point.y, c='red', s=120, zorder=5, edgecolor='black', linewidth=2)
        ax.annotate(f'P{point_id}', (point.x, point.y), xytext=(8, 8), 
                   textcoords='offset points', fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # 투영된 점들 그리기
    for point_id, data in projected_points.items():
        proj_point = data['projected']
        orig_point = data['original']
        
        # 투영점 표시
        ax.scatter(proj_point.x, proj_point.y, c='blue', s=80, zorder=4, 
                  marker='s', edgecolor='black', linewidth=1)
        
        # 원본점과 투영점 연결선
        ax.plot([orig_point.x, proj_point.x], [orig_point.y, proj_point.y], 
               'blue', linestyle='--', alpha=0.6, linewidth=1)
    
    # 연결된 점 쌍 그리기 (투영점들 사이)
    for pair in connected_pairs:
        proj1 = pair['proj1']
        proj2 = pair['proj2']
        
        ax.plot([proj1.x, proj2.x], [proj1.y, proj2.y], 
                'green', linewidth=4, alpha=0.8, zorder=3)
        
        # 거리 표시
        mid_x = (proj1.x + proj2.x) / 2
        mid_y = (proj1.y + proj2.y) / 2
        ax.annotate(f'{pair["distance"]:.1f}m', (mid_x, mid_y), 
                   fontsize=11, ha='center', fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))
    
    # 차단된 점 쌍 표시 (투영점들 사이)
    for pair in blocked_pairs:
        point1_proj = projected_points[pair['point1']]['projected']
        point2_proj = projected_points[pair['point2']]['projected']
        
        ax.plot([point1_proj.x, point2_proj.x], [point1_proj.y, point2_proj.y], 
                'red', linewidth=2, alpha=0.5, linestyle='--', zorder=2)
    
    ax.set_title('스켈레톤 투영 기반 점간 연결성 분석\n' +
                '빨간점: 원본점, 파란사각형: 투영점, 녹색실선: 직접연결', fontsize=14)
    ax.set_xlabel('X 좌표')
    ax.set_ylabel('Y 좌표')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('point_sample/skeleton_projection_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    print("🚀 스켈레톤 투영 기반 점간 연결성 분석 시작!")
    
    # 데이터 로드
    print("📂 데이터 로딩 중...")
    points_gdf = gpd.read_file('point_sample/p.geojson')
    road_gdf = gpd.read_file('point_sample/road.geojson')
    
    print(f"점 개수: {len(points_gdf)}")
    print(f"도로 폴리곤 개수: {len(road_gdf)}")
    
    # 스켈레톤 추출
    skeleton_lines = extract_skeleton_from_polygons(road_gdf, resolution=2.0)
    
    # 투영 기반 연결성 분석
    connected_pairs, blocked_pairs, projected_points = analyze_skeleton_projection_connectivity(
        points_gdf, skeleton_lines
    )
    
    # 결과 출력
    print(f"\n📊 분석 결과:")
    print(f"연결된 점 쌍: {len(connected_pairs)}개")
    print(f"차단된 점 쌍: {len(blocked_pairs)}개")
    
    print(f"\n✅ 직접 연결된 점 쌍들:")
    for pair in connected_pairs:
        print(f"  점 {pair['point1']} ↔ 점 {pair['point2']} (거리: {pair['distance']:.2f}m)")
    
    print(f"\n❌ 중간점으로 차단된 점 쌍들:")
    for pair in blocked_pairs:
        print(f"  점 {pair['point1']} ↔ 점 {pair['point2']} (차단점: {pair['blocking_point']})")
    
    # 시각화
    visualize_skeleton_projection(points_gdf, skeleton_lines, connected_pairs, blocked_pairs, projected_points)
    
    # 결과 저장
    result = {
        'method': 'skeleton_projection',
        'connected_pairs': [{k: v for k, v in pair.items() if k not in ['proj1', 'proj2']} for pair in connected_pairs],
        'blocked_pairs': blocked_pairs,
        'total_points': len(points_gdf),
        'total_connections': len(connected_pairs),
        'total_blocked': len(blocked_pairs)
    }
    
    with open('point_sample/skeleton_projection_result.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 결과가 저장되었습니다:")
    print(f"  - 시각화: point_sample/skeleton_projection_analysis.png")
    print(f"  - 결과 데이터: point_sample/skeleton_projection_result.json")

if __name__ == "__main__":
    main() 