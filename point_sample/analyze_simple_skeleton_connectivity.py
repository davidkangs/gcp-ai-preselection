import geopandas as gpd
import networkx as nx
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

def build_simple_skeleton_network(skeleton_points, connection_threshold=1.0):
    """1m 이내의 스켈레톤 포인트들을 모두 연결"""
    print(f"🔗 {connection_threshold}m 이내 스켈레톤 포인트들 연결 중...")
    
    G = nx.Graph()
    
    # 노드 추가
    for i, point in enumerate(skeleton_points):
        G.add_node(i, pos=point)
    
    # 거리 행렬 계산 (scipy 사용으로 고속화)
    coords_array = np.array(skeleton_points)
    distances = cdist(coords_array, coords_array)
    
    # 임계값 이내의 모든 점들 연결
    connections_made = 0
    for i in range(len(skeleton_points)):
        for j in range(i + 1, len(skeleton_points)):
            distance = distances[i, j]
            
            if distance <= connection_threshold:
                G.add_edge(i, j, weight=distance)
                connections_made += 1
    
    print(f"✅ {connections_made}개의 연결 생성 완료")
    print(f"📊 네트워크 노드: {G.number_of_nodes()}개")
    print(f"📊 네트워크 엣지: {G.number_of_edges()}개")
    
    # 연결 구성요소 확인
    components = list(nx.connected_components(G))
    print(f"📊 연결 구성요소: {len(components)}개 (최대: {max(len(c) for c in components)}개 노드)")
    
    return G, skeleton_points

def find_closest_skeleton_nodes(points_gdf, skeleton_points, threshold=10.0):
    """각 점에서 가장 가까운 스켈레톤 노드들 찾기"""
    print("📍 점-스켈레톤 매칭 중...")
    
    point_assignments = {}
    
    # 스켈레톤 포인트들을 numpy 배열로 변환
    skeleton_array = np.array(skeleton_points)
    
    for idx, point in points_gdf.iterrows():
        point_coord = np.array([point.geometry.x, point.geometry.y])
        
        # 모든 스켈레톤 포인트와의 거리 계산
        distances = np.linalg.norm(skeleton_array - point_coord, axis=1)
        
        # 가장 가까운 노드 찾기
        closest_idx = np.argmin(distances)
        min_distance = distances[closest_idx]
        
        point_assignments[idx] = {
            'skeleton_node': closest_idx,
            'distance': min_distance,
            'original_point': point.geometry,
            'skeleton_point': skeleton_points[closest_idx]
        }
        
        print(f"점 P{point['id']}: 스켈레톤 노드 {closest_idx} (거리: {min_distance:.1f}m)")
    
    return point_assignments

def calculate_point_distances(point_assignments, G, skeleton_points):
    """네트워크상에서 점간 거리 계산"""
    print("📏 점간 거리 계산 중...")
    
    results = {
        'connected_pairs': [],
        'unreachable_pairs': []
    }
    
    # 모든 점 쌍에 대해 계산
    all_points = list(point_assignments.keys())
    
    for i, point1_idx in enumerate(all_points):
        for j, point2_idx in enumerate(all_points[i+1:], i+1):
            assignment1 = point_assignments[point1_idx]
            assignment2 = point_assignments[point2_idx]
            
            node1 = assignment1['skeleton_node']
            node2 = assignment2['skeleton_node']
            
            try:
                # 네트워크상 최단 경로 계산
                path_length = nx.shortest_path_length(G, node1, node2, weight='weight')
                
                # 시작점과 끝점에서 실제 점까지의 거리 추가
                total_distance = (assignment1['distance'] + 
                                path_length + 
                                assignment2['distance'])
                
                results['connected_pairs'].append({
                    'point1': f"P{point_assignments[point1_idx]['original_point'].coords[0][0]:.0f}_{point_assignments[point1_idx]['original_point'].coords[0][1]:.0f}",
                    'point2': f"P{point_assignments[point2_idx]['original_point'].coords[0][0]:.0f}_{point_assignments[point2_idx]['original_point'].coords[0][1]:.0f}",
                    'skeleton_distance': path_length,
                    'total_distance': total_distance,
                    'point1_to_skeleton': assignment1['distance'],
                    'point2_to_skeleton': assignment2['distance']
                })
                
            except nx.NetworkXNoPath:
                results['unreachable_pairs'].append({
                    'point1': f"P{point_assignments[point1_idx]['original_point'].coords[0][0]:.0f}_{point_assignments[point1_idx]['original_point'].coords[0][1]:.0f}",
                    'point2': f"P{point_assignments[point2_idx]['original_point'].coords[0][0]:.0f}_{point_assignments[point2_idx]['original_point'].coords[0][1]:.0f}",
                    'reason': 'no_path_in_skeleton_network'
                })
    
    return results

def create_visualization(points_gdf, skeleton_points, point_assignments, results, G):
    """결과 시각화"""
    print("🎨 시각화 생성 중...")
    
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    
    # 스켈레톤 포인트들 그리기 (작은 점들)
    skeleton_array = np.array(skeleton_points)
    ax.scatter(skeleton_array[:, 0], skeleton_array[:, 1], c='lightgray', s=1, alpha=0.5, label='Skeleton Points')
    
    # 스켈레톤 연결선들 그리기 (1m 이내 연결)
    for edge in G.edges():
        node1, node2 = edge
        point1 = skeleton_points[node1]
        point2 = skeleton_points[node2]
        ax.plot([point1[0], point2[0]], [point1[1], point2[1]], 'gray', linewidth=0.5, alpha=0.3)
    
    # 원본 점들 그리기
    for idx, point in points_gdf.iterrows():
        x, y = point.geometry.x, point.geometry.y
        ax.plot(x, y, 'ro', markersize=8)
        ax.text(x+2, y+2, f"P{point['id']}", fontsize=10, fontweight='bold', color='red')
    
    # 점-스켈레톤 연결선 그리기
    for point_idx, assignment in point_assignments.items():
        orig_x, orig_y = assignment['original_point'].x, assignment['original_point'].y
        skel_x, skel_y = assignment['skeleton_point']
        ax.plot([orig_x, skel_x], [orig_y, skel_y], 'b--', alpha=0.7, linewidth=1)
        ax.plot(skel_x, skel_y, 'bs', markersize=6)
    
    # 연결된 점 쌍들 강조 표시
    colors = ['green', 'orange', 'purple', 'brown', 'pink']
    for i, pair in enumerate(results['connected_pairs']):
        color = colors[i % len(colors)]
        # 실제 연결선은 복잡해서 단순히 직선으로 표시
        # (실제로는 스켈레톤을 따라 가는 경로)
        pass
    
    ax.set_aspect('equal')
    ax.legend()
    ax.set_title('Simple Skeleton Network Connectivity (1m threshold)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('point_sample/simple_skeleton_connectivity.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    print("🚀 단순 스켈레톤 네트워크 기반 점간 연결성 분석 시작")
    
    # 데이터 로드
    points_gdf = gpd.read_file('point_sample/p.geojson')
    road_gdf = gpd.read_file('point_sample/road.geojson')
    
    print(f"📍 점 개수: {len(points_gdf)}")
    print(f"🛣️ 도로 개수: {len(road_gdf)}")
    
    # 스켈레톤 추출
    skeleton_lines = extract_skeleton_from_polygons(road_gdf)
    
    # 모든 스켈레톤 포인트 추출
    skeleton_points = extract_all_skeleton_points(skeleton_lines)
    
    # 2m 이내 모든 포인트들 연결 (스켈레톤 포인트 간격이 평균 1.6m)
    G, skeleton_points = build_simple_skeleton_network(skeleton_points, connection_threshold=2.0)
    
    # 점-스켈레톤 매칭
    point_assignments = find_closest_skeleton_nodes(points_gdf, skeleton_points)
    
    # 거리 계산
    results = calculate_point_distances(point_assignments, G, skeleton_points)
    
    # 시각화
    create_visualization(points_gdf, skeleton_points, point_assignments, results, G)
    
    # 결과 저장
    with open('point_sample/simple_skeleton_result.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    
    # 결과 출력
    print("\n" + "="*50)
    print("📊 단순 스켈레톤 네트워크 연결성 분석 결과")
    print("="*50)
    
    print(f"\n✅ 연결된 점 쌍: {len(results['connected_pairs'])}개")
    for pair in results['connected_pairs']:
        print(f"  {pair['point1']} ↔ {pair['point2']}: {pair['total_distance']:.1f}m")
        print(f"    (스켈레톤: {pair['skeleton_distance']:.1f}m + 접근: {pair['point1_to_skeleton']:.1f}m + {pair['point2_to_skeleton']:.1f}m)")
    
    print(f"\n❌ 연결 불가 점 쌍: {len(results['unreachable_pairs'])}개")
    for pair in results['unreachable_pairs']:
        print(f"  {pair['point1']} - {pair['point2']}: {pair['reason']}")
    
    print("\n🎯 분석 완료!")

if __name__ == "__main__":
    main() 