import geopandas as gpd
import networkx as nx
import numpy as np
from shapely.geometry import Point, LineString, MultiLineString
from shapely.ops import unary_union
from skimage.morphology import skeletonize
from skimage.measure import find_contours
import matplotlib.pyplot as plt
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
import json

def extract_skeleton_from_polygons(road_gdf, resolution=1.0):
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

def build_skeleton_network(skeleton_lines):
    """스켈레톤 라인들을 네트워크 그래프로 변환 (가까운 끝점들 연결)"""
    print("🔗 스켈레톤 네트워크 구축 중...")
    
    G = nx.Graph()
    
    # 모든 좌표점 수집
    all_coords = []
    line_coords = []
    line_endpoints = []  # 각 라인의 끝점들
    
    for i, line in enumerate(skeleton_lines):
        coords = list(line.coords)
        line_coords.append(coords)
        all_coords.extend(coords)
        
        # 라인의 시작점과 끝점 저장
        if len(coords) >= 2:
            line_endpoints.append((coords[0], coords[-1], i))  # (시작점, 끝점, 라인인덱스)
    
    # 중복 제거 및 인덱싱
    unique_coords = list(set(all_coords))
    coord_to_index = {coord: i for i, coord in enumerate(unique_coords)}
    
    # 노드 추가
    for i, coord in enumerate(unique_coords):
        G.add_node(i, pos=coord)
    
    # 1단계: 각 라인 내부의 연속된 점들 연결
    for coords in line_coords:
        for i in range(len(coords) - 1):
            idx1 = coord_to_index[coords[i]]
            idx2 = coord_to_index[coords[i + 1]]
            
            # 거리 계산
            p1 = np.array(coords[i])
            p2 = np.array(coords[i + 1])
            distance = np.linalg.norm(p2 - p1)
            
            G.add_edge(idx1, idx2, weight=distance)
    
    # 2단계: 각 끝점에서 가장 가까운 끝점 하나씩만 연결
    print("🔗 각 끝점에서 가장 가까운 끝점과 연결 중...")
    max_connection_distance = 50.0  # 최대 연결 거리
    connections_made = 0
    
    # 모든 끝점들 수집
    all_endpoints = []
    for line_idx, (start, end, orig_line_idx) in enumerate(line_endpoints):
        all_endpoints.append((start, orig_line_idx, line_idx, 'start'))
        all_endpoints.append((end, orig_line_idx, line_idx, 'end'))
    
    # 각 끝점에서 가장 가까운 다른 끝점 찾기
    for i, (ep1, line_idx1, line_pos1, type1) in enumerate(all_endpoints):
        min_dist = float('inf')
        closest_ep = None
        
        for j, (ep2, line_idx2, line_pos2, type2) in enumerate(all_endpoints):
            if i == j or line_idx1 == line_idx2:  # 같은 점이거나 같은 라인은 스킵
                continue
                
            dist = np.linalg.norm(np.array(ep1) - np.array(ep2))
            if dist < min_dist and dist <= max_connection_distance:
                min_dist = dist
                closest_ep = (ep2, j)
        
        # 가장 가까운 끝점과 연결
        if closest_ep is not None:
            ep2, j = closest_ep
            idx1 = coord_to_index[ep1]
            idx2 = coord_to_index[ep2]
            
            if not G.has_edge(idx1, idx2):  # 이미 연결되지 않은 경우만
                G.add_edge(idx1, idx2, weight=min_dist)
                connections_made += 1
                if connections_made <= 20:  # 처음 20개만 출력
                    print(f"  끝점 연결: 거리 {min_dist:.2f}m")
    
    print(f"✅ 네트워크 구축 완료: {G.number_of_nodes()}개 노드, {G.number_of_edges()}개 엣지")
    print(f"🔗 끝점 연결: {connections_made}개 추가 연결")
    
    # 네트워크 연결성 확인
    num_components = nx.number_connected_components(G)
    largest_component_size = len(max(nx.connected_components(G), key=len))
    print(f"📊 연결 구성요소: {num_components}개 (최대 구성요소: {largest_component_size}개 노드)")
    
    return G, unique_coords

def find_closest_skeleton_node(point, skeleton_coords, threshold=100.0):
    """점에서 가장 가까운 스켈레톤 노드 찾기 (무조건 가장 가까운 것 반환)"""
    point_coord = (point.x, point.y)
    min_dist = float('inf')
    closest_idx = None
    
    for i, skeleton_coord in enumerate(skeleton_coords):
        dist = np.linalg.norm(np.array(point_coord) - np.array(skeleton_coord))
        if dist < min_dist:
            min_dist = dist
            closest_idx = i
    
    # 임계값 내에 있거나, 100m보다 멀어도 가장 가까운 것은 반환
    if min_dist <= threshold or closest_idx is not None:
        return closest_idx, min_dist
    return None, min_dist

def check_path_has_intermediate_points(path_coords, all_points, point1_id, point2_id, threshold=15.0):
    """경로상에 다른 점들이 있는지 확인"""
    path_line = LineString(path_coords)
    
    for point_id, point_geom in all_points.items():
        if point_id in [point1_id, point2_id]:
            continue
            
        # 점과 경로의 거리 확인
        distance = path_line.distance(point_geom)
        if distance < threshold:
            return True, point_id
    
    return False, None

def analyze_point_connectivity(points_gdf, skeleton_graph, skeleton_coords):
    """점간 연결성 분석"""
    print("🔍 점간 연결성 분석 중...")
    
    # 각 점의 가장 가까운 스켈레톤 노드 찾기
    point_to_skeleton = {}
    all_points = {}
    
    for idx, row in points_gdf.iterrows():
        point_id = row['id']
        point_geom = row.geometry
        all_points[point_id] = point_geom
        
        closest_node, distance = find_closest_skeleton_node(point_geom, skeleton_coords)
        if closest_node is not None:
            point_to_skeleton[point_id] = closest_node
            print(f"점 {point_id}: 스켈레톤 노드 {closest_node} (거리: {distance:.2f}m)")
        else:
            print(f"점 {point_id}: 가까운 스켈레톤 노드 없음 (최소거리: {distance:.2f}m)")
    
    # 모든 점 쌍에 대해 연결성 확인
    connected_pairs = []
    blocked_pairs = []
    
    point_ids = list(point_to_skeleton.keys())
    
    for i, point1_id in enumerate(point_ids):
        for point2_id in point_ids[i+1:]:
            
            skeleton_node1 = point_to_skeleton[point1_id]
            skeleton_node2 = point_to_skeleton[point2_id]
            
            try:
                # 최단 경로 찾기
                path = nx.shortest_path(skeleton_graph, skeleton_node1, skeleton_node2, weight='weight')
                path_length = nx.shortest_path_length(skeleton_graph, skeleton_node1, skeleton_node2, weight='weight')
                
                # 경로의 실제 좌표들 가져오기
                path_coords = [skeleton_coords[node] for node in path]
                
                # 경로상에 다른 점이 있는지 확인
                has_intermediate, blocking_point = check_path_has_intermediate_points(
                    path_coords, all_points, point1_id, point2_id
                )
                
                if not has_intermediate:
                    connected_pairs.append({
                        'point1': point1_id,
                        'point2': point2_id,
                        'distance': path_length,
                        'path_coords': path_coords
                    })
                    print(f"✅ 점 {point1_id} - 점 {point2_id}: 연결됨 (거리: {path_length:.2f}m)")
                else:
                    blocked_pairs.append({
                        'point1': point1_id,
                        'point2': point2_id,
                        'distance': path_length,
                        'blocking_point': blocking_point
                    })
                    print(f"❌ 점 {point1_id} - 점 {point2_id}: 차단됨 (중간점: {blocking_point})")
                
            except nx.NetworkXNoPath:
                print(f"⚠️ 점 {point1_id} - 점 {point2_id}: 경로 없음")
    
    return connected_pairs, blocked_pairs

def visualize_connectivity(points_gdf, skeleton_lines, connected_pairs, blocked_pairs):
    """연결성 시각화"""
    print("🎨 시각화 생성 중...")
    
    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    
    # 스켈레톤 라인 그리기
    for line in skeleton_lines:
        x, y = line.xy
        ax.plot(x, y, 'gray', alpha=0.3, linewidth=1)
    
    # 점들 그리기
    for idx, row in points_gdf.iterrows():
        point = row.geometry
        point_id = row['id']
        ax.scatter(point.x, point.y, c='red', s=100, zorder=5)
        ax.annotate(f'P{point_id}', (point.x, point.y), xytext=(5, 5), 
                   textcoords='offset points', fontsize=12, fontweight='bold')
    
    # 연결된 점 쌍 그리기
    for pair in connected_pairs:
        point1 = points_gdf[points_gdf['id'] == pair['point1']].geometry.iloc[0]
        point2 = points_gdf[points_gdf['id'] == pair['point2']].geometry.iloc[0]
        
        ax.plot([point1.x, point2.x], [point1.y, point2.y], 
                'green', linewidth=3, alpha=0.7, zorder=3)
        
        # 거리 표시
        mid_x = (point1.x + point2.x) / 2
        mid_y = (point1.y + point2.y) / 2
        ax.annotate(f'{pair["distance"]:.1f}m', (mid_x, mid_y), 
                   fontsize=10, ha='center', 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
    
    # 차단된 점 쌍 표시
    for pair in blocked_pairs:
        point1 = points_gdf[points_gdf['id'] == pair['point1']].geometry.iloc[0]
        point2 = points_gdf[points_gdf['id'] == pair['point2']].geometry.iloc[0]
        
        ax.plot([point1.x, point2.x], [point1.y, point2.y], 
                'red', linewidth=2, alpha=0.5, linestyle='--', zorder=2)
    
    ax.set_title('점간 연결성 분석 결과\n녹색 실선: 직접 연결, 빨간 점선: 중간점으로 차단', fontsize=14)
    ax.set_xlabel('X 좌표')
    ax.set_ylabel('Y 좌표')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('point_sample/point_connectivity_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    print("🚀 점간 연결성 분석 시작!")
    
    # 데이터 로드
    print("📂 데이터 로딩 중...")
    points_gdf = gpd.read_file('point_sample/p.geojson')
    road_gdf = gpd.read_file('point_sample/road.geojson')
    
    print(f"점 개수: {len(points_gdf)}")
    print(f"도로 폴리곤 개수: {len(road_gdf)}")
    
    # 스켈레톤 추출 (해상도 증가)
    skeleton_lines = extract_skeleton_from_polygons(road_gdf, resolution=2.0)
    
    # 네트워크 구축
    skeleton_graph, skeleton_coords = build_skeleton_network(skeleton_lines)
    
    # 연결성 분석
    connected_pairs, blocked_pairs = analyze_point_connectivity(
        points_gdf, skeleton_graph, skeleton_coords
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
    visualize_connectivity(points_gdf, skeleton_lines, connected_pairs, blocked_pairs)
    
    # 결과 저장
    result = {
        'connected_pairs': connected_pairs,
        'blocked_pairs': blocked_pairs,
        'total_points': len(points_gdf),
        'total_connections': len(connected_pairs),
        'total_blocked': len(blocked_pairs)
    }
    
    # 경로 좌표는 JSON 직렬화가 안되므로 제거
    for pair in result['connected_pairs']:
        if 'path_coords' in pair:
            del pair['path_coords']
    
    with open('point_sample/connectivity_result.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 결과가 저장되었습니다:")
    print(f"  - 시각화: point_sample/point_connectivity_analysis.png")
    print(f"  - 결과 데이터: point_sample/connectivity_result.json")

if __name__ == "__main__":
    main() 