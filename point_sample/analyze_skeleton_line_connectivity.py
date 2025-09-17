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

def build_skeleton_line_network(skeleton_lines, connection_threshold=5.0):
    """스켈레톤 라인들 사이의 거리를 기준으로 네트워크 구축"""
    print(f"🔗 {connection_threshold}m 이내 스켈레톤 라인들 연결 중...")
    
    G = nx.Graph()
    
    # 각 라인을 노드로 추가
    for i, line in enumerate(skeleton_lines):
        G.add_node(i, geometry=line)
    
    # 라인들 사이의 거리 계산 및 연결
    connections_made = 0
    
    for i in range(len(skeleton_lines)):
        for j in range(i + 1, len(skeleton_lines)):
            line1 = skeleton_lines[i]
            line2 = skeleton_lines[j]
            
            # 두 라인 사이의 최단 거리 계산
            distance = line1.distance(line2)
            
            if distance <= connection_threshold:
                G.add_edge(i, j, weight=distance)
                connections_made += 1
    
    print(f"✅ {connections_made}개의 라인 연결 생성 완료")
    print(f"📊 네트워크 라인: {G.number_of_nodes()}개")
    print(f"📊 네트워크 연결: {G.number_of_edges()}개")
    
    # 연결 구성요소 확인
    components = list(nx.connected_components(G))
    print(f"📊 연결 구성요소: {len(components)}개 (최대: {max(len(c) for c in components)}개 라인)")
    
    return G

def find_closest_skeleton_lines(points_gdf, skeleton_lines):
    """각 점에서 가장 가까운 스켈레톤 라인 찾기"""
    print("📍 점-스켈레톤라인 매칭 중...")
    
    point_assignments = {}
    
    for idx, point in points_gdf.iterrows():
        point_geom = point.geometry
        closest_line_idx = None
        min_distance = float('inf')
        projection_point = None
        
        for line_idx, line in enumerate(skeleton_lines):
            # 점에서 라인까지의 거리
            distance = point_geom.distance(line)
            
            if distance < min_distance:
                min_distance = distance
                closest_line_idx = line_idx
                # 투영점 계산
                projection_point = line.interpolate(line.project(point_geom))
        
        point_assignments[idx] = {
            'skeleton_line_idx': closest_line_idx,
            'distance_to_line': min_distance,
            'projection': projection_point,
            'original_point': point_geom
        }
        
        print(f"점 P{point['id']}: 스켈레톤 라인 {closest_line_idx} (거리: {min_distance:.1f}m)")
    
    return point_assignments

def calculate_point_distances_via_lines(point_assignments, G, skeleton_lines):
    """라인 네트워크를 통한 점간 거리 계산"""
    print("📏 라인 네트워크 기반 점간 거리 계산 중...")
    
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
            
            line1_idx = assignment1['skeleton_line_idx']
            line2_idx = assignment2['skeleton_line_idx']
            
            try:
                if line1_idx == line2_idx:
                    # 같은 라인에 있는 경우: 투영점들 사이의 라인상 거리
                    line = skeleton_lines[line1_idx]
                    proj1 = assignment1['projection']
                    proj2 = assignment2['projection']
                    
                    # 라인상에서 두 투영점 사이의 거리
                    pos1 = line.project(proj1)
                    pos2 = line.project(proj2)
                    line_distance = abs(pos2 - pos1)
                    
                    total_distance = (assignment1['distance_to_line'] + 
                                    line_distance + 
                                    assignment2['distance_to_line'])
                else:
                    # 다른 라인에 있는 경우: 라인 네트워크를 통한 최단 경로
                    path_length = nx.shortest_path_length(G, line1_idx, line2_idx, weight='weight')
                    
                    total_distance = (assignment1['distance_to_line'] + 
                                    path_length + 
                                    assignment2['distance_to_line'])
                
                results['connected_pairs'].append({
                    'point1': f"P{point_assignments[point1_idx]['original_point'].coords[0][0]:.0f}_{point_assignments[point1_idx]['original_point'].coords[0][1]:.0f}",
                    'point2': f"P{point_assignments[point2_idx]['original_point'].coords[0][0]:.0f}_{point_assignments[point2_idx]['original_point'].coords[0][1]:.0f}",
                    'line1': line1_idx,
                    'line2': line2_idx,
                    'total_distance': total_distance,
                    'point1_to_line': assignment1['distance_to_line'],
                    'point2_to_line': assignment2['distance_to_line'],
                    'same_line': line1_idx == line2_idx
                })
                
            except nx.NetworkXNoPath:
                results['unreachable_pairs'].append({
                    'point1': f"P{point_assignments[point1_idx]['original_point'].coords[0][0]:.0f}_{point_assignments[point1_idx]['original_point'].coords[0][1]:.0f}",
                    'point2': f"P{point_assignments[point2_idx]['original_point'].coords[0][0]:.0f}_{point_assignments[point2_idx]['original_point'].coords[0][1]:.0f}",
                    'line1': line1_idx,
                    'line2': line2_idx,
                    'reason': 'no_path_between_lines'
                })
    
    return results

def create_visualization(points_gdf, skeleton_lines, point_assignments, results, G):
    """결과 시각화"""
    print("🎨 시각화 생성 중...")
    
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    
    # 스켈레톤 라인들 그리기 (연결된 것과 분리된 것 구분)
    components = list(nx.connected_components(G))
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    for comp_idx, component in enumerate(components[:8]):  # 처음 8개 구성요소만
        color = colors[comp_idx % len(colors)]
        for line_idx in component:
            line = skeleton_lines[line_idx]
            x, y = line.xy
            ax.plot(x, y, color=color, linewidth=2, alpha=0.7, 
                   label=f'구성요소 {comp_idx}' if line_idx == list(component)[0] else "")
    
    # 나머지 작은 구성요소들은 회색으로
    all_large_component_lines = set()
    for component in components[:8]:
        all_large_component_lines.update(component)
    
    for line_idx, line in enumerate(skeleton_lines):
        if line_idx not in all_large_component_lines:
            x, y = line.xy
            ax.plot(x, y, color='lightgray', linewidth=1, alpha=0.3)
    
    # 원본 점들 그리기
    for idx, point in points_gdf.iterrows():
        x, y = point.geometry.x, point.geometry.y
        ax.plot(x, y, 'ko', markersize=8)
        ax.text(x+2, y+2, f"P{point['id']}", fontsize=10, fontweight='bold', color='black')
    
    # 점-라인 연결선 그리기
    for point_idx, assignment in point_assignments.items():
        orig_x, orig_y = assignment['original_point'].x, assignment['original_point'].y
        proj_x, proj_y = assignment['projection'].x, assignment['projection'].y
        ax.plot([orig_x, proj_x], [orig_y, proj_y], 'b--', alpha=0.7, linewidth=1)
        ax.plot(proj_x, proj_y, 'bs', markersize=4)
    
    ax.set_aspect('equal')
    ax.legend()
    ax.set_title('Skeleton Line Network Connectivity')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('point_sample/skeleton_line_connectivity.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    print("🚀 스켈레톤 라인 네트워크 기반 점간 연결성 분석 시작")
    
    # 데이터 로드
    points_gdf = gpd.read_file('point_sample/p.geojson')
    road_gdf = gpd.read_file('point_sample/road.geojson')
    
    print(f"📍 점 개수: {len(points_gdf)}")
    print(f"🛣️ 도로 개수: {len(road_gdf)}")
    
    # 스켈레톤 추출
    skeleton_lines = extract_skeleton_from_polygons(road_gdf)
    
    # 라인들 사이의 거리 기준으로 네트워크 구축 (5m 이내 연결)
    G = build_skeleton_line_network(skeleton_lines, connection_threshold=5.0)
    
    # 점-라인 매칭
    point_assignments = find_closest_skeleton_lines(points_gdf, skeleton_lines)
    
    # 거리 계산
    results = calculate_point_distances_via_lines(point_assignments, G, skeleton_lines)
    
    # 시각화
    create_visualization(points_gdf, skeleton_lines, point_assignments, results, G)
    
    # 결과 저장
    with open('point_sample/skeleton_line_result.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    
    # 결과 출력
    print("\n" + "="*50)
    print("📊 스켈레톤 라인 네트워크 연결성 분석 결과")
    print("="*50)
    
    print(f"\n✅ 연결된 점 쌍: {len(results['connected_pairs'])}개")
    for pair in results['connected_pairs']:
        line_info = f"라인{pair['line1']}" if pair['same_line'] else f"라인{pair['line1']}→{pair['line2']}"
        print(f"  {pair['point1']} ↔ {pair['point2']}: {pair['total_distance']:.1f}m ({line_info})")
    
    print(f"\n❌ 연결 불가 점 쌍: {len(results['unreachable_pairs'])}개")
    for pair in results['unreachable_pairs']:
        print(f"  {pair['point1']} - {pair['point2']}: {pair['reason']} (라인{pair['line1']} vs 라인{pair['line2']})")
    
    print("\n🎯 분석 완료!")

if __name__ == "__main__":
    main() 