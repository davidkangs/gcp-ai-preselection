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
from collections import defaultdict

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

def build_skeleton_network(skeleton_lines):
    """스켈레톤을 네트워크로 변환하고 연결"""
    print("🔗 스켈레톤 네트워크 구축 중...")
    
    G = nx.Graph()
    
    # 모든 좌표점 수집
    all_coords = []
    line_coords = []
    
    for i, line in enumerate(skeleton_lines):
        coords = list(line.coords)
        line_coords.append((coords, i))
        all_coords.extend(coords)
    
    # 중복 제거 및 인덱싱
    unique_coords = list(set(all_coords))
    coord_to_index = {coord: i for i, coord in enumerate(unique_coords)}
    
    # 노드 추가
    for i, coord in enumerate(unique_coords):
        G.add_node(i, pos=coord)
    
    # 각 라인의 연속된 점들을 연결
    for coords, line_idx in line_coords:
        for i in range(len(coords) - 1):
            idx1 = coord_to_index[coords[i]]
            idx2 = coord_to_index[coords[i + 1]]
            
            # 거리 계산
            p1 = np.array(coords[i])
            p2 = np.array(coords[i + 1])
            dist = np.linalg.norm(p2 - p1)
            
            G.add_edge(idx1, idx2, weight=dist, line_idx=line_idx)
    
    # 가까운 끝점들 연결 (각 끝점에서 가장 가까운 것 하나만)
    print("🔗 가까운 끝점들 연결 중...")
    
    # 각 라인의 끝점들 찾기
    line_endpoints = []
    for coords, line_idx in line_coords:
        if len(coords) >= 2:
            start_coord = coords[0]
            end_coord = coords[-1]
            line_endpoints.append((start_coord, end_coord, line_idx))
    
    # 각 끝점에서 가장 가까운 다른 끝점 찾아서 연결
    for i, (start1, end1, line_idx1) in enumerate(line_endpoints):
        for endpoint1 in [start1, end1]:
            closest_dist = float('inf')
            closest_endpoint = None
            closest_line_idx = None
            
            for j, (start2, end2, line_idx2) in enumerate(line_endpoints):
                if line_idx1 == line_idx2:  # 같은 라인은 스킵
                    continue
                    
                for endpoint2 in [start2, end2]:
                    dist = np.linalg.norm(np.array(endpoint1) - np.array(endpoint2))
                    if dist < closest_dist and dist <= 20.0:  # 20m 이내만
                        closest_dist = dist
                        closest_endpoint = endpoint2
                        closest_line_idx = line_idx2
            
            # 가장 가까운 끝점과 연결
            if closest_endpoint is not None:
                idx1 = coord_to_index[endpoint1]
                idx2 = coord_to_index[closest_endpoint]
                
                if not G.has_edge(idx1, idx2):
                    G.add_edge(idx1, idx2, weight=closest_dist, line_idx=-1)  # 연결선은 -1
    
    print(f"📊 네트워크 노드: {G.number_of_nodes()}개")
    print(f"📊 네트워크 엣지: {G.number_of_edges()}개")
    
    return G, unique_coords, coord_to_index

def find_network_segments(G, coord_to_index):
    """네트워크에서 세그먼트(구간) 추출"""
    print("🔍 도로 세그먼트 추출 중...")
    
    # 교차점/분기점/끝점 찾기 (degree가 1이거나 3+ 인 노드)
    junction_nodes = []
    for node in G.nodes():
        degree = G.degree(node)
        if degree == 1 or degree >= 3:  # 끝점 또는 교차점
            junction_nodes.append(node)
    
    print(f"📍 교차점/끝점: {len(junction_nodes)}개")
    
    # 교차점들 사이의 경로를 세그먼트로 만들기
    segments = []
    visited_edges = set()
    
    for start_node in junction_nodes:
        for neighbor in G.neighbors(start_node):
            edge = tuple(sorted([start_node, neighbor]))
            if edge in visited_edges:
                continue
                
            # 경로 추적
            path = [start_node, neighbor]
            current = neighbor
            visited_edges.add(edge)
            
            # degree가 2인 노드들을 따라 계속 진행
            while G.degree(current) == 2 and current not in junction_nodes:
                next_nodes = [n for n in G.neighbors(current) if n != path[-2]]
                if not next_nodes:
                    break
                next_node = next_nodes[0]
                
                edge = tuple(sorted([current, next_node]))
                if edge in visited_edges:
                    break
                    
                visited_edges.add(edge)
                path.append(next_node)
                current = next_node
            
            # 세그먼트 좌표 변환
            segment_coords = []
            for node in path:
                pos = G.nodes[node]['pos']
                segment_coords.append(pos)
            
            if len(segment_coords) >= 2:
                segment_line = LineString(segment_coords)
                segments.append({
                    'line': segment_line,
                    'nodes': path,
                    'start_junction': path[0] in junction_nodes,
                    'end_junction': path[-1] in junction_nodes
                })
    
    print(f"🛣️ 추출된 세그먼트: {len(segments)}개")
    return segments, junction_nodes

def assign_points_to_segments(points_gdf, segments):
    """점들을 가장 가까운 세그먼트에 할당"""
    print("📍 점-세그먼트 매칭 중...")
    
    point_assignments = {}
    
    for idx, point in points_gdf.iterrows():
        point_geom = point.geometry
        closest_segment_idx = None
        min_distance = float('inf')
        projection_point = None
        
        for seg_idx, segment in enumerate(segments):
            # 점에서 세그먼트까지의 거리
            distance = point_geom.distance(segment['line'])
            
            if distance < min_distance:
                min_distance = distance
                closest_segment_idx = seg_idx
                # 투영점 계산
                projection_point = segment['line'].interpolate(segment['line'].project(point_geom))
        
        point_assignments[idx] = {
            'segment_idx': closest_segment_idx,
            'distance_to_segment': min_distance,
            'projection': projection_point,
            'original_point': point_geom
        }
        
        print(f"점 P{point['id']}: 세그먼트 {closest_segment_idx} (거리: {min_distance:.1f}m)")
    
    return point_assignments

def find_adjacent_segments(segments, junction_nodes, G):
    """인접한 세그먼트들 찾기"""
    print("🔗 세그먼트 인접성 분석 중...")
    
    adjacency = defaultdict(set)
    
    for i, seg1 in enumerate(segments):
        for j, seg2 in enumerate(segments):
            if i >= j:
                continue
                
            # 세그먼트들이 공통 교차점을 공유하는지 확인
            seg1_junctions = set()
            seg2_junctions = set()
            
            if seg1['start_junction']:
                seg1_junctions.add(seg1['nodes'][0])
            if seg1['end_junction']:
                seg1_junctions.add(seg1['nodes'][-1])
                
            if seg2['start_junction']:
                seg2_junctions.add(seg2['nodes'][0])
            if seg2['end_junction']:
                seg2_junctions.add(seg2['nodes'][-1])
            
            # 공통 교차점이 있으면 인접
            if seg1_junctions & seg2_junctions:
                adjacency[i].add(j)
                adjacency[j].add(i)
                print(f"세그먼트 {i} ↔ {j} 인접")
    
    return adjacency

def calculate_segment_distances(point_assignments, segments, adjacency):
    """세그먼트 기반 점간 거리 계산"""
    print("📏 점간 거리 계산 중...")
    
    results = {
        'connected_pairs': [],
        'blocked_pairs': [],
        'segment_info': {}
    }
    
    # 세그먼트별 점들 그룹핑
    segment_points = defaultdict(list)
    for point_idx, assignment in point_assignments.items():
        segment_points[assignment['segment_idx']].append(point_idx)
    
    results['segment_info'] = {
        seg_idx: [f"P{point_assignments[p]['original_point'].coords[0]}" for p in points] 
        for seg_idx, points in segment_points.items()
    }
    
    # 모든 점 쌍에 대해 검사
    all_points = list(point_assignments.keys())
    
    for i, point1_idx in enumerate(all_points):
        for j, point2_idx in enumerate(all_points[i+1:], i+1):
            assignment1 = point_assignments[point1_idx]
            assignment2 = point_assignments[point2_idx]
            
            seg1_idx = assignment1['segment_idx']
            seg2_idx = assignment2['segment_idx']
            
            # 같은 세그먼트인 경우
            if seg1_idx == seg2_idx:
                # 중간에 다른 점이 있는지 확인
                segment_points_in_seg = segment_points[seg1_idx]
                if len(segment_points_in_seg) == 2:  # 이 세그먼트에 점이 2개뿐
                    distance = assignment1['projection'].distance(assignment2['projection'])
                    results['connected_pairs'].append({
                        'point1': f"P{point_assignments[point1_idx]['original_point'].coords[0]}",
                        'point2': f"P{point_assignments[point2_idx]['original_point'].coords[0]}",
                        'distance': distance,
                        'type': 'same_segment'
                    })
                else:
                    results['blocked_pairs'].append({
                        'point1': f"P{point_assignments[point1_idx]['original_point'].coords[0]}",
                        'point2': f"P{point_assignments[point2_idx]['original_point'].coords[0]}",
                        'reason': 'intermediate_points_in_segment'
                    })
            
            # 인접한 세그먼트인 경우
            elif seg2_idx in adjacency[seg1_idx]:
                distance = assignment1['projection'].distance(assignment2['projection'])
                results['connected_pairs'].append({
                    'point1': f"P{point_assignments[point1_idx]['original_point'].coords[0]}",
                    'point2': f"P{point_assignments[point2_idx]['original_point'].coords[0]}",
                    'distance': distance,
                    'type': 'adjacent_segments'
                })
            
            # 비인접 세그먼트인 경우
            else:
                results['blocked_pairs'].append({
                    'point1': f"P{point_assignments[point1_idx]['original_point'].coords[0]}",
                    'point2': f"P{point_assignments[point2_idx]['original_point'].coords[0]}",
                    'reason': 'non_adjacent_segments'
                })
    
    return results

def create_visualization(points_gdf, segments, point_assignments, results):
    """결과 시각화"""
    print("🎨 시각화 생성 중...")
    
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    
    # 세그먼트 그리기 (각기 다른 색상)
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    for i, segment in enumerate(segments):
        color = colors[i % len(colors)]
        x, y = segment['line'].xy
        ax.plot(x, y, color=color, linewidth=2, label=f'세그먼트 {i}')
    
    # 원본 점들 그리기
    for idx, point in points_gdf.iterrows():
        x, y = point.geometry.x, point.geometry.y
        ax.plot(x, y, 'ko', markersize=8)
        ax.text(x+2, y+2, f"P{point['id']}", fontsize=10, fontweight='bold')
    
    # 투영점들 그리기
    for point_idx, assignment in point_assignments.items():
        proj_x, proj_y = assignment['projection'].x, assignment['projection'].y
        ax.plot(proj_x, proj_y, 'bs', markersize=6)
        
        # 투영선 그리기
        orig_x, orig_y = assignment['original_point'].x, assignment['original_point'].y
        ax.plot([orig_x, proj_x], [orig_y, proj_y], 'b--', alpha=0.5)
    
    # 연결된 점 쌍 그리기
    for pair in results['connected_pairs']:
        # 실제 좌표로 변환 필요 (간단히 하기 위해 생략)
        pass
    
    ax.set_aspect('equal')
    ax.legend()
    ax.set_title('스켈레톤 세그먼트 기반 점간 연결성 분석')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('point_sample/segment_connectivity_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    print("🚀 스켈레톤 세그먼트 기반 점간 연결성 분석 시작")
    
    # 데이터 로드
    points_gdf = gpd.read_file('point_sample/p.geojson')
    road_gdf = gpd.read_file('point_sample/road.geojson')
    
    print(f"📍 점 개수: {len(points_gdf)}")
    print(f"🛣️ 도로 개수: {len(road_gdf)}")
    
    # 스켈레톤 추출
    skeleton_lines = extract_skeleton_from_polygons(road_gdf)
    
    # 네트워크 구축
    G, unique_coords, coord_to_index = build_skeleton_network(skeleton_lines)
    
    # 세그먼트 추출
    segments, junction_nodes = find_network_segments(G, coord_to_index)
    
    # 점-세그먼트 매칭
    point_assignments = assign_points_to_segments(points_gdf, segments)
    
    # 세그먼트 인접성 분석
    adjacency = find_adjacent_segments(segments, junction_nodes, G)
    
    # 거리 계산
    results = calculate_segment_distances(point_assignments, segments, adjacency)
    
    # 시각화
    create_visualization(points_gdf, segments, point_assignments, results)
    
    # 결과 저장
    with open('point_sample/segment_connectivity_result.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    
    # 결과 출력
    print("\n" + "="*50)
    print("📊 세그먼트 기반 연결성 분석 결과")
    print("="*50)
    
    print(f"\n✅ 연결된 점 쌍: {len(results['connected_pairs'])}개")
    for pair in results['connected_pairs']:
        print(f"  {pair['point1']} ↔ {pair['point2']}: {pair['distance']:.1f}m ({pair['type']})")
    
    print(f"\n❌ 차단된 점 쌍: {len(results['blocked_pairs'])}개")
    for pair in results['blocked_pairs']:
        print(f"  {pair['point1']} - {pair['point2']}: {pair['reason']}")
    
    print(f"\n🛣️ 세그먼트별 점 분포:")
    for seg_idx, points in results['segment_info'].items():
        print(f"  세그먼트 {seg_idx}: {', '.join(points)}")
    
    print("\n🎯 분석 완료!")

if __name__ == "__main__":
    main() 