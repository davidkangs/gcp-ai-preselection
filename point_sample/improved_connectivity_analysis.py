import geopandas as gpd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import json
from shapely.geometry import Point, LineString, MultiLineString
from shapely.ops import unary_union, nearest_points
from skimage.morphology import skeletonize
from skimage.measure import find_contours
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from scipy.spatial.distance import cdist

def extract_skeleton_from_polygons(road_gdf, resolution=1.5):
    """폴리곤에서 스켈레톤 라인 추출 (해상도 개선)"""
    print("📍 도로 폴리곤에서 스켈레톤 추출 중...")
    
    union_geom = unary_union(road_gdf.geometry)
    bounds = union_geom.bounds
    width = int((bounds[2] - bounds[0]) / resolution)
    height = int((bounds[3] - bounds[1]) / resolution)
    transform = from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], width, height)
    
    if hasattr(union_geom, 'geoms'):
        geoms = list(union_geom.geoms)
    else:
        geoms = [union_geom]
    
    raster = rasterize(geoms, out_shape=(height, width), transform=transform, fill=0, default_value=1)
    skeleton = skeletonize(raster.astype(bool))
    
    lines = []
    contours = find_contours(skeleton.astype(float), 0.5)
    
    for contour in contours:
        if len(contour) >= 2:
            coords = []
            for i, j in contour:
                x, y = rasterio.transform.xy(transform, i, j)
                coords.append((x, y))
            if len(coords) >= 2:
                lines.append(LineString(coords))
    
    print(f"✅ {len(lines)}개의 스켈레톤 라인 추출 완료")
    return lines

def connect_skeleton_lines_properly(skeleton_lines, max_gap=3.0):
    """스켈레톤 라인들을 올바르게 연결하여 네트워크 구성"""
    print("🔗 스켈레톤 라인 네트워크 구성 중...")
    
    G = nx.Graph()
    
    # 각 라인을 노드로 추가
    for i, line in enumerate(skeleton_lines):
        G.add_node(i, geometry=line)
    
    # 라인들 간의 연결 찾기 (끝점 기준)
    connections = 0
    for i, line1 in enumerate(skeleton_lines):
        for j, line2 in enumerate(skeleton_lines):
            if i >= j:
                continue
                
            # 각 라인의 시작점과 끝점
            line1_start = Point(line1.coords[0])
            line1_end = Point(line1.coords[-1])
            line2_start = Point(line2.coords[0])
            line2_end = Point(line2.coords[-1])
            
            # 모든 끝점 조합 중 가장 가까운 거리 찾기
            distances = [
                line1_start.distance(line2_start),
                line1_start.distance(line2_end),
                line1_end.distance(line2_start),
                line1_end.distance(line2_end)
            ]
            
            min_distance = min(distances)
            
            # 거리가 임계값 이하이면 연결
            if min_distance <= max_gap:
                G.add_edge(i, j, weight=min_distance)
                connections += 1
    
    # 연결성 분석
    components = list(nx.connected_components(G))
    print(f"✅ {connections}개 연결, {len(components)}개 구성요소")
    
    return G, components

def assign_points_to_lines(points_gdf, skeleton_lines, max_distance=5.0):
    """점들을 가장 가까운 스켈레톤 라인에 할당"""
    print("📍 점들을 스켈레톤 라인에 할당 중...")
    
    assignments = {}
    for idx, point in points_gdf.iterrows():
        point_geom = point.geometry
        point_id = f"P{point['id']}"
        
        min_distance = float('inf')
        best_line = -1
        best_projection = None
        
        for line_idx, line in enumerate(skeleton_lines):
            distance = point_geom.distance(line)
            if distance < min_distance and distance <= max_distance:
                min_distance = distance
                best_line = line_idx
                # 투영점 계산
                best_projection = line.interpolate(line.project(point_geom))
        
        if best_line != -1:
            assignments[point_id] = {
                'line_idx': best_line,
                'distance_to_line': min_distance,
                'original_point': point_geom,
                'projection': best_projection
            }
            print(f"  {point_id} → 라인{best_line} (거리: {min_distance:.2f}m)")
        else:
            print(f"  ❌ {point_id}: 할당 불가 (가장 가까운 거리: {min_distance:.2f}m)")
    
    return assignments

def find_direct_connections(point_assignments, line_network, skeleton_lines):
    """직접 연결된 점 쌍만 찾기 (중간에 다른 점이 없는 경우)"""
    print("🔍 직접 연결 가능한 점 쌍 찾기 중...")
    
    point_names = list(point_assignments.keys())
    direct_connections = []
    blocked_connections = []
    
    for i, point1 in enumerate(point_names):
        for j, point2 in enumerate(point_names):
            if i >= j:
                continue
                
            line1 = point_assignments[point1]['line_idx']
            line2 = point_assignments[point2]['line_idx']
            
            # 같은 라인에 있는 경우
            if line1 == line2:
                # 같은 라인 상에서 두 점 사이에 다른 점이 있는지 확인
                line_geom = skeleton_lines[line1]
                proj1 = point_assignments[point1]['projection']
                proj2 = point_assignments[point2]['projection']
                
                # 라인 상에서의 위치 (0~1)
                pos1 = line_geom.project(proj1, normalized=True)
                pos2 = line_geom.project(proj2, normalized=True)
                
                # 두 점 사이 구간에 다른 점이 있는지 확인
                min_pos, max_pos = min(pos1, pos2), max(pos1, pos2)
                has_intermediate = False
                
                for other_point in point_names:
                    if other_point in [point1, point2]:
                        continue
                    if point_assignments[other_point]['line_idx'] == line1:
                        other_proj = point_assignments[other_point]['projection']
                        other_pos = line_geom.project(other_proj, normalized=True)
                        if min_pos < other_pos < max_pos:
                            has_intermediate = True
                            break
                
                if not has_intermediate:
                    distance = proj1.distance(proj2)
                    direct_connections.append({
                        'point1': point1,
                        'point2': point2,
                        'line1': line1,
                        'line2': line2,
                        'distance': distance,
                        'type': 'same_line'
                    })
                else:
                    blocked_connections.append({
                        'point1': point1,
                        'point2': point2,
                        'line1': line1,
                        'line2': line2,
                        'reason': 'intermediate_point_on_same_line'
                    })
            
            # 다른 라인에 있는 경우 - 네트워크에서 연결되어 있는지 확인
            else:
                if line_network.has_node(line1) and line_network.has_node(line2):
                    try:
                        # 최단 경로 찾기
                        path = nx.shortest_path(line_network, line1, line2)
                        if len(path) == 2:  # 직접 연결된 경우만
                            # 경로 상에 다른 점이 있는지 확인
                            has_intermediate = False
                            for other_point in point_names:
                                if other_point in [point1, point2]:
                                    continue
                                other_line = point_assignments[other_point]['line_idx']
                                if other_line in path:
                                    has_intermediate = True
                                    break
                            
                            if not has_intermediate:
                                # 거리 계산 (두 투영점 간 직선거리)
                                proj1 = point_assignments[point1]['projection']
                                proj2 = point_assignments[point2]['projection']
                                distance = proj1.distance(proj2)
                                
                                direct_connections.append({
                                    'point1': point1,
                                    'point2': point2,
                                    'line1': line1,
                                    'line2': line2,
                                    'distance': distance,
                                    'type': 'connected_lines'
                                })
                            else:
                                blocked_connections.append({
                                    'point1': point1,
                                    'point2': point2,
                                    'line1': line1,
                                    'line2': line2,
                                    'reason': 'intermediate_point_on_path'
                                })
                        else:
                            blocked_connections.append({
                                'point1': point1,
                                'point2': point2,
                                'line1': line1,
                                'line2': line2,
                                'reason': 'not_directly_connected'
                            })
                    except nx.NetworkXNoPath:
                        blocked_connections.append({
                            'point1': point1,
                            'point2': point2,
                            'line1': line1,
                            'line2': line2,
                            'reason': 'no_path'
                        })
                else:
                    blocked_connections.append({
                        'point1': point1,
                        'point2': point2,
                        'line1': line1,
                        'line2': line2,
                        'reason': 'line_not_in_network'
                    })
    
    print(f"✅ 직접 연결: {len(direct_connections)}개")
    print(f"❌ 차단된 연결: {len(blocked_connections)}개")
    
    return direct_connections, blocked_connections

def create_comprehensive_visualization(points_gdf, skeleton_lines, point_assignments, 
                                     direct_connections, blocked_connections, line_network):
    """종합 시각화 생성"""
    print("🎨 종합 시각화 생성 중...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # 1. 원본 도로 폴리곤과 점들
    road_gdf = gpd.read_file('road.geojson')
    road_gdf.plot(ax=ax1, color='lightblue', alpha=0.7, edgecolor='blue')
    points_gdf.plot(ax=ax1, color='red', markersize=100)
    for idx, point in points_gdf.iterrows():
        ax1.text(point.geometry.x+2, point.geometry.y+2, f"P{point['id']}", 
                fontsize=12, fontweight='bold', color='black')
    ax1.set_title('1. 원본 도로 폴리곤과 점들', fontsize=14, fontweight='bold')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    
    # 2. 스켈레톤 라인과 네트워크 연결
    components = list(nx.connected_components(line_network))
    colors = plt.cm.Set3(np.linspace(0, 1, len(components)))
    
    for comp_idx, component in enumerate(components):
        color = colors[comp_idx]
        for line_idx in component:
            line = skeleton_lines[line_idx]
            x, y = line.xy
            ax2.plot(x, y, color=color, linewidth=3, alpha=0.8)
    
    # 네트워크 연결선 표시
    for edge in line_network.edges():
        line1 = skeleton_lines[edge[0]]
        line2 = skeleton_lines[edge[1]]
        # 가장 가까운 끝점들 찾기
        endpoints1 = [Point(line1.coords[0]), Point(line1.coords[-1])]
        endpoints2 = [Point(line2.coords[0]), Point(line2.coords[-1])]
        
        min_dist = float('inf')
        best_points = None
        for p1 in endpoints1:
            for p2 in endpoints2:
                dist = p1.distance(p2)
                if dist < min_dist:
                    min_dist = dist
                    best_points = (p1, p2)
        
        if best_points:
            ax2.plot([best_points[0].x, best_points[1].x], 
                    [best_points[0].y, best_points[1].y], 
                    'r--', linewidth=2, alpha=0.7)
    
    points_gdf.plot(ax=ax2, color='red', markersize=80)
    for idx, point in points_gdf.iterrows():
        ax2.text(point.geometry.x+2, point.geometry.y+2, f"P{point['id']}", 
                fontsize=11, fontweight='bold', color='black')
    ax2.set_title(f'2. 스켈레톤 네트워크 ({len(components)}개 구성요소)', fontsize=14, fontweight='bold')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    
    # 3. 점-라인 할당과 직접 연결
    for line_idx, line in enumerate(skeleton_lines):
        x, y = line.xy
        ax3.plot(x, y, color='lightgray', linewidth=2, alpha=0.5)
    
    # 점-라인 할당 표시
    for point_id, assignment in point_assignments.items():
        orig = assignment['original_point']
        proj = assignment['projection']
        line_idx = assignment['line_idx']
        
        # 할당된 라인 강조
        line = skeleton_lines[line_idx]
        x, y = line.xy
        ax3.plot(x, y, color='blue', linewidth=3, alpha=0.8)
        
        # 투영선
        ax3.plot([orig.x, proj.x], [orig.y, proj.y], 'g--', linewidth=2, alpha=0.7)
        ax3.plot(proj.x, proj.y, 'go', markersize=8)
        ax3.plot(orig.x, orig.y, 'ro', markersize=10)
        ax3.text(orig.x+2, orig.y+2, point_id, fontsize=11, fontweight='bold', color='black')
    
    # 직접 연결 표시
    for conn in direct_connections:
        point1_proj = point_assignments[conn['point1']]['projection']
        point2_proj = point_assignments[conn['point2']]['projection']
        ax3.plot([point1_proj.x, point2_proj.x], [point1_proj.y, point2_proj.y], 
                'orange', linewidth=4, alpha=0.8)
    
    ax3.set_title(f'3. 점-라인 할당 및 직접 연결 ({len(direct_connections)}개)', fontsize=14, fontweight='bold')
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)
    
    # 4. 연결성 분석 결과 요약
    ax4.axis('off')
    
    # 결과 텍스트
    result_text = f"""
연결성 분석 결과 요약

📊 전체 점 개수: {len(points_gdf)}개
🔗 직접 연결 가능: {len(direct_connections)}개
❌ 연결 불가: {len(blocked_connections)}개

✅ 직접 연결된 점 쌍:
"""
    
    for i, conn in enumerate(direct_connections):
        result_text += f"{i+1:2d}. {conn['point1']} ↔ {conn['point2']} ({conn['distance']:.1f}m)\n"
        result_text += f"    타입: {conn['type']}, 라인: {conn['line1']}"
        if conn['line1'] != conn['line2']:
            result_text += f" → {conn['line2']}"
        result_text += "\n"
    
    result_text += f"\n❌ 연결 불가 사유:\n"
    reason_counts = {}
    for conn in blocked_connections:
        reason = conn['reason']
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    
    for reason, count in reason_counts.items():
        result_text += f"• {reason}: {count}개\n"
    
    ax4.text(0.05, 0.95, result_text, transform=ax4.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    ax4.set_title('4. 연결성 분석 결과', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('improved_connectivity_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    print("🚀 개선된 점간 연결성 분석 시작")
    print("="*50)
    
    # 데이터 로드
    points_gdf = gpd.read_file('p.geojson')
    road_gdf = gpd.read_file('road.geojson')
    
    print(f"📍 점 개수: {len(points_gdf)}")
    print(f"🛣️ 도로 개수: {len(road_gdf)}")
    
    # 1. 스켈레톤 추출
    skeleton_lines = extract_skeleton_from_polygons(road_gdf, resolution=1.5)
    
    # 2. 스켈레톤 라인 네트워크 구성
    line_network, components = connect_skeleton_lines_properly(skeleton_lines, max_gap=3.0)
    
    # 3. 점-라인 할당
    point_assignments = assign_points_to_lines(points_gdf, skeleton_lines, max_distance=5.0)
    
    # 4. 직접 연결 찾기
    direct_connections, blocked_connections = find_direct_connections(
        point_assignments, line_network, skeleton_lines)
    
    # 5. 시각화
    create_comprehensive_visualization(points_gdf, skeleton_lines, point_assignments,
                                     direct_connections, blocked_connections, line_network)
    
    # 6. 결과 저장
    results = {
        'analysis_info': {
            'total_points': len(points_gdf),
            'skeleton_lines': len(skeleton_lines),
            'network_components': len(components),
            'assigned_points': len(point_assignments)
        },
        'direct_connections': direct_connections,
        'blocked_connections': blocked_connections
    }
    
    with open('improved_connectivity_result.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    
    # 결과 출력
    print("\n" + "="*50)
    print("📊 개선된 연결성 분석 결과")
    print("="*50)
    
    print(f"\n✅ 직접 연결된 점 쌍: {len(direct_connections)}개")
    for conn in direct_connections:
        type_str = "같은라인" if conn['type'] == 'same_line' else "연결라인"
        print(f"  {conn['point1']} ↔ {conn['point2']}: {conn['distance']:.1f}m ({type_str})")
    
    print(f"\n❌ 연결 불가 점 쌍: {len(blocked_connections)}개")
    reason_counts = {}
    for conn in blocked_connections:
        reason = conn['reason']
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        print(f"  {conn['point1']} - {conn['point2']}: {conn['reason']}")
    
    print(f"\n📈 차단 사유 요약:")
    for reason, count in reason_counts.items():
        print(f"  • {reason}: {count}개")
    
    print("\n🎯 분석 완료!")

if __name__ == "__main__":
    main() 