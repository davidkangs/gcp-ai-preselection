#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
샘플 점간 연결성 분석 테스트
목적: 도로 스켈레톤을 따라 점들이 어떻게 연결되는지 시각화 및 분석
"""

import geopandas as gpd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from shapely.geometry import Point, LineString
from shapely.ops import unary_union, nearest_points
from skimage.morphology import skeletonize
from skimage.measure import find_contours
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from scipy.spatial.distance import cdist
import json
from pathlib import Path

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def load_data():
    """테스트 데이터 로드"""
    print("📂 데이터 로딩 중...")
    
    # 데이터 로드
    points_gdf = gpd.read_file("p.geojson")
    road_gdf = gpd.read_file("road.geojson")
    
    print(f"   점 데이터: {len(points_gdf)}개")
    print(f"   도로 데이터: {len(road_gdf)}개 폴리곤")
    
    return points_gdf, road_gdf

def extract_skeleton_network(road_gdf, resolution=1.0):
    """도로 폴리곤에서 스켈레톤 네트워크 추출"""
    print("🦴 스켈레톤 추출 중...")
    
    # 도로 폴리곤 통합
    union_geom = unary_union(road_gdf.geometry)
    bounds = union_geom.bounds
    
    # 래스터화 설정
    width = int((bounds[2] - bounds[0]) / resolution)
    height = int((bounds[3] - bounds[1]) / resolution)
    transform = from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], width, height)
    
    # 폴리곤을 래스터로 변환
    if hasattr(union_geom, 'geoms'):
        geoms = list(union_geom.geoms)
    else:
        geoms = [union_geom]
    
    raster = rasterize(geoms, out_shape=(height, width), transform=transform, fill=0, default_value=1)
    
    # 스켈레톤 추출
    skeleton = skeletonize(raster.astype(bool))
    
    # 스켈레톤을 라인으로 변환
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
    
    print(f"   {len(lines)}개의 스켈레톤 라인 추출")
    return lines, union_geom

def build_network_graph(skeleton_lines, connection_threshold=5.0):
    """스켈레톤 라인들을 네트워크 그래프로 구성"""
    print("🕸️ 네트워크 그래프 구성 중...")
    
    G = nx.Graph()
    
    # 모든 스켈레톤 점들을 노드로 추가
    node_id = 0
    line_points = []
    
    for line in skeleton_lines:
        coords = list(line.coords)
        for coord in coords:
            G.add_node(node_id, pos=coord, coord=coord)
            line_points.append((node_id, coord))
            node_id += 1
    
    print(f"   {len(G.nodes)}개 노드 생성")
    
    # 가까운 점들 간 엣지 생성
    coords = np.array([coord for _, coord in line_points])
    distances = cdist(coords, coords)
    
    edge_count = 0
    for i, (node_i, coord_i) in enumerate(line_points):
        for j, (node_j, coord_j) in enumerate(line_points):
            if i < j and distances[i, j] <= connection_threshold:
                G.add_edge(node_i, node_j, weight=distances[i, j])
                edge_count += 1
    
    print(f"   {edge_count}개 엣지 생성 (임계값: {connection_threshold}m)")
    return G

def project_points_to_skeleton(points_gdf, skeleton_lines):
    """점들을 가장 가까운 스켈레톤에 투영"""
    print("📍 점들을 스켈레톤에 투영 중...")
    
    projected_points = []
    
    for idx, point_row in points_gdf.iterrows():
        point = point_row.geometry
        point_id = point_row['id']
        
        min_distance = float('inf')
        best_projection = None
        
        # 모든 스켈레톤 라인에 대해 가장 가까운 점 찾기
        for line in skeleton_lines:
            try:
                proj_point = line.interpolate(line.project(point))
                distance = point.distance(proj_point)
                
                if distance < min_distance:
                    min_distance = distance
                    best_projection = proj_point
            except:
                continue
        
        if best_projection:
            projected_points.append({
                'id': point_id,
                'original': (point.x, point.y),
                'projected': (best_projection.x, best_projection.y),
                'distance_to_skeleton': min_distance
            })
    
    print(f"   {len(projected_points)}개 점 투영 완료")
    return projected_points

def find_skeleton_connections(projected_points, network_graph, max_path_length=200.0):
    """네트워크 그래프를 사용하여 점간 연결 찾기"""
    print("🔍 점간 연결성 분석 중...")
    
    connections = []
    
    # 각 투영점에 가장 가까운 네트워크 노드 찾기
    point_to_node = {}
    
    for proj_point in projected_points:
        point_id = proj_point['id']
        proj_coord = proj_point['projected']
        
        min_distance = float('inf')
        closest_node = None
        
        for node_id, node_data in network_graph.nodes(data=True):
            node_coord = node_data['coord']
            distance = np.sqrt((proj_coord[0] - node_coord[0])**2 + (proj_coord[1] - node_coord[1])**2)
            
            if distance < min_distance:
                min_distance = distance
                closest_node = node_id
        
        if closest_node is not None:
            point_to_node[point_id] = closest_node
    
    print(f"   {len(point_to_node)}개 점이 네트워크 노드에 매핑됨")
    
    # 모든 점 쌍에 대해 경로 찾기
    point_ids = list(point_to_node.keys())
    
    for i, point1_id in enumerate(point_ids):
        for j, point2_id in enumerate(point_ids):
            if i >= j:
                continue
            
            node1 = point_to_node[point1_id]
            node2 = point_to_node[point2_id]
            
            try:
                # 최단 경로 찾기 (연결성 확인용)
                path = nx.shortest_path(network_graph, node1, node2, weight='weight')
                skeleton_path_length = nx.shortest_path_length(network_graph, node1, node2, weight='weight')
                
                if skeleton_path_length <= max_path_length:
                    # 경로상의 좌표들 수집 (시각화용)
                    path_coords = []
                    for node_id in path:
                        coord = network_graph.nodes[node_id]['coord']
                        path_coords.append(coord)
                    
                    # 중간에 다른 점이 있는지 확인
                    has_intermediate = False
                    for other_point_id in point_ids:
                        if other_point_id in [point1_id, point2_id]:
                            continue
                        
                        if other_point_id in point_to_node:
                            other_node = point_to_node[other_point_id]
                            if other_node in path[1:-1]:  # 시작점과 끝점 제외
                                has_intermediate = True
                                break
                    
                    # 실제 점들의 원본 좌표로 유클리드 거리 계산
                    point1_orig = next(p['original'] for p in projected_points if p['id'] == point1_id)
                    point2_orig = next(p['original'] for p in projected_points if p['id'] == point2_id)
                    euclidean_distance = np.sqrt((point1_orig[0] - point2_orig[0])**2 + (point1_orig[1] - point2_orig[1])**2)
                    
                    connections.append({
                        'point1_id': point1_id,
                        'point2_id': point2_id,
                        'skeleton_path_length': skeleton_path_length,  # 스켈레톤 경로 거리 (참고용)
                        'euclidean_distance': euclidean_distance,      # 실제 사용할 직선 거리
                        'path_coords': path_coords,
                        'has_intermediate': has_intermediate,
                        'direct_connection': not has_intermediate,
                        'point1_coord': point1_orig,
                        'point2_coord': point2_orig
                    })
                
            except nx.NetworkXNoPath:
                # 연결된 경로가 없음
                continue
    
    print(f"   {len(connections)}개 연결 후보 발견")
    
    # 직접 연결만 필터링
    direct_connections = [conn for conn in connections if conn['direct_connection']]
    print(f"   {len(direct_connections)}개 직접 연결")
    
    return connections, direct_connections

def visualize_results(points_gdf, road_gdf, skeleton_lines, projected_points, connections, direct_connections):
    """결과 시각화"""
    print("🎨 결과 시각화 중...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # 1. 원본 데이터
    ax1.set_title("1. Original Data (Points + Roads)", fontsize=14, fontweight='bold')
    road_gdf.plot(ax=ax1, color='lightgray', edgecolor='gray', alpha=0.7)
    points_gdf.plot(ax=ax1, color='red', markersize=100, alpha=0.8)
    
    for idx, point_row in points_gdf.iterrows():
        ax1.annotate(f'P{point_row["id"]}', 
                    (point_row.geometry.x, point_row.geometry.y),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=12, fontweight='bold', color='darkred')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    
    # 2. 스켈레톤 네트워크
    ax2.set_title("2. Skeleton Network", fontsize=14, fontweight='bold')
    road_gdf.plot(ax=ax2, color='lightgray', edgecolor='gray', alpha=0.3)
    
    for line in skeleton_lines:
        x, y = line.xy
        ax2.plot(x, y, 'blue', linewidth=1.5, alpha=0.7)
    
    points_gdf.plot(ax=ax2, color='red', markersize=100, alpha=0.8)
    for idx, point_row in points_gdf.iterrows():
        ax2.annotate(f'P{point_row["id"]}', 
                    (point_row.geometry.x, point_row.geometry.y),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=12, fontweight='bold', color='darkred')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    
    # 3. 투영된 점들
    ax3.set_title("3. Points Projected to Skeleton", fontsize=14, fontweight='bold')
    road_gdf.plot(ax=ax3, color='lightgray', edgecolor='gray', alpha=0.3)
    
    for line in skeleton_lines:
        x, y = line.xy
        ax3.plot(x, y, 'blue', linewidth=1.5, alpha=0.7)
    
    # 원본 점들과 투영된 점들
    for proj in projected_points:
        orig = proj['original']
        projected = proj['projected']
        
        # 원본 점
        ax3.scatter(orig[0], orig[1], c='red', s=100, alpha=0.8, zorder=5)
        ax3.annotate(f'P{proj["id"]}', orig, xytext=(5, 5), textcoords='offset points',
                    fontsize=12, fontweight='bold', color='darkred')
        
        # 투영된 점
        ax3.scatter(projected[0], projected[1], c='green', s=80, alpha=0.8, zorder=5, marker='^')
        
        # 연결선
        ax3.plot([orig[0], projected[0]], [orig[1], projected[1]], 'gray', linestyle='--', alpha=0.5)
    
         ax3.set_aspect('equal')
     ax3.grid(True, alpha=0.3)
     
     # 4. 최종 연결 결과 (유클리드 직선 거리)
     ax4.set_title("4. Final Connections (Euclidean Distance)", fontsize=14, fontweight='bold')
     road_gdf.plot(ax=ax4, color='lightgray', edgecolor='gray', alpha=0.3)
     
     for line in skeleton_lines:
         x, y = line.xy
         ax4.plot(x, y, 'blue', linewidth=1, alpha=0.4)
     
     # 점들 표시
     points_gdf.plot(ax=ax4, color='red', markersize=100, alpha=0.8)
     for idx, point_row in points_gdf.iterrows():
         ax4.annotate(f'P{point_row["id"]}', 
                     (point_row.geometry.x, point_row.geometry.y),
                     xytext=(5, 5), textcoords='offset points',
                     fontsize=12, fontweight='bold', color='darkred')
     
     # 직접 연결 표시 (직선으로)
     for conn in direct_connections:
         point1_coord = conn['point1_coord']
         point2_coord = conn['point2_coord']
         
         # 직선 연결선 그리기
         ax4.plot([point1_coord[0], point2_coord[0]], 
                 [point1_coord[1], point2_coord[1]], 
                 'green', linewidth=3, alpha=0.8, linestyle='-')
         
         # 거리 표시 (중점에)
         mid_x = (point1_coord[0] + point2_coord[0]) / 2
         mid_y = (point1_coord[1] + point2_coord[1]) / 2
         ax4.annotate(f'{conn["euclidean_distance"]:.1f}m', 
                     (mid_x, mid_y), xytext=(0, 10), textcoords='offset points',
                     fontsize=10, ha='center', bbox=dict(boxstyle='round,pad=0.3', 
                     facecolor='white', alpha=0.8))
    
    ax4.set_aspect('equal')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sample_connectivity_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 결과 요약 출력
    print("\n" + "="*60)
    print("📊 분석 결과 요약")
    print("="*60)
    print(f"총 점 개수: {len(points_gdf)}")
    print(f"총 연결 후보: {len(connections)}")
    print(f"직접 연결: {len(direct_connections)}")
    print()
    
    if direct_connections:
        print("✅ 직접 연결된 점 쌍들:")
        for conn in direct_connections:
            print(f"   P{conn['point1_id']} ↔ P{conn['point2_id']}: {conn['euclidean_distance']:.1f}m")
    else:
        print("❌ 직접 연결된 점 쌍이 없습니다")
    
    print()
    blocked_connections = [conn for conn in connections if not conn['direct_connection']]
    if blocked_connections:
        print("🚫 중간점으로 인해 차단된 연결들:")
        for conn in blocked_connections:
            print(f"   P{conn['point1_id']} - P{conn['point2_id']}: {conn['skeleton_path_length']:.1f}m (중간점 존재)")

def save_results(connections, direct_connections, projected_points):
    """결과를 JSON 파일로 저장"""
    results = {
        'total_connections': len(connections),
        'direct_connections': len(direct_connections),
        'projected_points': projected_points,
        'all_connections': connections,
        'direct_connections_only': direct_connections
    }
    
    with open('sample_connectivity_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"💾 결과가 'sample_connectivity_results.json'에 저장되었습니다")

def main():
    """메인 실행 함수"""
    print("🚀 점간 연결성 분석 시작")
    print("="*60)
    
    try:
        # 1. 데이터 로드
        points_gdf, road_gdf = load_data()
        
        # 2. 스켈레톤 추출
        skeleton_lines, road_union = extract_skeleton_network(road_gdf)
        
        # 3. 네트워크 그래프 구성
        network_graph = build_network_graph(skeleton_lines)
        
        # 4. 점들을 스켈레톤에 투영
        projected_points = project_points_to_skeleton(points_gdf, skeleton_lines)
        
        # 5. 연결성 분석
        connections, direct_connections = find_skeleton_connections(projected_points, network_graph)
        
        # 6. 시각화
        visualize_results(points_gdf, road_gdf, skeleton_lines, projected_points, connections, direct_connections)
        
        # 7. 결과 저장
        save_results(connections, direct_connections, projected_points)
        
        print("\n🎉 분석 완료!")
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 