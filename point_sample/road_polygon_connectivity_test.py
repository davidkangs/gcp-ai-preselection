#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
도로망 폴리곤 기반 점간 연결성 분석
목적: 도로 폴리곤 내에서 직선으로 연결 가능한 점들만 찾기
"""

import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString
from shapely.ops import unary_union
import json

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def load_data():
    """테스트 데이터 로드"""
    print("📂 데이터 로딩 중...")
    
    points_gdf = gpd.read_file("p.geojson")
    road_gdf = gpd.read_file("road.geojson")
    
    # 도로 폴리곤 통합
    road_union = unary_union(road_gdf.geometry)
    
    print(f"   점 데이터: {len(points_gdf)}개")
    print(f"   도로 데이터: {len(road_gdf)}개 폴리곤 → 통합됨")
    
    return points_gdf, road_gdf, road_union

def find_direct_connections(points_gdf, road_union):
    """도로 폴리곤 내에서 직접 연결 가능한 점들 찾기"""
    print("🔍 직접 연결 분석 중...")
    
    connections = []
    all_points = [(row['id'], row.geometry.x, row.geometry.y) for _, row in points_gdf.iterrows()]
    
    for i, (id1, x1, y1) in enumerate(all_points):
        for j, (id2, x2, y2) in enumerate(all_points):
            if i >= j:
                continue
            
            # 1. 두 점을 직선으로 연결
            line = LineString([(x1, y1), (x2, y2)])
            
            # 2. 직선이 도로 폴리곤 안에 포함되는지 확인
            line_in_road = road_union.contains(line)
            
            # 3. 직선 상에 다른 점이 있는지 확인
            has_intermediate_point = False
            
            for k, (id3, x3, y3) in enumerate(all_points):
                if id3 in [id1, id2]:  # 시작점, 끝점은 제외
                    continue
                
                # 점과 직선 사이의 거리 계산
                point = Point(x3, y3)
                distance_to_line = point.distance(line)
                
                # 매우 가까우면 (1m 이내) 직선 상에 있다고 판단
                if distance_to_line < 1.0:
                    # 점이 직선의 양 끝점 사이에 있는지 확인 (projection 사용)
                    projection_param = line.project(point) / line.length
                    if 0.01 < projection_param < 0.99:  # 양 끝을 제외한 중간 부분
                        has_intermediate_point = True
                        break
            
            # 4. 연결 조건: 도로 안에 있고 + 중간점이 없어야 함
            if line_in_road and not has_intermediate_point:
                euclidean_distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                connections.append({
                    'point1_id': id1,
                    'point2_id': id2,
                    'point1_coord': (x1, y1),
                    'point2_coord': (x2, y2),
                    'euclidean_distance': euclidean_distance,
                    'line_in_road': line_in_road,
                    'has_intermediate': has_intermediate_point
                })
    
    print(f"   {len(connections)}개 직접 연결 발견")
    return connections

def visualize_results(points_gdf, road_gdf, road_union, connections):
    """결과 시각화"""
    print("🎨 결과 시각화 중...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # 1. 원본 데이터 + 모든 가능한 직선들
    ax1.set_title("1. All Possible Lines vs Road Polygon", fontsize=14, fontweight='bold')
    
    # 도로 폴리곤 표시
    if hasattr(road_union, 'geoms'):
        for geom in road_union.geoms:
            x, y = geom.exterior.xy
            ax1.plot(x, y, color='gray', alpha=0.7)
            ax1.fill(x, y, color='lightgray', alpha=0.3)
    else:
        x, y = road_union.exterior.xy
        ax1.plot(x, y, color='gray', alpha=0.7)
        ax1.fill(x, y, color='lightgray', alpha=0.3)
    
    # 점들 표시
    points_gdf.plot(ax=ax1, color='red', markersize=100, alpha=0.8)
    for idx, point_row in points_gdf.iterrows():
        ax1.annotate(f'P{point_row["id"]}', 
                    (point_row.geometry.x, point_row.geometry.y),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=12, fontweight='bold', color='darkred')
    
    # 모든 가능한 직선들 표시 (회색 점선)
    all_points = [(row.geometry.x, row.geometry.y) for _, row in points_gdf.iterrows()]
    for i, (x1, y1) in enumerate(all_points):
        for j, (x2, y2) in enumerate(all_points):
            if i >= j:
                continue
            ax1.plot([x1, x2], [y1, y2], 'gray', linestyle='--', alpha=0.3, linewidth=1)
    
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    
    # 2. 도로 내 직접 연결 결과
    ax2.set_title("2. Direct Connections Within Road", fontsize=14, fontweight='bold')
    
    # 도로 폴리곤 표시
    if hasattr(road_union, 'geoms'):
        for geom in road_union.geoms:
            x, y = geom.exterior.xy
            ax2.plot(x, y, color='gray', alpha=0.7)
            ax2.fill(x, y, color='lightgray', alpha=0.3)
    else:
        x, y = road_union.exterior.xy
        ax2.plot(x, y, color='gray', alpha=0.7)
        ax2.fill(x, y, color='lightgray', alpha=0.3)
    
    # 점들 표시
    points_gdf.plot(ax=ax2, color='red', markersize=100, alpha=0.8)
    for idx, point_row in points_gdf.iterrows():
        ax2.annotate(f'P{point_row["id"]}', 
                    (point_row.geometry.x, point_row.geometry.y),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=12, fontweight='bold', color='darkred')
    
    # 유효한 연결들만 표시 (녹색 실선)
    for conn in connections:
        x1, y1 = conn['point1_coord']
        x2, y2 = conn['point2_coord']
        
        # 직선 연결선 그리기
        ax2.plot([x1, x2], [y1, y2], 'green', linewidth=3, alpha=0.8)
        
        # 거리 표시
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        ax2.annotate(f'{conn["euclidean_distance"]:.1f}m', 
                    (mid_x, mid_y), xytext=(0, 10), textcoords='offset points',
                    fontsize=10, ha='center', bbox=dict(boxstyle='round,pad=0.3', 
                    facecolor='white', alpha=0.8))
    
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    
    plt.tight_layout()
    plt.savefig('road_polygon_connectivity.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 결과 요약 출력
    print("\n" + "="*60)
    print("📊 도로 폴리곤 기반 연결성 분석 결과")
    print("="*60)
    print(f"총 점 개수: {len(points_gdf)}")
    print(f"직접 연결: {len(connections)}개")
    print()
    
    if connections:
        print("✅ 도로 내에서 직접 연결된 점 쌍들:")
        for conn in connections:
            print(f"   P{conn['point1_id']} ↔ P{conn['point2_id']}: {conn['euclidean_distance']:.1f}m")
    else:
        print("❌ 직접 연결된 점 쌍이 없습니다")

def check_all_connections(points_gdf, road_union):
    """모든 연결 가능성을 상세히 체크"""
    print("\n🔍 모든 점 쌍 상세 분석:")
    print("-" * 50)
    
    all_points = [(row['id'], row.geometry.x, row.geometry.y) for _, row in points_gdf.iterrows()]
    
    for i, (id1, x1, y1) in enumerate(all_points):
        for j, (id2, x2, y2) in enumerate(all_points):
            if i >= j:
                continue
            
            line = LineString([(x1, y1), (x2, y2)])
            line_in_road = road_union.contains(line)
            
            # 중간점 체크
            intermediate_points = []
            for k, (id3, x3, y3) in enumerate(all_points):
                if id3 in [id1, id2]:
                    continue
                
                point = Point(x3, y3)
                distance_to_line = point.distance(line)
                
                if distance_to_line < 1.0:
                    projection_param = line.project(point) / line.length
                    if 0.01 < projection_param < 0.99:
                        intermediate_points.append(f"P{id3}")
            
            has_intermediate = len(intermediate_points) > 0
            euclidean_dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            
            status = "✅" if (line_in_road and not has_intermediate) else "❌"
            reason = ""
            if not line_in_road:
                reason += "도로 밖 "
            if has_intermediate:
                reason += f"중간점({','.join(intermediate_points)}) "
            
            print(f"{status} P{id1}-P{id2}: {euclidean_dist:.1f}m {reason}")

def save_results(connections):
    """결과 저장"""
    results = {
        'method': 'road_polygon_based',
        'total_connections': len(connections),
        'connections': connections
    }
    
    with open('road_polygon_connectivity_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 결과가 'road_polygon_connectivity_results.json'에 저장되었습니다")

def main():
    """메인 실행 함수"""
    print("🚀 도로 폴리곤 기반 점간 연결성 분석 시작")
    print("="*60)
    
    try:
        # 1. 데이터 로드
        points_gdf, road_gdf, road_union = load_data()
        
        # 2. 직접 연결 분석
        connections = find_direct_connections(points_gdf, road_union)
        
        # 3. 시각화
        visualize_results(points_gdf, road_gdf, road_union, connections)
        
        # 4. 상세 분석
        check_all_connections(points_gdf, road_union)
        
        # 5. 결과 저장
        save_results(connections)
        
        print("\n🎉 분석 완료!")
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 