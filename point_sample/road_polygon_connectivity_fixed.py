#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
도로망 폴리곤 기반 점간 연결성 분석 (개선버전)
- 개별 도로 폴리곤 유지 (convex hull 방지)
- 3m 버퍼 적용으로 현실적인 연결 판단
"""

import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString
import json

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def load_data():
    """테스트 데이터 로드"""
    print("📂 데이터 로딩 중...")
    
    points_gdf = gpd.read_file("p.geojson")
    road_gdf = gpd.read_file("road.geojson")
    
    print(f"   점 데이터: {len(points_gdf)}개")
    print(f"   도로 데이터: {len(road_gdf)}개 폴리곤 (개별 유지)")
    
    return points_gdf, road_gdf

def is_line_in_roads(line, road_gdf, buffer_distance=3.0):
    """선이 도로망(버퍼 적용) 내에 있는지 확인"""
    
    for _, road_row in road_gdf.iterrows():
        road_polygon = road_row.geometry
        
        # 3m 버퍼 적용
        buffered_road = road_polygon.buffer(buffer_distance)
        
        # 선이 버퍼된 도로와 교차하는지 확인
        if buffered_road.contains(line) or buffered_road.intersects(line):
            # 추가 조건: 선의 상당 부분이 버퍼 내에 있어야 함
            intersection = buffered_road.intersection(line)
            if hasattr(intersection, 'length'):
                overlap_ratio = intersection.length / line.length
                if overlap_ratio >= 0.8:  # 80% 이상 겹치면 도로 내로 판단
                    return True
    
    return False

def find_direct_connections(points_gdf, road_gdf):
    """도로 폴리곤 내에서 직접 연결 가능한 점들 찾기"""
    print("🔍 직접 연결 분석 중... (개별 도로 + 3m 버퍼)")
    
    connections = []
    all_points = [(row['id'], row.geometry.x, row.geometry.y) for _, row in points_gdf.iterrows()]
    
    for i, (id1, x1, y1) in enumerate(all_points):
        for j, (id2, x2, y2) in enumerate(all_points):
            if i >= j:
                continue
            
            # 1. 두 점을 직선으로 연결
            line = LineString([(x1, y1), (x2, y2)])
            
            # 2. 직선이 도로망(3m 버퍼) 내에 있는지 확인
            line_in_road = is_line_in_roads(line, road_gdf, buffer_distance=3.0)
            
            # 3. 직선 상에 다른 점이 있는지 확인
            has_intermediate_point = False
            intermediate_points = []
            
            for k, (id3, x3, y3) in enumerate(all_points):
                if id3 in [id1, id2]:  # 시작점, 끝점은 제외
                    continue
                
                # 점과 직선 사이의 거리 계산
                point = Point(x3, y3)
                distance_to_line = point.distance(line)
                
                # 매우 가까우면 (1m 이내) 직선 상에 있다고 판단
                if distance_to_line < 1.0:
                    # 점이 직선의 양 끝점 사이에 있는지 확인
                    projection_param = line.project(point) / line.length
                    if 0.01 < projection_param < 0.99:  # 양 끝을 제외한 중간 부분
                        has_intermediate_point = True
                        intermediate_points.append(id3)
            
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
                    'has_intermediate': has_intermediate_point,
                    'intermediate_points': intermediate_points
                })
    
    print(f"   {len(connections)}개 직접 연결 발견")
    return connections

def visualize_results(points_gdf, road_gdf, connections):
    """결과 시각화"""
    print("🎨 결과 시각화 중...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # 1. 개별 도로 폴리곤들 + 3m 버퍼
    ax1.set_title("1. Individual Road Polygons + 3m Buffer", fontsize=14, fontweight='bold')
    
    # 원본 도로 폴리곤들 (회색)
    road_gdf.plot(ax=ax1, color='lightgray', edgecolor='gray', alpha=0.5)
    
    # 3m 버퍼된 도로들 (연한 파란색)
    buffered_roads = road_gdf.geometry.buffer(3.0)
    buffered_gdf = gpd.GeoDataFrame(geometry=buffered_roads)
    buffered_gdf.plot(ax=ax1, color='lightblue', alpha=0.3, edgecolor='blue')
    
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
            ax1.plot([x1, x2], [y1, y2], 'gray', linestyle='--', alpha=0.2, linewidth=1)
    
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.legend(['Original Roads', 'Road + 3m Buffer', 'Points', 'All Possible Lines'])
    
    # 2. 최종 연결 결과
    ax2.set_title("2. Valid Connections (Road + Buffer + No Intermediate)", fontsize=14, fontweight='bold')
    
    # 원본 도로 폴리곤들
    road_gdf.plot(ax=ax2, color='lightgray', edgecolor='gray', alpha=0.5)
    
    # 3m 버퍼된 도로들 (연한 색)
    buffered_gdf.plot(ax=ax2, color='lightblue', alpha=0.2, edgecolor='blue', linestyle='--')
    
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
        ax2.plot([x1, x2], [y1, y2], 'green', linewidth=4, alpha=0.8)
        
        # 거리 표시
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        ax2.annotate(f'{conn["euclidean_distance"]:.1f}m', 
                    (mid_x, mid_y), xytext=(0, 10), textcoords='offset points',
                    fontsize=11, ha='center', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))
    
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.legend(['Original Roads', 'Road + 3m Buffer', 'Points', 'Valid Connections'])
    
    plt.tight_layout()
    plt.savefig('road_polygon_connectivity_fixed.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 결과 요약 출력
    print("\n" + "="*60)
    print("📊 개선된 도로 폴리곤 기반 연결성 분석 결과")
    print("="*60)
    print(f"총 점 개수: {len(points_gdf)}")
    print(f"개별 도로 폴리곤: {len(road_gdf)}개")
    print(f"직접 연결: {len(connections)}개")
    print()
    
    if connections:
        print("✅ 도로(+3m버퍼) 내에서 직접 연결된 점 쌍들:")
        for conn in connections:
            print(f"   P{conn['point1_id']} ↔ P{conn['point2_id']}: {conn['euclidean_distance']:.1f}m")
    else:
        print("❌ 직접 연결된 점 쌍이 없습니다")

def check_all_connections_detailed(points_gdf, road_gdf):
    """모든 연결 가능성을 상세히 체크"""
    print("\n🔍 모든 점 쌍 상세 분석 (개별 도로 + 3m 버퍼):")
    print("-" * 70)
    
    all_points = [(row['id'], row.geometry.x, row.geometry.y) for _, row in points_gdf.iterrows()]
    
    for i, (id1, x1, y1) in enumerate(all_points):
        for j, (id2, x2, y2) in enumerate(all_points):
            if i >= j:
                continue
            
            line = LineString([(x1, y1), (x2, y2)])
            line_in_road = is_line_in_roads(line, road_gdf, buffer_distance=3.0)
            
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
                reason += "도로+버퍼 밖 "
            if has_intermediate:
                reason += f"중간점({','.join(intermediate_points)}) "
            
            print(f"{status} P{id1}-P{id2}: {euclidean_dist:.1f}m {reason}")

def save_results(connections):
    """결과 저장"""
    results = {
        'method': 'individual_road_polygons_with_3m_buffer',
        'total_connections': len(connections),
        'connections': connections
    }
    
    with open('road_polygon_connectivity_fixed_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 결과가 'road_polygon_connectivity_fixed_results.json'에 저장되었습니다")

def main():
    """메인 실행 함수"""
    print("🚀 개선된 도로 폴리곤 기반 점간 연결성 분석 시작")
    print("   ✨ 개별 도로 유지 + 3m 버퍼 적용")
    print("="*60)
    
    try:
        # 1. 데이터 로드
        points_gdf, road_gdf = load_data()
        
        # 2. 직접 연결 분석
        connections = find_direct_connections(points_gdf, road_gdf)
        
        # 3. 시각화
        visualize_results(points_gdf, road_gdf, connections)
        
        # 4. 상세 분석
        check_all_connections_detailed(points_gdf, road_gdf)
        
        # 5. 결과 저장
        save_results(connections)
        
        print("\n🎉 분석 완료!")
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 