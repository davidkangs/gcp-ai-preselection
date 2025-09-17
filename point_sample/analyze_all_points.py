#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
도로 중요도 분석 실험 스크립트 (전체 점)
1~8번 모든 점의 도로 중요도를 분석하여 어떤 점이 삭제되어야 하는지 판단
"""

import geopandas as gpd
import numpy as np
from shapely.geometry import Point, LineString
from shapely.ops import unary_union
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import networkx as nx

def load_data():
    """데이터 로드"""
    print("📁 데이터 로드 중...")
    
    # 점 데이터 로드
    points_gdf = gpd.read_file("p.geojson")
    print(f"✅ 점 데이터: {len(points_gdf)}개 점")
    
    # 도로망 데이터 로드
    road_gdf = gpd.read_file("road.geojson")
    print(f"✅ 도로망 데이터: {len(road_gdf)}개 폴리곤")
    
    return points_gdf, road_gdf

def extract_skeleton(road_gdf, sample_distance=5.0):
    """도로망에서 스켈레톤 추출"""
    print("🦴 스켈레톤 추출 중...")
    
    # 도로 폴리곤 통합
    road_union = unary_union(road_gdf.geometry)
    
    # 경계선에서 스켈레톤 점 추출
    skeleton_points = []
    
    if hasattr(road_union, 'geoms'):
        # MultiPolygon인 경우
        for polygon in road_union.geoms:
            boundary = polygon.boundary
            if hasattr(boundary, 'geoms'):
                # MultiLineString인 경우
                for line in boundary.geoms:
                    coords = list(line.coords)
                    for i in range(0, len(coords), int(sample_distance)):
                        if i < len(coords):
                            skeleton_points.append(coords[i])
            else:
                # LineString인 경우
                coords = list(boundary.coords)
                for i in range(0, len(coords), int(sample_distance)):
                    if i < len(coords):
                        skeleton_points.append(coords[i])
    else:
        # 단일 Polygon인 경우
        boundary = road_union.boundary
        if hasattr(boundary, 'geoms'):
            for line in boundary.geoms:
                coords = list(line.coords)
                for i in range(0, len(coords), int(sample_distance)):
                    if i < len(coords):
                        skeleton_points.append(coords[i])
        else:
            coords = list(boundary.coords)
            for i in range(0, len(coords), int(sample_distance)):
                if i < len(coords):
                    skeleton_points.append(coords[i])
    
    print(f"✅ 스켈레톤 점: {len(skeleton_points)}개")
    return skeleton_points

def calculate_road_importance(point, road_gdf, skeleton_points, radius=15):
    """점의 도로 중요도 계산"""
    
    # 1. 스켈레톤-폴리곤 거리 분석
    point_geom = Point(point)
    
    # 해당 점에서 가장 가까운 도로 폴리곤 찾기
    min_distance = float('inf')
    nearest_road = None
    
    for idx, road in road_gdf.iterrows():
        distance = point_geom.distance(road.geometry)
        if distance < min_distance:
            min_distance = distance
            nearest_road = road.geometry
    
    if nearest_road is None:
        return {"importance": 0, "road_width": 0, "skeleton_density": 0}
    
    # 2. 지역적 도로 폭 계산
    point_buffer = point_geom.buffer(radius)
    local_road = nearest_road.intersection(point_buffer)
    
    if local_road.is_empty:
        road_width = 0
    else:
        # 지역적 도로 폭 (면적 / 길이로 추정)
        road_width = local_road.area / max(local_road.length, 1)
    
    # 3. 스켈레톤 밀도 계산
    skeleton_tree = KDTree(skeleton_points)
    nearby_skeleton = skeleton_tree.query_ball_point(point, radius)
    skeleton_density = len(nearby_skeleton) / (np.pi * radius**2)
    
    # 4. 종합 중요도 점수
    importance = (road_width * 10) + (skeleton_density * 1000)
    
    return {
        "importance": importance,
        "road_width": road_width,
        "skeleton_density": skeleton_density,
        "min_distance_to_road": min_distance
    }

def analyze_all_points():
    """1~8번 모든 점 분석"""
    print("\n🔍 1~8번 모든 점 분석 시작...")
    
    # 데이터 로드
    points_gdf, road_gdf = load_data()
    
    # 스켈레톤 추출
    skeleton_points = extract_skeleton(road_gdf)
    
    print(f"\n📊 분석 대상 점:")
    for idx, point in points_gdf.iterrows():
        print(f"  {point['id']}번 점: ({point.geometry.x:.1f}, {point.geometry.y:.1f})")
    
    # 각 점의 중요도 계산
    results = []
    for idx, point in points_gdf.iterrows():
        coords = (point.geometry.x, point.geometry.y)
        importance_data = calculate_road_importance(coords, road_gdf, skeleton_points)
        
        results.append({
            "id": point['id'],
            "coords": coords,
            **importance_data
        })
        
        print(f"\n🎯 {point['id']}번 점 분석 결과:")
        print(f"  - 도로 폭 점수: {importance_data['road_width']:.3f}")
        print(f"  - 스켈레톤 밀도: {importance_data['skeleton_density']:.3f}")
        print(f"  - 종합 중요도: {importance_data['importance']:.1f}")
        print(f"  - 도로까지 최단거리: {importance_data['min_distance_to_road']:.1f}m")
    
    # 중요도 순으로 정렬
    sorted_results = sorted(results, key=lambda x: x['importance'], reverse=True)
    
    print(f"\n🏆 중요도 순위:")
    for i, result in enumerate(sorted_results):
        print(f"  {i+1}위: {result['id']}번 점 (중요도 {result['importance']:.1f})")
    
    # 클러스터링 시뮬레이션
    print(f"\n🎯 클러스터링 시뮬레이션 (20m 임계값):")
    
    clusters = []
    used = set()
    
    for i, p1 in enumerate(sorted_results):
        if i in used:
            continue
            
        cluster = [p1]
        used.add(i)
        
        print(f"\n📌 기준점 {p1['id']}번 (중요도 {p1['importance']:.1f}):")
        
        for j, p2 in enumerate(sorted_results):
            if j in used:
                continue
                
            dist = Point(p1['coords']).distance(Point(p2['coords']))
            
            if dist <= 20.0:
                cluster.append(p2)
                used.add(j)
                print(f"  ✅ {p2['id']}번 점 포함 (거리 {dist:.1f}m, 중요도 {p2['importance']:.1f})")
            else:
                print(f"  ❌ {p2['id']}번 점 제외 (거리 {dist:.1f}m, 중요도 {p2['importance']:.1f})")
        
        if len(cluster) > 1:
            clusters.append(cluster)
            print(f"🎯 클러스터 {len(clusters)} 생성: {len(cluster)}개 점")
        else:
            print(f"📌 단독 점: 클러스터 생성 안함")
    
    # 클러스터별 선택 결과
    print(f"\n🗑️ 클러스터별 선택 결과:")
    all_removed = []
    all_selected = []
    
    for i, cluster in enumerate(clusters):
        best_point = max(cluster, key=lambda x: x['importance'])
        removed_points = [p for p in cluster if p != best_point]
        
        all_selected.append(best_point)
        all_removed.extend(removed_points)
        
        print(f"\n클러스터 {i+1}:")
        print(f"  🏆 선택: {best_point['id']}번 점 (중요도 {best_point['importance']:.1f})")
        for removed in removed_points:
            print(f"  🗑️ 삭제: {removed['id']}번 점 (중요도 {removed['importance']:.1f})")
    
    # 단독 점들
    single_points = [p for p in sorted_results if not any(p in cluster for cluster in clusters)]
    all_selected.extend(single_points)
    
    if single_points:
        print(f"\n📌 단독 점들 (삭제 안됨):")
        for point in single_points:
            print(f"  ✅ {point['id']}번 점 (중요도 {point['importance']:.1f})")
    
    # 최종 결과 요약
    print(f"\n📋 최종 결과 요약:")
    print(f"  🏆 유지되는 점: {[p['id'] for p in all_selected]}")
    print(f"  🗑️ 삭제되는 점: {[p['id'] for p in all_removed]}")
    
    # 시각화
    visualize_results(points_gdf, road_gdf, results, clusters, all_removed)
    
    return results, clusters, all_removed

def visualize_results(points_gdf, road_gdf, results, clusters, removed_points):
    """결과 시각화"""
    print(f"\n📊 결과 시각화...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 전체 도로망과 점들
    road_gdf.plot(ax=ax1, alpha=0.3, color='gray')
    
    # 모든 점들 플롯
    points_gdf.plot(ax=ax1, color='blue', markersize=20, alpha=0.7)
    
    # 점 번호 표시
    for idx, point in points_gdf.iterrows():
        ax1.annotate(f"{point['id']}", 
                    (point.geometry.x, point.geometry.y),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    # 삭제되는 점들 강조
    removed_ids = [p['id'] for p in removed_points]
    removed_points_gdf = points_gdf[points_gdf['id'].isin(removed_ids)]
    if not removed_points_gdf.empty:
        removed_points_gdf.plot(ax=ax1, color='red', markersize=30, alpha=0.8, marker='x')
    
    ax1.set_title('전체 도로망과 분석 점들 (빨간 X = 삭제 예정)')
    ax1.set_aspect('equal')
    
    # 중요도 비교 차트
    ids = [r['id'] for r in results]
    importances = [r['importance'] for r in results]
    road_widths = [r['road_width'] for r in results]
    skeleton_densities = [r['skeleton_density'] for r in results]
    
    x = np.arange(len(ids))
    width = 0.35
    
    ax2.bar(x - width/2, road_widths, width, label='도로 폭 점수', alpha=0.7)
    ax2.bar(x + width/2, [d*100 for d in skeleton_densities], width, label='스켈레톤 밀도 (x100)', alpha=0.7)
    
    # 삭제되는 점들 강조
    for i, point_id in enumerate(ids):
        if point_id in removed_ids:
            ax2.bar(x[i] - width/2, road_widths[i], width, color='red', alpha=0.8)
            ax2.bar(x[i] + width/2, skeleton_densities[i]*100, width, color='red', alpha=0.8)
    
    ax2.set_xlabel('점 번호')
    ax2.set_ylabel('점수')
    ax2.set_title('점별 중요도 비교 (빨간색 = 삭제 예정)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(ids)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('road_importance_analysis_all.png', dpi=300, bbox_inches='tight')
    print(f"✅ 시각화 결과 저장: road_importance_analysis_all.png")
    
    plt.show()

if __name__ == "__main__":
    print("🚀 도로 중요도 분석 실험 시작! (전체 점)")
    print("=" * 50)
    
    results, clusters, removed_points = analyze_all_points()
    
    print("\n" + "=" * 50)
    print("✅ 분석 완료!")
    print("\n💡 결론:")
    print("1. 중요도가 가장 높은 점이 기준점이 됨")
    print("2. 기준점으로부터 20m 이내의 모든 점들이 같은 클러스터에 포함됨")
    print("3. 클러스터 내에서 중요도가 가장 높은 점만 유지")
    print("4. 단독 점들은 모두 유지됨") 