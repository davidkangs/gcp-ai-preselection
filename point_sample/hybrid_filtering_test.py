#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
다층 필터링 하이브리드 시스템 테스트 스크립트
기존 방법과 새로운 방법을 조화시킨 4단계 필터링 시스템
"""

import geopandas as gpd
import numpy as np
from shapely.geometry import Point, LineString
from shapely.ops import unary_union, nearest_points
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist


def load_data():
    print("📁 데이터 로드 중...")
    points_gdf = gpd.read_file("p.geojson")
    road_gdf = gpd.read_file("road.geojson")
    return points_gdf, road_gdf


def calculate_distance(point1, point2):
    """두 점 간의 유클리드 거리 계산"""
    return np.linalg.norm(np.array(point1) - np.array(point2))


def filter_1_distance_clustering(points_gdf, cluster_radius=20):
    """
    1단계: 거리 필터 (기존 방법)
    20m 반경 내 점들을 클러스터링
    """
    print("\n🔍 1단계: 거리 필터 (20m 반경 클러스터링)")
    
    # 점 좌표 추출
    coords = np.array([(row.geometry.x, row.geometry.y) for _, row in points_gdf.iterrows()])
    point_ids = [row['id'] for _, row in points_gdf.iterrows()]
    
    # DBSCAN으로 클러스터링 (eps=20m)
    clustering = DBSCAN(eps=cluster_radius, min_samples=1).fit(coords)
    labels = clustering.labels_
    
    # 클러스터별 점들 그룹화
    clusters = {}
    for i, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(point_ids[i])
    
    print(f"  📊 클러스터 개수: {len(clusters)}")
    for cluster_id, points in clusters.items():
        print(f"  클러스터 {cluster_id}: {points}")
    
    return clusters, coords, point_ids


def filter_2_connectivity_analysis(points_gdf, clusters, coords, point_ids):
    """
    2단계: 연결성 필터 (새로운 방법)
    각 클러스터 내에서 네트워크 연결성 분석
    """
    print("\n🔗 2단계: 연결성 필터 (네트워크 그래프 분석)")
    
    # 도로 경계선 추출
    road_gdf, _ = load_data()
    road_union = unary_union(road_gdf.geometry)
    
    # 도로 경계선의 모든 vertex 추출
    lines = []
    if hasattr(road_union, 'geoms'):
        for poly in road_union.geoms:
            boundary = poly.boundary
            if hasattr(boundary, 'geoms'):
                for line in boundary.geoms:
                    lines.append(line)
            else:
                lines.append(boundary)
    else:
        boundary = road_union.boundary
        if hasattr(boundary, 'geoms'):
            for line in boundary.geoms:
                lines.append(line)
        else:
            lines.append(boundary)
    
    connectivity_scores = {}
    
    for cluster_id, cluster_points in clusters.items():
        print(f"\n  📋 클러스터 {cluster_id} ({cluster_points}) 연결성 분석:")
        
        # 클러스터 내 점들의 연결성 분석
        cluster_connectivity = {}
        for point_id in cluster_points:
            point_idx = point_ids.index(point_id)
            point_coord = coords[point_idx]
            
            # 도로 경계선과의 연결성 계산
            connectivity_score = 0
            for line in lines:
                # 점이 도로 경계선에 얼마나 가까운지
                distance_to_line = line.distance(Point(point_coord))
                if distance_to_line < 10:  # 10m 이내면 연결된 것으로 간주
                    connectivity_score += 1 / (1 + distance_to_line)
            
            cluster_connectivity[point_id] = connectivity_score
            print(f"    {point_id}번 점: 연결성 점수 = {connectivity_score:.3f}")
        
        connectivity_scores[cluster_id] = cluster_connectivity
    
    return connectivity_scores


def filter_3_importance_calculation(points_gdf, clusters, coords, point_ids):
    """
    3단계: 중요도 필터 (기존 방법)
    도로 폭, 스켈레톤 밀도 기반 중요도 계산
    """
    print("\n⭐ 3단계: 중요도 필터 (도로 중요도 계산)")
    
    # 도로 데이터 로드
    road_gdf, _ = load_data()
    
    importance_scores = {}
    
    for cluster_id, cluster_points in clusters.items():
        print(f"\n  📋 클러스터 {cluster_id} ({cluster_points}) 중요도 분석:")
        
        cluster_importance = {}
        for point_id in cluster_points:
            point_idx = point_ids.index(point_id)
            point_coord = coords[point_idx]
            point_geom = Point(point_coord)
            
            # 도로 폭 기반 중요도
            road_width_importance = 0
            # 스켈레톤 밀도 기반 중요도 (간단한 근사)
            skeleton_density_importance = 0
            
            for _, road_row in road_gdf.iterrows():
                road_geom = road_row.geometry
                distance_to_road = point_geom.distance(road_geom)
                
                if distance_to_road < 50:  # 50m 이내 도로만 고려
                    # 도로 폭 (면적/길이로 근사)
                    if hasattr(road_geom, 'area') and hasattr(road_geom, 'length'):
                        road_width = road_geom.area / road_geom.length if road_geom.length > 0 else 0
                        road_width_importance += road_width / (1 + distance_to_road)
                    
                    # 스켈레톤 밀도 (도로 경계선 복잡도로 근사)
                    if hasattr(road_geom, 'boundary'):
                        boundary = road_geom.boundary
                        try:
                            # 도로 경계선 복잡도 계산
                            if hasattr(boundary, 'geoms'):
                                # MultiLineString인 경우
                                complexity = sum(len(list(line.coords)) for line in boundary.geoms)
                            elif hasattr(boundary, 'coords'):
                                # LineString인 경우
                                complexity = len(list(boundary.coords))
                            else:
                                complexity = 10  # 기본값
                            skeleton_density_importance += complexity / (1 + distance_to_road)
                        except Exception:
                            # 좌표 추출 실패 시 기본값
                            skeleton_density_importance += 10 / (1 + distance_to_road)
            
            # 종합 중요도 점수
            total_importance = road_width_importance + skeleton_density_importance
            cluster_importance[point_id] = total_importance
            
            print(f"    {point_id}번 점: 중요도 점수 = {total_importance:.3f}")
        
        importance_scores[cluster_id] = cluster_importance
    
    return importance_scores


def filter_4_role_priority(points_gdf, clusters, coords, point_ids):
    """
    4단계: 역할 필터 (새로운 방법)
    끝점, 중간점, 분기점 우선순위 적용
    """
    print("\n🎭 4단계: 역할 필터 (점 역할 기반 우선순위)")
    
    # 수동 연결 관계 정의 (테스트용)
    connections = [
        (1, 4), (4, 5), (4, 6), (6, 7), (7, 8)
    ]
    
    # 네트워크 그래프 생성
    G = nx.Graph()
    for n1, n2 in connections:
        G.add_edge(n1, n2)
    
    role_scores = {}
    
    for cluster_id, cluster_points in clusters.items():
        print(f"\n  📋 클러스터 {cluster_id} ({cluster_points}) 역할 분석:")
        
        cluster_roles = {}
        for point_id in cluster_points:
            if point_id in G.nodes:
                degree = int(G.degree(point_id))
                if degree == 1:
                    role = '끝점'
                    role_score = 1.0
                elif degree == 2:
                    role = '중간점'
                    role_score = 2.0
                elif degree >= 3:
                    role = '분기점/교차점'
                    role_score = 3.0
                else:
                    role = '고립점'
                    role_score = 0.0
            else:
                role = '미연결'
                role_score = 0.5
            
            cluster_roles[point_id] = role_score
            print(f"    {point_id}번 점: 역할 = {role}, 점수 = {role_score}")
        
        role_scores[cluster_id] = cluster_roles
    
    return role_scores


def apply_hybrid_filtering(connectivity_scores, importance_scores, role_scores, clusters):
    """
    다층 필터링 결과를 종합하여 최적 점 선택
    """
    print("\n🎯 다층 필터링 결과 종합 및 최적 점 선택")
    
    # 가중치 설정
    weights = {
        'connectivity': 0.3,
        'importance': 0.3,
        'role': 0.4
    }
    
    final_selections = {}
    
    for cluster_id in clusters.keys():
        print(f"\n  📋 클러스터 {cluster_id} 최종 점수:")
        
        cluster_points = clusters[cluster_id]
        final_scores = {}
        
        for point_id in cluster_points:
            # 각 필터 점수 정규화 (0-1 범위)
            connectivity_score = connectivity_scores[cluster_id].get(point_id, 0)
            importance_score = importance_scores[cluster_id].get(point_id, 0)
            role_score = role_scores[cluster_id].get(point_id, 0)
            
            # 정규화 (각 클러스터 내에서 상대적 점수)
            max_connectivity = max(connectivity_scores[cluster_id].values()) if connectivity_scores[cluster_id] else 1
            max_importance = max(importance_scores[cluster_id].values()) if importance_scores[cluster_id] else 1
            max_role = max(role_scores[cluster_id].values()) if role_scores[cluster_id] else 1
            
            normalized_connectivity = connectivity_score / max_connectivity if max_connectivity > 0 else 0
            normalized_importance = importance_score / max_importance if max_importance > 0 else 0
            normalized_role = role_score / max_role if max_role > 0 else 0
            
            # 가중 평균 계산
            final_score = (
                weights['connectivity'] * normalized_connectivity +
                weights['importance'] * normalized_importance +
                weights['role'] * normalized_role
            )
            
            final_scores[point_id] = final_score
            
            print(f"    {point_id}번 점: 연결성={normalized_connectivity:.3f}, 중요도={normalized_importance:.3f}, 역할={normalized_role:.3f}, 최종점수={final_score:.3f}")
        
        # 최고 점수 점 선택
        best_point = max(final_scores.keys(), key=lambda k: final_scores[k])
        final_selections[cluster_id] = best_point
        
        print(f"  ✅ 클러스터 {cluster_id} 최적 점: {best_point}번 (점수: {final_scores[best_point]:.3f})")
    
    return final_selections


def visualize_results(points_gdf, clusters, final_selections):
    """결과 시각화"""
    print("\n📊 결과 시각화...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 원본 점들
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    # 왼쪽: 클러스터링 결과
    for cluster_id, cluster_points in clusters.items():
        cluster_color = colors[cluster_id % len(colors)]
        for point_id in cluster_points:
            point_data = points_gdf[points_gdf['id'] == point_id]
            if not point_data.empty:
                x, y = point_data.iloc[0].geometry.x, point_data.iloc[0].geometry.y
                ax1.scatter(x, y, c=cluster_color, s=100, alpha=0.7)
                ax1.annotate(f'{point_id}', (x, y), xytext=(5, 5), textcoords='offset points')
    
    ax1.set_title('클러스터링 결과')
    ax1.set_aspect('equal')
    
    # 오른쪽: 최종 선택 결과
    for cluster_id, selected_point in final_selections.items():
        point_data = points_gdf[points_gdf['id'] == selected_point]
        if not point_data.empty:
            x, y = point_data.iloc[0].geometry.x, point_data.iloc[0].geometry.y
            ax2.scatter(x, y, c='red', s=200, marker='*', edgecolors='black', linewidth=2)
            ax2.annotate(f'{selected_point}*', (x, y), xytext=(5, 5), textcoords='offset points', 
                        fontweight='bold', fontsize=12)
    
    ax2.set_title('최종 선택된 점들 (*)')
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('hybrid_filtering_results.png', dpi=300)
    print("✅ 시각화 결과 저장: hybrid_filtering_results.png")
    plt.show()


def main():
    print("🚀 다층 필터링 하이브리드 시스템 테스트 시작!")
    print("="*60)
    
    # 데이터 로드
    points_gdf, road_gdf = load_data()
    
    # 1단계: 거리 필터
    clusters, coords, point_ids = filter_1_distance_clustering(points_gdf)
    
    # 2단계: 연결성 필터
    connectivity_scores = filter_2_connectivity_analysis(points_gdf, clusters, coords, point_ids)
    
    # 3단계: 중요도 필터
    importance_scores = filter_3_importance_calculation(points_gdf, clusters, coords, point_ids)
    
    # 4단계: 역할 필터
    role_scores = filter_4_role_priority(points_gdf, clusters, coords, point_ids)
    
    # 다층 필터링 적용
    final_selections = apply_hybrid_filtering(connectivity_scores, importance_scores, role_scores, clusters)
    
    # 결과 시각화
    visualize_results(points_gdf, clusters, final_selections)
    
    print("\n🎉 다층 필터링 테스트 완료!")
    print("\n💡 결론:")
    print("1. 4단계 필터링으로 정확한 점 선택")
    print("2. 기존 방법과 새로운 방법의 장점 결합")
    print("3. 연결성, 중요도, 역할을 종합적으로 고려")
    print("4. 각 클러스터별 최적 점 자동 선택")


if __name__ == "__main__":
    main() 