#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
도로 흐름과 연결성 분석 스크립트
도로의 실제 흐름, 커브, 교차점 등을 고려한 점 중요도 분석
"""

import geopandas as gpd
import numpy as np
from shapely.geometry import Point, LineString
from shapely.ops import unary_union
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import networkx as nx
from sklearn.cluster import DBSCAN

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

def extract_road_centerline(road_gdf):
    """도로 중심선 추출"""
    print("🛣️ 도로 중심선 추출 중...")
    
    # 도로 폴리곤 통합
    road_union = unary_union(road_gdf.geometry)
    
    # 중심선 추출 (단순화된 방법)
    centerlines = []
    
    if hasattr(road_union, 'geoms'):
        # MultiPolygon인 경우
        for polygon in road_union.geoms:
            # 경계선을 중심선으로 사용
            boundary = polygon.boundary
            if hasattr(boundary, 'geoms'):
                for line in boundary.geoms:
                    centerlines.append(line)
            else:
                centerlines.append(boundary)
    else:
        # 단일 Polygon인 경우
        boundary = road_union.boundary
        if hasattr(boundary, 'geoms'):
            for line in boundary.geoms:
                centerlines.append(line)
        else:
            centerlines.append(boundary)
    
    print(f"✅ 중심선: {len(centerlines)}개")
    return centerlines

def analyze_road_curvature(centerlines, points_gdf, radius=30):
    """도로 커브 분석"""
    print("🔄 도로 커브 분석 중...")
    
    curvature_scores = {}
    
    for idx, point in points_gdf.iterrows():
        point_geom = Point(point.geometry.x, point.geometry.y)
        point_buffer = point_geom.buffer(radius)
        
        # 반경 내 중심선 찾기
        nearby_lines = []
        for line in centerlines:
            if line.intersects(point_buffer):
                nearby_lines.append(line)
        
        if not nearby_lines:
            curvature_scores[point['id']] = 0
            continue
        
        # 커브 점수 계산
        curvature_score = 0
        
        for line in nearby_lines:
            coords = list(line.coords)
            if len(coords) < 3:
                continue
            
            # 연속된 3점으로 각도 변화 계산
            angles = []
            for i in range(len(coords) - 2):
                p1, p2, p3 = coords[i], coords[i+1], coords[i+2]
                
                # 벡터 계산
                v1 = np.array([p2[0] - p1[0], p2[1] - p1[1]])
                v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
                
                # 각도 계산
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = np.arccos(cos_angle)
                
                angles.append(angle)
            
            if angles:
                # 평균 각도 변화 (커브 정도)
                avg_angle_change = np.mean(angles)
                curvature_score += avg_angle_change
        
        curvature_scores[point['id']] = curvature_score
    
    return curvature_scores

def analyze_connectivity(points_gdf, centerlines, radius=50):
    """도로 연결성 분석"""
    print("🔗 도로 연결성 분석 중...")
    
    connectivity_scores = {}
    
    for idx, point in points_gdf.iterrows():
        point_geom = Point(point.geometry.x, point.geometry.y)
        point_buffer = point_geom.buffer(radius)
        
        # 반경 내 중심선 찾기
        nearby_lines = []
        for line in centerlines:
            if line.intersects(point_buffer):
                nearby_lines.append(line)
        
        if len(nearby_lines) < 2:
            connectivity_scores[point['id']] = 0
            continue
        
        # 연결성 점수 계산
        # 1. 교차점 여부 (여러 선이 만나는 지점)
        intersection_count = 0
        for i, line1 in enumerate(nearby_lines):
            for j, line2 in enumerate(nearby_lines):
                if i < j:
                    if line1.intersects(line2):
                        intersection_count += 1
        
        # 2. 선의 방향 다양성
        directions = []
        for line in nearby_lines:
            coords = list(line.coords)
            if len(coords) >= 2:
                # 선의 방향 벡터
                direction = np.array([coords[-1][0] - coords[0][0], 
                                    coords[-1][1] - coords[0][1]])
                if np.linalg.norm(direction) > 0:
                    direction = direction / np.linalg.norm(direction)
                    directions.append(direction)
        
        # 방향 다양성 계산
        direction_diversity = 0
        if len(directions) >= 2:
            for i, dir1 in enumerate(directions):
                for j, dir2 in enumerate(directions):
                    if i < j:
                        angle = np.arccos(np.clip(np.dot(dir1, dir2), -1, 1))
                        direction_diversity += angle
        
        connectivity_score = intersection_count * 10 + direction_diversity * 5
        connectivity_scores[point['id']] = connectivity_score
    
    return connectivity_scores

def analyze_traffic_flow(points_gdf, centerlines, radius=40):
    """교통 흐름 분석"""
    print("🚗 교통 흐름 분석 중...")
    
    flow_scores = {}
    
    for idx, point in points_gdf.iterrows():
        point_geom = Point(point.geometry.x, point.geometry.y)
        point_buffer = point_geom.buffer(radius)
        
        # 반경 내 중심선 찾기
        nearby_lines = []
        for line in centerlines:
            if line.intersects(point_buffer):
                nearby_lines.append(line)
        
        if not nearby_lines:
            flow_scores[point['id']] = 0
            continue
        
        # 교통 흐름 점수 계산
        flow_score = 0
        
        for line in nearby_lines:
            # 선의 길이 (교통량 추정)
            line_length = line.length
            flow_score += line_length
            
            # 선의 복잡도 (커브, 교차 등)
            coords = list(line.coords)
            if len(coords) >= 3:
                # 방향 변화 횟수
                direction_changes = 0
                for i in range(len(coords) - 2):
                    v1 = np.array([coords[i+1][0] - coords[i][0], 
                                  coords[i+1][1] - coords[i][1]])
                    v2 = np.array([coords[i+2][0] - coords[i+1][0], 
                                  coords[i+2][1] - coords[i+1][1]])
                    
                    if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                        v1 = v1 / np.linalg.norm(v1)
                        v2 = v2 / np.linalg.norm(v2)
                        angle = np.arccos(np.clip(np.dot(v1, v2), -1, 1))
                        if angle > np.pi/6:  # 30도 이상 변화
                            direction_changes += 1
                
                flow_score += direction_changes * 5
        
        flow_scores[point['id']] = flow_score
    
    return flow_scores

def calculate_comprehensive_importance(points_gdf, road_gdf, centerlines):
    """종합 중요도 계산"""
    print("🎯 종합 중요도 계산 중...")
    
    # 기존 중요도 계산 (도로 폭 + 스켈레톤 밀도)
    skeleton_points = extract_skeleton(road_gdf)
    
    results = []
    for idx, point in points_gdf.iterrows():
        coords = (point.geometry.x, point.geometry.y)
        
        # 기존 중요도
        basic_importance = calculate_basic_importance(coords, road_gdf, skeleton_points)
        
        # 새로운 분석들
        curvature_score = analyze_road_curvature(centerlines, points_gdf.iloc[[idx]], radius=30)[point['id']]
        connectivity_score = analyze_connectivity(points_gdf.iloc[[idx]], centerlines, radius=50)[point['id']]
        flow_score = analyze_traffic_flow(points_gdf.iloc[[idx]], centerlines, radius=40)[point['id']]
        
        # 종합 중요도 (가중치 조정 가능)
        comprehensive_importance = (
            basic_importance['importance'] * 0.3 +  # 기존 중요도
            curvature_score * 0.2 +                # 커브 중요도
            connectivity_score * 0.3 +             # 연결성 중요도
            flow_score * 0.2                       # 교통 흐름 중요도
        )
        
        results.append({
            "id": point['id'],
            "coords": coords,
            "basic_importance": basic_importance['importance'],
            "curvature_score": curvature_score,
            "connectivity_score": connectivity_score,
            "flow_score": flow_score,
            "comprehensive_importance": comprehensive_importance
        })
        
        print(f"\n🎯 {point['id']}번 점 종합 분석:")
        print(f"  - 기존 중요도: {basic_importance['importance']:.1f}")
        print(f"  - 커브 점수: {curvature_score:.1f}")
        print(f"  - 연결성 점수: {connectivity_score:.1f}")
        print(f"  - 교통 흐름 점수: {flow_score:.1f}")
        print(f"  - 종합 중요도: {comprehensive_importance:.1f}")
    
    return results

def extract_skeleton(road_gdf, sample_distance=5.0):
    """도로망에서 스켈레톤 추출"""
    road_union = unary_union(road_gdf.geometry)
    skeleton_points = []
    
    if hasattr(road_union, 'geoms'):
        for polygon in road_union.geoms:
            boundary = polygon.boundary
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
    else:
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
    
    return skeleton_points

def calculate_basic_importance(point, road_gdf, skeleton_points, radius=15):
    """기본 중요도 계산"""
    point_geom = Point(point)
    
    min_distance = float('inf')
    nearest_road = None
    
    for idx, road in road_gdf.iterrows():
        distance = point_geom.distance(road.geometry)
        if distance < min_distance:
            min_distance = distance
            nearest_road = road.geometry
    
    if nearest_road is None:
        return {"importance": 0, "road_width": 0, "skeleton_density": 0}
    
    point_buffer = point_geom.buffer(radius)
    local_road = nearest_road.intersection(point_buffer)
    
    if local_road.is_empty:
        road_width = 0
    else:
        road_width = local_road.area / max(local_road.length, 1)
    
    skeleton_tree = KDTree(skeleton_points)
    nearby_skeleton = skeleton_tree.query_ball_point(point, radius)
    skeleton_density = len(nearby_skeleton) / (np.pi * radius**2)
    
    importance = (road_width * 10) + (skeleton_density * 1000)
    
    return {
        "importance": importance,
        "road_width": road_width,
        "skeleton_density": skeleton_density,
        "min_distance_to_road": min_distance
    }

def analyze_all_points_with_flow():
    """도로 흐름을 고려한 전체 점 분석"""
    print("\n🔍 도로 흐름 기반 분석 시작...")
    
    # 데이터 로드
    points_gdf, road_gdf = load_data()
    
    # 중심선 추출
    centerlines = extract_road_centerline(road_gdf)
    
    # 종합 중요도 계산
    results = calculate_comprehensive_importance(points_gdf, road_gdf, centerlines)
    
    # 중요도 순으로 정렬
    sorted_results = sorted(results, key=lambda x: x['comprehensive_importance'], reverse=True)
    
    print(f"\n🏆 종합 중요도 순위:")
    for i, result in enumerate(sorted_results):
        print(f"  {i+1}위: {result['id']}번 점 (종합 중요도 {result['comprehensive_importance']:.1f})")
    
    # 클러스터링 시뮬레이션
    print(f"\n🎯 클러스터링 시뮬레이션 (20m 임계값):")
    
    clusters = []
    used = set()
    
    for i, p1 in enumerate(sorted_results):
        if i in used:
            continue
            
        cluster = [p1]
        used.add(i)
        
        print(f"\n📌 기준점 {p1['id']}번 (종합 중요도 {p1['comprehensive_importance']:.1f}):")
        
        for j, p2 in enumerate(sorted_results):
            if j in used:
                continue
                
            dist = Point(p1['coords']).distance(Point(p2['coords']))
            
            if dist <= 20.0:
                cluster.append(p2)
                used.add(j)
                print(f"  ✅ {p2['id']}번 점 포함 (거리 {dist:.1f}m, 종합 중요도 {p2['comprehensive_importance']:.1f})")
            else:
                print(f"  ❌ {p2['id']}번 점 제외 (거리 {dist:.1f}m, 종합 중요도 {p2['comprehensive_importance']:.1f})")
        
        if len(cluster) > 1:
            clusters.append(cluster)
            print(f"🎯 클러스터 {len(clusters)} 생성: {len(cluster)}개 점")
        else:
            print(f"📌 단독 점: 클러스터 생성 안함")
    
    # 결과 요약
    all_removed = []
    all_selected = []
    
    for i, cluster in enumerate(clusters):
        best_point = max(cluster, key=lambda x: x['comprehensive_importance'])
        removed_points = [p for p in cluster if p != best_point]
        
        all_selected.append(best_point)
        all_removed.extend(removed_points)
        
        print(f"\n클러스터 {i+1}:")
        print(f"  🏆 선택: {best_point['id']}번 점 (종합 중요도 {best_point['comprehensive_importance']:.1f})")
        for removed in removed_points:
            print(f"  🗑️ 삭제: {removed['id']}번 점 (종합 중요도 {removed['comprehensive_importance']:.1f})")
    
    # 단독 점들
    single_points = [p for p in sorted_results if not any(p in cluster for cluster in clusters)]
    all_selected.extend(single_points)
    
    if single_points:
        print(f"\n📌 단독 점들 (삭제 안됨):")
        for point in single_points:
            print(f"  ✅ {point['id']}번 점 (종합 중요도 {point['comprehensive_importance']:.1f})")
    
    print(f"\n📋 최종 결과 요약:")
    print(f"  🏆 유지되는 점: {[p['id'] for p in all_selected]}")
    print(f"  🗑️ 삭제되는 점: {[p['id'] for p in all_removed]}")
    
    return results, clusters, all_removed

if __name__ == "__main__":
    print("🚀 도로 흐름 기반 중요도 분석 시작!")
    print("=" * 50)
    
    results, clusters, removed_points = analyze_all_points_with_flow()
    
    print("\n" + "=" * 50)
    print("✅ 분석 완료!")
    print("\n💡 결론:")
    print("1. 도로 흐름, 커브, 연결성을 종합적으로 고려")
    print("2. 교통상 필수적인 점들도 중요도에 반영")
    print("3. 단순히 도로 폭/밀도만이 아닌 실제 교통 기능 고려") 