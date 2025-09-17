#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
커브 중요성 상세 분석 스크립트
7번 점이 왜 커브에 필수적인지 구체적으로 분석
"""

import geopandas as gpd
import numpy as np
from shapely.geometry import Point, LineString
from shapely.ops import unary_union
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

def load_data():
    """데이터 로드"""
    print("📁 데이터 로드 중...")
    
    points_gdf = gpd.read_file("p.geojson")
    road_gdf = gpd.read_file("road.geojson")
    
    return points_gdf, road_gdf

def analyze_curve_detection(road_gdf, points_gdf, radius=25):
    """커브 검출 및 점의 커브 중요성 분석"""
    print("🔄 커브 검출 및 중요성 분석 중...")
    
    # 도로 폴리곤 통합
    road_union = unary_union(road_gdf.geometry)
    
    # 경계선에서 점 추출 (더 조밀하게)
    boundary_points = []
    
    if hasattr(road_union, 'geoms'):
        for polygon in road_union.geoms:
            boundary = polygon.boundary
            if hasattr(boundary, 'geoms'):
                for line in boundary.geoms:
                    coords = list(line.coords)
                    for i in range(0, len(coords), 2):  # 더 조밀하게
                        if i < len(coords):
                            boundary_points.append(coords[i])
            else:
                coords = list(boundary.coords)
                for i in range(0, len(coords), 2):
                    if i < len(coords):
                        boundary_points.append(coords[i])
    else:
        boundary = road_union.boundary
        if hasattr(boundary, 'geoms'):
            for line in boundary.geoms:
                coords = list(line.coords)
                for i in range(0, len(coords), 2):
                    if i < len(coords):
                        boundary_points.append(coords[i])
        else:
            coords = list(boundary.coords)
            for i in range(0, len(coords), 2):
                if i < len(coords):
                    boundary_points.append(coords[i])
    
    print(f"✅ 경계선 점: {len(boundary_points)}개")
    
    # 각 점의 커브 중요성 분석
    curve_analysis = {}
    
    for idx, point in points_gdf.iterrows():
        point_geom = Point(point.geometry.x, point.geometry.y)
        point_buffer = point_geom.buffer(radius)
        
        # 반경 내 경계선 점들 찾기
        nearby_boundary_points = []
        for bp in boundary_points:
            if point_geom.distance(Point(bp)) <= radius:
                nearby_boundary_points.append(bp)
        
        if len(nearby_boundary_points) < 3:
            curve_analysis[point['id']] = {
                'curve_score': 0,
                'curve_density': 0,
                'curve_complexity': 0,
                'is_critical_curve': False
            }
            continue
        
        # 커브 점수 계산
        curve_score = 0
        curve_density = len(nearby_boundary_points) / (np.pi * radius**2)
        
        # 커브 복잡도 계산 (연속된 점들의 각도 변화)
        if len(nearby_boundary_points) >= 3:
            angles = []
            for i in range(len(nearby_boundary_points) - 2):
                p1 = np.array(nearby_boundary_points[i])
                p2 = np.array(nearby_boundary_points[i+1])
                p3 = np.array(nearby_boundary_points[i+2])
                
                v1 = p2 - p1
                v2 = p3 - p2
                
                if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    cos_angle = np.clip(cos_angle, -1, 1)
                    angle = np.arccos(cos_angle)
                    angles.append(angle)
            
            if angles:
                avg_angle = np.mean(angles)
                max_angle = np.max(angles)
                curve_complexity = avg_angle + max_angle * 0.5
                curve_score = curve_density * curve_complexity * 100
            else:
                curve_complexity = 0
        else:
            curve_complexity = 0
        
        # 임계 커브 여부 판단 (30도 이상 변화가 있으면 임계 커브)
        is_critical_curve = False
        if len(nearby_boundary_points) >= 3:
            for i in range(len(nearby_boundary_points) - 2):
                p1 = np.array(nearby_boundary_points[i])
                p2 = np.array(nearby_boundary_points[i+1])
                p3 = np.array(nearby_boundary_points[i+2])
                
                v1 = p2 - p1
                v2 = p3 - p2
                
                if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    cos_angle = np.clip(cos_angle, -1, 1)
                    angle = np.arccos(cos_angle)
                    
                    if angle > np.pi/6:  # 30도 이상
                        is_critical_curve = True
                        break
        
        curve_analysis[point['id']] = {
            'curve_score': curve_score,
            'curve_density': curve_density,
            'curve_complexity': curve_complexity,
            'is_critical_curve': is_critical_curve,
            'nearby_points_count': len(nearby_boundary_points)
        }
    
    return curve_analysis, boundary_points

def analyze_traffic_flow_detailed(road_gdf, points_gdf, radius=40):
    """교통 흐름 상세 분석"""
    print("🚗 교통 흐름 상세 분석 중...")
    
    # 도로 폴리곤 통합
    road_union = unary_union(road_gdf.geometry)
    
    flow_analysis = {}
    
    for idx, point in points_gdf.iterrows():
        point_geom = Point(point.geometry.x, point.geometry.y)
        point_buffer = point_geom.buffer(radius)
        
        # 반경 내 도로 영역 찾기
        nearby_roads = []
        if hasattr(road_union, 'geoms'):
            for polygon in road_union.geoms:
                if polygon.intersects(point_buffer):
                    intersection = polygon.intersection(point_buffer)
                    nearby_roads.append(intersection)
        else:
            if road_union.intersects(point_buffer):
                intersection = road_union.intersection(point_buffer)
                nearby_roads.append(intersection)
        
        if not nearby_roads:
            flow_analysis[point['id']] = {
                'flow_score': 0,
                'road_area': 0,
                'road_length': 0,
                'is_major_route': False
            }
            continue
        
        # 교통 흐름 점수 계산
        total_area = sum(road.area for road in nearby_roads)
        total_length = sum(road.length for road in nearby_roads)
        
        # 주요 경로 여부 판단 (면적과 길이 기준)
        is_major_route = total_area > 1000 or total_length > 100
        
        flow_score = total_area * 0.1 + total_length * 10
        
        flow_analysis[point['id']] = {
            'flow_score': flow_score,
            'road_area': total_area,
            'road_length': total_length,
            'is_major_route': is_major_route
        }
    
    return flow_analysis

def analyze_point_importance_comprehensive():
    """종합적인 점 중요도 분석"""
    print("🎯 종합적인 점 중요도 분석 시작...")
    
    points_gdf, road_gdf = load_data()
    
    # 커브 분석
    curve_analysis, boundary_points = analyze_curve_detection(road_gdf, points_gdf)
    
    # 교통 흐름 분석
    flow_analysis = analyze_traffic_flow_detailed(road_gdf, points_gdf)
    
    # 결과 출력
    print(f"\n📊 각 점의 상세 분석 결과:")
    
    for idx, point in points_gdf.iterrows():
        point_id = point['id']
        curve_data = curve_analysis[point_id]
        flow_data = flow_analysis[point_id]
        
        print(f"\n🎯 {point_id}번 점 상세 분석:")
        print(f"  📍 위치: ({point.geometry.x:.1f}, {point.geometry.y:.1f})")
        print(f"  🔄 커브 분석:")
        print(f"    - 커브 점수: {curve_data['curve_score']:.1f}")
        print(f"    - 커브 밀도: {curve_data['curve_density']:.3f}")
        print(f"    - 커브 복잡도: {curve_data['curve_complexity']:.3f}")
        print(f"    - 임계 커브 여부: {'✅ 예' if curve_data['is_critical_curve'] else '❌ 아니오'}")
        print(f"    - 근처 경계선 점: {curve_data['nearby_points_count']}개")
        print(f"  🚗 교통 흐름 분석:")
        print(f"    - 교통 흐름 점수: {flow_data['flow_score']:.1f}")
        print(f"    - 도로 면적: {flow_data['road_area']:.1f}")
        print(f"    - 도로 길이: {flow_data['road_length']:.1f}")
        print(f"    - 주요 경로 여부: {'✅ 예' if flow_data['is_major_route'] else '❌ 아니오'}")
        
        # 특별한 점 강조
        if curve_data['is_critical_curve']:
            print(f"  ⚠️  주의: 이 점은 임계 커브에 위치하여 교통상 필수적!")
        if flow_data['is_major_route']:
            print(f"  ⚠️  주의: 이 점은 주요 교통 경로에 위치!")
    
    # 시각화
    visualize_curve_analysis(points_gdf, road_gdf, curve_analysis, flow_analysis, boundary_points)
    
    return curve_analysis, flow_analysis

def visualize_curve_analysis(points_gdf, road_gdf, curve_analysis, flow_analysis, boundary_points):
    """커브 분석 결과 시각화"""
    print(f"\n📊 결과 시각화...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 전체 도로망과 점들
    road_gdf.plot(ax=ax1, alpha=0.3, color='gray')
    points_gdf.plot(ax=ax1, color='blue', markersize=30, alpha=0.7)
    
    # 점 번호 표시
    for idx, point in points_gdf.iterrows():
        ax1.annotate(f"{point['id']}", 
                    (point.geometry.x, point.geometry.y),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    # 경계선 점들 표시
    boundary_x = [bp[0] for bp in boundary_points]
    boundary_y = [bp[1] for bp in boundary_points]
    ax1.scatter(boundary_x, boundary_y, c='red', s=5, alpha=0.5, label='경계선 점')
    
    ax1.set_title('전체 도로망과 분석 점들')
    ax1.set_aspect('equal')
    ax1.legend()
    
    # 2. 커브 점수 비교
    ids = [point['id'] for _, point in points_gdf.iterrows()]
    curve_scores = [curve_analysis[id]['curve_score'] for id in ids]
    
    bars = ax2.bar(ids, curve_scores, color='skyblue', alpha=0.7)
    
    # 임계 커브 점들 강조
    for i, point_id in enumerate(ids):
        if curve_analysis[point_id]['is_critical_curve']:
            bars[i].set_color('red')
            bars[i].set_alpha(0.8)
    
    ax2.set_xlabel('점 번호')
    ax2.set_ylabel('커브 점수')
    ax2.set_title('점별 커브 점수 (빨간색 = 임계 커브)')
    
    # 3. 교통 흐름 점수 비교
    flow_scores = [flow_analysis[id]['flow_score'] for id in ids]
    
    bars = ax3.bar(ids, flow_scores, color='lightgreen', alpha=0.7)
    
    # 주요 경로 점들 강조
    for i, point_id in enumerate(ids):
        if flow_analysis[point_id]['is_major_route']:
            bars[i].set_color('orange')
            bars[i].set_alpha(0.8)
    
    ax3.set_xlabel('점 번호')
    ax3.set_ylabel('교통 흐름 점수')
    ax3.set_title('점별 교통 흐름 점수 (주황색 = 주요 경로)')
    
    # 4. 종합 중요도 (커브 + 교통 흐름)
    combined_scores = []
    for point_id in ids:
        curve_score = curve_analysis[point_id]['curve_score']
        flow_score = flow_analysis[point_id]['flow_score']
        combined = curve_score * 0.4 + flow_score * 0.6
        combined_scores.append(combined)
    
    bars = ax4.bar(ids, combined_scores, color='purple', alpha=0.7)
    
    # 특별한 점들 강조
    for i, point_id in enumerate(ids):
        is_critical = curve_analysis[point_id]['is_critical_curve']
        is_major = flow_analysis[point_id]['is_major_route']
        
        if is_critical and is_major:
            bars[i].set_color('red')
            bars[i].set_alpha(0.9)
        elif is_critical or is_major:
            bars[i].set_color('orange')
            bars[i].set_alpha(0.8)
    
    ax4.set_xlabel('점 번호')
    ax4.set_ylabel('종합 중요도')
    ax4.set_title('점별 종합 중요도 (커브 + 교통 흐름)')
    
    plt.tight_layout()
    plt.savefig('curve_importance_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✅ 시각화 결과 저장: curve_importance_analysis.png")
    
    plt.show()

if __name__ == "__main__":
    print("🚀 커브 중요성 상세 분석 시작!")
    print("=" * 50)
    
    curve_analysis, flow_analysis = analyze_point_importance_comprehensive()
    
    print("\n" + "=" * 50)
    print("✅ 분석 완료!")
    print("\n💡 결론:")
    print("1. 임계 커브에 위치한 점들은 교통상 필수적")
    print("2. 주요 교통 경로에 위치한 점들도 중요")
    print("3. 7번 점의 특별한 중요성을 구체적으로 파악 가능") 