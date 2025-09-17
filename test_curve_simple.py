#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
간단한 곡률 테스트
"""

import numpy as np
from shapely.geometry import Point, LineString
import geopandas as gpd

def test_curvature_calculation():
    # 간단한 곡선 생성
    t = np.linspace(0, 2*np.pi, 100)
    x = np.cos(t) * 100  # 반지름 100m 원
    y = np.sin(t) * 100
    
    # LineString 생성
    coords = list(zip(x, y))
    line = LineString(coords)
    
    # 버퍼 생성
    polygon = line.buffer(6.0)
    boundary = polygon.exterior
    
    print(f"원형 도로:")
    print(f"  - 길이: {boundary.length:.1f}m")
    print(f"  - 타입: {boundary.geom_type}")
    
    # 곡률 계산
    sample_distance = 15.0
    num_samples = max(5, int(boundary.length / sample_distance))
    print(f"  - 샘플 수: {num_samples}")
    
    curvatures = []
    for j in range(num_samples):
        distance = (j * sample_distance) % boundary.length
        curvature = calculate_curvature_at_distance(boundary, distance, sample_distance)
        curvatures.append(curvature)
    
    print(f"  - 곡률 범위: {min(curvatures):.3f} ~ {max(curvatures):.3f}")
    print(f"  - 평균 곡률: {np.mean(curvatures):.3f}")
    
    # 다양한 임계값 테스트
    for threshold in [0.01, 0.05, 0.1, 0.15, 0.2, 0.3]:
        count = sum(1 for c in curvatures if c > threshold)
        print(f"  - 임계값 {threshold}: {count}개 검출")

def calculate_curvature_at_distance(boundary, distance, window=20.0):
    """특정 거리에서의 곡률 계산"""
    try:
        # 앞뒤 점들 구하기
        d1 = max(0, distance - window)
        d2 = min(boundary.length, distance + window)
        
        if d2 - d1 < window * 0.5:
            return 0.0
        
        p1 = boundary.interpolate(d1)
        p2 = boundary.interpolate(distance)
        p3 = boundary.interpolate(d2)
        
        # 벡터 계산
        v1 = np.array([p2.x - p1.x, p2.y - p1.y])
        v2 = np.array([p3.x - p2.x, p3.y - p2.y])
        
        # 각도 변화 계산
        len1 = np.linalg.norm(v1)
        len2 = np.linalg.norm(v2)
        
        if len1 > 0 and len2 > 0:
            v1_norm = v1 / len1
            v2_norm = v2 / len2
            
            cos_angle = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
            angle = np.arccos(cos_angle)
            
            return angle
        
        return 0.0
        
    except:
        return 0.0

if __name__ == '__main__':
    test_curvature_calculation() 