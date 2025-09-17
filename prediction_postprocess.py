"""
예측 후처리 함수들
중복 제거 및 특징점 필터링
"""

import numpy as np
from scipy.spatial.distance import cdist
from collections import defaultdict

def postprocess_predictions(skeleton, predictions, confidence_threshold=0.7, min_distance=20):
    """
    예측 결과 후처리
    
    Args:
        skeleton: 스켈레톤 포인트들
        predictions: 각 포인트의 예측 클래스 (0: 일반, 1: 교차점, 2: 커브, 3: 끝점)
        confidence_threshold: 신뢰도 임계값
        min_distance: 같은 클래스 내 최소 거리
    
    Returns:
        필터링된 특징점들
    """
    
    # 결과 저장
    filtered_points = {
        'intersection': [],
        'curve': [],
        'endpoint': []
    }
    
    # 클래스별 포인트 수집
    class_points = defaultdict(list)
    for i, pred in enumerate(predictions):
        if pred > 0:  # 0은 일반 포인트
            class_points[pred].append(i)
    
    # 각 클래스별로 처리
    class_names = {1: 'intersection', 2: 'curve', 3: 'endpoint'}
    
    for class_id, indices in class_points.items():
        if len(indices) == 0:
            continue
        
        # 해당 클래스의 모든 포인트
        points = np.array([skeleton[i] for i in indices])
        
        if class_id == 1:  # 교차점
            # 교차점은 서로 가까운 것들을 그룹화하고 중심점만 선택
            filtered = filter_intersections(points, min_distance=30)
        
        elif class_id == 2:  # 커브
            # 커브는 연속된 커브 구간에서 가장 곡률이 큰 점만 선택
            filtered = filter_curves(skeleton, indices, min_distance=50)
        
        elif class_id == 3:  # 끝점
            # 끝점은 서로 떨어진 것만 유지
            filtered = filter_endpoints(points, min_distance=40)
        
        else:
            filtered = points
        
        # 결과에 추가
        class_name = class_names.get(class_id, 'unknown')
        filtered_points[class_name] = [tuple(p) for p in filtered]
    
    return filtered_points

def filter_intersections(points, min_distance=30):
    """교차점 필터링 - 가까운 점들을 그룹화하고 중심점 선택"""
    if len(points) == 0:
        return []
    
    # 거리 행렬 계산
    distances = cdist(points, points)
    
    # 이미 처리된 점 추적
    processed = set()
    filtered = []
    
    for i in range(len(points)):
        if i in processed:
            continue
        
        # 가까운 점들 찾기
        nearby = np.where(distances[i] < min_distance)[0]
        
        # 그룹의 중심점 계산
        group_points = points[nearby]
        center = np.mean(group_points, axis=0)
        filtered.append(center)
        
        # 처리됨으로 표시
        processed.update(nearby)
    
    return np.array(filtered)

def filter_curves(skeleton, curve_indices, min_distance=50):
    """커브 필터링 - 연속된 커브 구간에서 대표점 선택"""
    if len(curve_indices) == 0:
        return []
    
    # 연속된 구간 찾기
    curve_indices = sorted(curve_indices)
    segments = []
    current_segment = [curve_indices[0]]
    
    for i in range(1, len(curve_indices)):
        if curve_indices[i] - curve_indices[i-1] <= 3:  # 연속된 것으로 간주
            current_segment.append(curve_indices[i])
        else:
            segments.append(current_segment)
            current_segment = [curve_indices[i]]
    
    if current_segment:
        segments.append(current_segment)
    
    # 각 구간에서 대표점 선택 (곡률이 가장 큰 점)
    filtered = []
    for segment in segments:
        if len(segment) < 5:  # 너무 짧은 구간은 무시
            continue
        
        # 구간의 중간점 선택
        mid_idx = segment[len(segment)//2]
        filtered.append(skeleton[mid_idx])
    
    return np.array(filtered)

def filter_endpoints(points, min_distance=40):
    """끝점 필터링 - 서로 떨어진 점만 유지"""
    if len(points) == 0:
        return []
    
    # 거리 기반 필터링
    filtered = []
    for i, point in enumerate(points):
        # 이미 선택된 점들과의 거리 확인
        if len(filtered) == 0:
            filtered.append(point)
        else:
            distances = cdist([point], filtered)[0]
            if np.min(distances) > min_distance:
                filtered.append(point)
    
    return np.array(filtered)

def calculate_curvature(skeleton, index, window=5):
    """특정 점에서의 곡률 계산"""
    start = max(0, index - window)
    end = min(len(skeleton), index + window + 1)
    
    if end - start < 3:
        return 0
    
    points = skeleton[start:end]
    
    # 1차, 2차 미분 근사
    if len(points) >= 3:
        # 중심 차분으로 미분 계산
        dx = np.gradient(points[:, 0])
        dy = np.gradient(points[:, 1])
        
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        
        # 곡률 계산
        numerator = abs(dx[len(dx)//2] * ddy[len(ddy)//2] - dy[len(dy)//2] * ddx[len(ddx)//2])
        denominator = (dx[len(dx)//2]**2 + dy[len(dy)//2]**2)**1.5
        
        if denominator > 0:
            return numerator / denominator
    
    return 0
