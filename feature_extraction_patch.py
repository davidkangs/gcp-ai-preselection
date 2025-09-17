"""
특징 추출 함수 패치
numpy array 처리 보장
"""

import numpy as np
from src.utils import extract_point_features as original_extract_point_features

def extract_point_features_safe(point, window_points, skeleton):
    """안전한 특징 추출 - numpy array 변환 보장"""
    
    # 모든 입력을 numpy array로 변환
    point = np.array(point)
    window_points = np.array(window_points)
    skeleton = np.array(skeleton)
    
    # 원본 함수 호출
    return original_extract_point_features(point, window_points, skeleton)

# 전역 함수 교체
import src.utils
src.utils.extract_point_features = extract_point_features_safe

print("✓ 특징 추출 함수 패치 적용됨")
