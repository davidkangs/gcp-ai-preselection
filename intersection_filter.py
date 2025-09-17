"""
교차점 필터링 학습
사용자가 삭제하는 교차점 패턴 학습
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

class IntersectionFilter:
    """교차점 필터 학습 모델"""
    
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=50, max_depth=10)
        self.is_trained = False
    
    def extract_intersection_features(self, intersection, all_intersections, skeleton):
        """교차점 특징 추출"""
        features = []
        
        # 1. 교차점 크기 (연결된 스켈레톤 포인트 수)
        connected_count = 0
        for skel_point in skeleton:
            if np.linalg.norm(intersection - skel_point) < 20:
                connected_count += 1
        features.append(connected_count)
        
        # 2. 가장 가까운 다른 교차점과의 거리
        min_distance = float('inf')
        for other in all_intersections:
            if not np.array_equal(intersection, other):
                dist = np.linalg.norm(intersection - other)
                if dist < min_distance:
                    min_distance = dist
        features.append(min_distance if min_distance != float('inf') else 1000)
        
        # 3. 주변 교차점 밀도 (반경 100m 내)
        nearby_count = 0
        for other in all_intersections:
            if np.linalg.norm(intersection - other) < 100:
                nearby_count += 1
        features.append(nearby_count)
        
        # 4. 교차점의 각도 다양성
        angles = []
        for i in range(len(skeleton) - 1):
            if np.linalg.norm(skeleton[i] - intersection) < 30:
                # 연결된 도로의 각도
                next_point = skeleton[i + 1]
                angle = np.arctan2(next_point[1] - intersection[1], 
                                 next_point[0] - intersection[0])
                angles.append(angle)
        
        if len(angles) >= 2:
            # 각도의 표준편차 (다양성)
            features.append(np.std(angles))
        else:
            features.append(0)
        
        return np.array(features)
    
    def train(self, kept_intersections, removed_intersections, all_intersections, skeleton):
        """사용자 패턴 학습"""
        X = []
        y = []
        
        # 유지된 교차점
        for intersection in kept_intersections:
            features = self.extract_intersection_features(
                intersection, all_intersections, skeleton
            )
            X.append(features)
            y.append(1)  # 유지
        
        # 제거된 교차점
        for intersection in removed_intersections:
            features = self.extract_intersection_features(
                intersection, all_intersections, skeleton
            )
            X.append(features)
            y.append(0)  # 제거
        
        if len(X) > 10:  # 충분한 데이터가 있을 때만 학습
            self.model.fit(X, y)
            self.is_trained = True
            print(f"교차점 필터 학습 완료: {len(X)}개 샘플")
    
    def predict(self, intersections, skeleton):
        """교차점 필터링"""
        if not self.is_trained:
            return intersections  # 학습 전에는 모두 반환
        
        filtered = []
        for intersection in intersections:
            features = self.extract_intersection_features(
                intersection, intersections, skeleton
            ).reshape(1, -1)
            
            if self.model.predict(features)[0] == 1:
                filtered.append(intersection)
        
        return filtered
    
    def save(self, path):
        """모델 저장"""
        if self.is_trained:
            joblib.dump(self.model, path)
    
    def load(self, path):
        """모델 로드"""
        try:
            self.model = joblib.load(path)
            self.is_trained = True
        except:
            pass
