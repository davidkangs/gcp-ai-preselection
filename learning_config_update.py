"""
업데이트된 학습 설정
교차점과 커브는 휴리스틱, 끝점만 AI 학습
"""

# 새로운 학습 설정
LEARNING_CONFIG = {
    'classes': ['normal', 'endpoint'],  # 2개 클래스만
    'action_size': 2,
    'use_heuristic': {
        'intersection': True,  # 휴리스틱 사용
        'curve': True,        # 휴리스틱 사용
        'endpoint': False     # AI 학습
    }
}

def prepare_training_data_endpoint_only(sessions):
    """끝점만 학습하는 데이터 준비"""
    all_features = []
    all_labels = []
    
    for session in sessions:
        skeleton = np.array(session['skeleton'])
        labels = session['labels']
        
        for i, point in enumerate(skeleton):
            # 특징 추출 (기존과 동일)
            features = extract_point_features(point, skeleton)
            
            # 라벨: 0=일반, 1=끝점
            label = 0
            
            # 끝점 확인
            for endpoint in labels.get('endpoint', []):
                if np.linalg.norm(point - np.array(endpoint)) < 5:
                    label = 1
                    break
            
            all_features.append(features)
            all_labels.append(label)
    
    return np.array(all_features), np.array(all_labels)
