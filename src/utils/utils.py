import numpy as np
from pathlib import Path
from datetime import datetime
import json

def save_session(file_path, labels, skeleton, metadata=None, user_actions=None, polygon_info=None, dqn_samples=None):
    session_dir = Path("sessions")
    session_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = Path(file_path).stem
    session_path = session_dir / f"{filename}_{timestamp}.json"
    session_data = {
        'file_path': str(file_path),
        'timestamp': datetime.now().isoformat(),
        'labels': labels,
        'skeleton': skeleton.tolist() if hasattr(skeleton, 'tolist') else skeleton,
        'metadata': metadata or {},
        'user_actions': user_actions or [],
        'polygon_info': polygon_info or {},
        'samples': dqn_samples or []
    }
    with open(session_path, 'w', encoding='utf-8') as f:
        json.dump(session_data, f, indent=2, ensure_ascii=False)
    return str(session_path)

def load_session(session_path):
    try:
        with open(session_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"세션 로드 오류: {e}")
        return None

def list_sessions():
    session_dir = Path("sessions")
    if not session_dir.exists():
        return []
    sessions = []
    for session_file in sorted(session_dir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            with open(session_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            sessions.append({
                'path': str(session_file),
                'file_path': data.get('file_path', ''),
                'timestamp': data.get('timestamp', ''),
                'label_counts': {
                    'intersection': len(data.get('labels', {}).get('intersection', [])),
                    'curve': len(data.get('labels', {}).get('curve', [])),
                    'endpoint': len(data.get('labels', {}).get('endpoint', []))
                }
            })
        except:
            continue
    return sessions

def extract_point_features(point, skeleton, idx, heuristic_results=None):
    features = []
    x, y = float(point[0]), float(point[1])
    features.extend([x, y])  # 0, 1

    # prev_dist, prev_angle
    if idx > 0 and len(skeleton[idx-1]) >= 2:
        prev_x, prev_y = skeleton[idx-1][0], skeleton[idx-1][1]
        prev_dist = np.sqrt((x - prev_x)**2 + (y - prev_y)**2)
        prev_angle = np.arctan2(y - prev_y, x - prev_x)
    else:
        prev_dist, prev_angle = 0.0, 0.0
    features.extend([prev_dist, prev_angle])  # 2, 3

    # next_dist, next_angle
    if idx < len(skeleton) - 1 and len(skeleton[idx+1]) >= 2:
        next_x, next_y = skeleton[idx+1][0], skeleton[idx+1][1]
        next_dist = np.sqrt((next_x - x)**2 + (next_y - y)**2)
        next_angle = np.arctan2(next_y - y, next_x - x)
    else:
        next_dist, next_angle = 0.0, 0.0
    features.extend([next_dist, next_angle])  # 4, 5

    # density (반경 50 내 skeleton 포인트 밀도)
    density = sum(
        1 for p in skeleton if len(p) >= 2 and np.sqrt((x-p[0])**2 + (y-p[1])**2) <= 50
    )
    features.append(density / len(skeleton) if skeleton else 0.0)  # 6

    # curvature (양옆각도차)
    if idx > 0 and idx < len(skeleton) - 1 and len(skeleton[idx-1]) >= 2 and len(skeleton[idx+1]) >= 2:
        p1 = np.array(skeleton[idx-1][:2])
        p2 = np.array([x, y])
        p3 = np.array(skeleton[idx+1][:2])
        v1 = p2 - p1
        v2 = p3 - p2
        angle1 = np.arctan2(v1[1], v1[0])
        angle2 = np.arctan2(v2[1], v2[0])
        curvature = abs(angle2 - angle1)
        if curvature > np.pi:
            curvature = 2 * np.pi - curvature
    else:
        curvature = 0.0
    features.append(curvature)  # 7

    # === Heuristic 특성 ===
    # 8~11: heuristic class onehot (intersection, curve, endpoint, 기타)
    heuristic_class = 0
    if heuristic_results:
        threshold = 5.0
        for i, cat in enumerate(['intersection', 'curve', 'endpoint']):
            for px, py in heuristic_results.get(cat, []):
                if np.sqrt((x - px)**2 + (y - py)**2) < threshold:
                    heuristic_class = i + 1
                    break
    onehot = [0, 0, 0, 0]
    onehot[heuristic_class] = 1
    features.extend(onehot)

    # 12: heuristic_confidence
    if heuristic_class == 0:
        heuristic_confidence = 0.0
    else:
        heuristic_confidence = [0.0, 0.9, 0.7, 0.8][heuristic_class]  # intersection, curve, endpoint
    features.append(heuristic_confidence)

    # 13~15: nearby_counts(교차점, 커브, 끝점, 반경 50)
    for cat in ['intersection', 'curve', 'endpoint']:
        count = 0
        if heuristic_results:
            for px, py in heuristic_results.get(cat, []):
                if np.sqrt((x - px)**2 + (y - py)**2) <= 50:
                    count += 1
        features.append(float(count))

    # 16: nearest_dist (가장 가까운 heuristic point)
    min_dist = 1000.0
    if heuristic_results:
        for cat in ['intersection', 'curve', 'endpoint']:
            for px, py in heuristic_results.get(cat, []):
                dist = np.sqrt((x - px)**2 + (y - py)**2)
                min_dist = min(min_dist, dist)
    features.append(min_dist)

    # 17: density_2 (heuristic point 밀도: 총 nearby_count / 반경 면적 * 1000)
    total_count = 0
    if heuristic_results:
        for cat in ['intersection', 'curve', 'endpoint']:
            for px, py in heuristic_results.get(cat, []):
                if np.sqrt((x - px)**2 + (y - py)**2) <= 50:
                    total_count += 1
    features.append(total_count / (np.pi * 50 * 50) * 1000 if total_count > 0 else 0)

    # 18~19: (패딩)
    while len(features) < 20:
        features.append(0.0)

    return np.array(features[:20])

def prepare_training_data(sessions):
    """
    3-액션 시스템 학습 데이터 준비
    
    액션:
    - 0: keep (유지) - 교차점, 끝점, 일반 점
    - 1: add_curve (커브 추가)  
    - 2: delete (삭제)
    
    역할 분담:
    - 휴리스틱: 교차점, 끝점 검출
    - AI: 커브 추가, 삭제 판단
    """
    all_features = []
    all_labels = []
    
    for session in sessions:
        skeleton = session.get('skeleton', [])
        if not skeleton:
            continue
            
        labels = session.get('labels', {})
        deleted_points = session.get('deleted_points', {})
        heuristic_results = session.get('metadata', {}).get('heuristic_results', {})
        
        # 사용자가 추가한 커브 점들
        curve_points = set()
        for px, py in labels.get('curve', []):
            curve_points.add((float(px), float(py)))
        
        # 사용자가 삭제한 점들 (모든 카테고리 포함)
        delete_points = set()
        for category in ['intersection', 'curve', 'endpoint']:
            for px, py in deleted_points.get(category, []):
                delete_points.add((float(px), float(py)))
        
        for i, point in enumerate(skeleton):
            if len(point) < 2:
                continue
                
            features = extract_point_features(point, skeleton, i, heuristic_results)
            x, y = float(point[0]), float(point[1])
            
            # 라벨 결정 (AI가 판단해야 할 것만)
            label = 0  # 기본: keep (유지)
            
            # 1순위: 삭제된 점 확인
            min_delete_dist = min([np.sqrt((x - dx)**2 + (y - dy)**2) 
                                 for dx, dy in delete_points], default=float('inf'))
            if min_delete_dist < 5:
                label = 2  # delete
            else:
                # 2순위: 커브 점 확인
                min_curve_dist = min([np.sqrt((x - cx)**2 + (y - cy)**2) 
                                    for cx, cy in curve_points], default=float('inf'))
                if min_curve_dist < 5:
                    label = 1  # add_curve
                # else: label = 0 (keep) - 교차점, 끝점, 일반 점은 모두 유지
            
            all_features.append(features)
            all_labels.append(label)
    
    return np.array(all_features), np.array(all_labels)


def get_polygon_session_name(base_filename, polygon_index, total_polygons):
    """멀티폴리곤 세션 파일명 생성"""
    if total_polygons > 1:
        return f"{base_filename}-{polygon_index}"
    return base_filename

def load_polygon_sessions(base_session_path):
    """멀티폴리곤 관련 모든 세션 로드"""
    sessions = []
    base_path = Path(base_session_path)
    base_stem = base_path.stem.split('_')[0]  # timestamp 제거
    
    session_dir = base_path.parent
    for session_file in session_dir.glob(f"{base_stem}*.json"):
        session_data = load_session(session_file)
        if session_data:
            sessions.append(session_data)
    
    return sessions

