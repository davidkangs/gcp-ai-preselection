"""
기존 세션 JSON → 새로운 DQN 학습 데이터 변환기
38개의 기존 라벨링 세션을 DQN 학습용으로 변환
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.learning import DQNDataCollector

def analyze_existing_session(session_path):
    """기존 세션 JSON 분석"""
    try:
        with open(session_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"📁 {Path(session_path).name}")
        print(f"   파일: {data.get('file_path', 'N/A')}")
        print(f"   스켈레톤: {len(data.get('skeleton', []))}개")
        
        labels = data.get('labels', {})
        print(f"   라벨: 교차점={len(labels.get('intersection', []))}, "
              f"커브={len(labels.get('curve', []))}, "
              f"끝점={len(labels.get('endpoint', []))}")
        
        deleted_points = data.get('deleted_points', {})
        if deleted_points:
            total_deleted = sum(len(points) for points in deleted_points.values())
            print(f"   삭제: {total_deleted}개")
        
        return data
        
    except Exception as e:
        print(f"❌ 분석 실패 {session_path}: {e}")
        return None

def simulate_user_edits(original_skeleton, final_labels, deleted_points=None):
    """
    최종 라벨링 결과에서 사용자 편집 과정을 역추정
    
    전략:
    1. 휴리스틱이 자동 검출했을 것 같은 포인트들 추정
    2. 사용자 최종 결과와 비교하여 편집 행동 추출
    """
    
    edits = []
    
    # 1. 사용자가 추가한 포인트들 (휴리스틱에 없었을 것)
    for category in ['curve', 'endpoint']:  # intersection은 대부분 휴리스틱이 잘 찾음
        for point in final_labels.get(category, []):
            x, y = point[0], point[1]
            
            # 상태 벡터 생성 (간단 버전)
            state_vector = create_simple_state_vector(x, y, original_skeleton)
            
            edit = {
                'timestamp': datetime.now().isoformat(),
                'action': 'add',
                'category': category,
                'position': [float(x), float(y)],
                'state_vector': state_vector,
                'reward': 1.0,
                'context': {
                    'edit_type': 'user_addition',
                    'confidence': 0.8
                }
            }
            edits.append(edit)
    
    # 2. 사용자가 제거한 포인트들
    if deleted_points:
        for category, points in deleted_points.items():
            for point in points:
                x, y = point[0], point[1]
                
                state_vector = create_simple_state_vector(x, y, original_skeleton)
                
                edit = {
                    'timestamp': datetime.now().isoformat(),
                    'action': 'remove',
                    'category': category,
                    'position': [float(x), float(y)],
                    'state_vector': state_vector,
                    'reward': 1.0,
                    'context': {
                        'edit_type': 'user_removal',
                        'confidence': 0.9  # 삭제는 확실한 의도
                    }
                }
                edits.append(edit)
    
    # 3. 휴리스틱 vs 사용자 결과 차이로 편집 추정
    # 스켈레톤의 일부 포인트를 "휴리스틱이 잘못 검출했을 것"으로 가정
    skeleton_points = original_skeleton if original_skeleton else []
    
    # 스켈레톤 샘플링 (너무 많으면)
    if len(skeleton_points) > 1000:
        step = len(skeleton_points) // 1000
        skeleton_points = skeleton_points[::step]
    
    # 일부 스켈레톤 포인트를 "휴리스틱 오검출"로 가정하고 제거 시뮬레이션
    for i, point in enumerate(skeleton_points):
        if i % 50 == 0:  # 50개마다 하나씩만
            x, y = point[0], point[1]
            
            # 사용자 최종 라벨에 없으면 "제거한 것"으로 간주
            is_in_final = False
            for category_points in final_labels.values():
                for final_point in category_points:
                    if np.linalg.norm(np.array([x, y]) - np.array(final_point)) < 20:
                        is_in_final = True
                        break
                if is_in_final:
                    break
            
            if not is_in_final:
                state_vector = create_simple_state_vector(x, y, original_skeleton)
                
                edit = {
                    'timestamp': datetime.now().isoformat(),
                    'action': 'remove',
                    'category': 'curve',  # 대부분 잘못된 커브 검출
                    'position': [float(x), float(y)],
                    'state_vector': state_vector,
                    'reward': 1.0,
                    'context': {
                        'edit_type': 'heuristic_correction',
                        'confidence': 0.6  # 추정이므로 낮은 신뢰도
                    }
                }
                edits.append(edit)
    
    return edits

def create_simple_state_vector(x, y, skeleton, vector_size=20):
    """간단한 상태 벡터 생성"""
    features = [float(x), float(y)]
    
    # 주변 포인트 분석
    if skeleton and len(skeleton) > 0:
        skeleton_array = np.array(skeleton)
        
        # 가장 가까운 포인트들까지의 거리
        distances = np.linalg.norm(skeleton_array - np.array([x, y]), axis=1)
        nearest_distances = np.sort(distances)[:5]
        features.extend(nearest_distances.tolist())
        
        # 밀도 계산
        radius_counts = [
            np.sum(distances < 25),   # 25m 반경
            np.sum(distances < 50),   # 50m 반경
            np.sum(distances < 100),  # 100m 반경
        ]
        features.extend(radius_counts)
        
        # 기하학적 특성 (간단)
        features.extend([
            x % 100,
            y % 100,
            (x + y) % 50,
            abs(x - y) % 30
        ])
    else:
        # 스켈레톤이 없으면 기본값
        features.extend([0.0] * 13)
    
    # 20차원 맞추기
    while len(features) < vector_size:
        features.append(0.0)
    
    return features[:vector_size]

def convert_session_to_dqn(session_path, output_dir="data/training_samples"):
    """단일 세션을 DQN 형식으로 변환"""
    
    # 기존 세션 로드
    with open(session_path, 'r', encoding='utf-8') as f:
        session_data = json.load(f)
    
    # 데이터 추출
    file_path = session_data.get('file_path', '')
    skeleton = session_data.get('skeleton', [])
    labels = session_data.get('labels', {})
    deleted_points = session_data.get('deleted_points', {})
    
    # 편집 행동 시뮬레이션
    edits = simulate_user_edits(skeleton, labels, deleted_points)
    
    if not edits:
        print(f"   ⚠️ 편집 데이터가 없습니다.")
        return None
    
    # DQN 세션 형식으로 변환
    session_id = Path(session_path).stem  # 기존 파일명 사용
    
    dqn_session = {
        'session_id': session_id,
        'start_time': datetime.now().isoformat(),
        'end_time': datetime.now().isoformat(),
        'file_path': file_path,
        'samples': edits,
        'total_samples': len(edits),
        'conversion_info': {
            'source': str(session_path),
            'original_labels': {k: len(v) for k, v in labels.items()},
            'original_deleted': {k: len(v) for k, v in deleted_points.items()} if deleted_points else {},
            'converted_edits': len(edits)
        }
    }
    
    # 저장
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    output_file = output_path / f"session_{session_id}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dqn_session, f, indent=2, ensure_ascii=False)
    
    print(f"   ✅ 변환 완료: {len(edits)}개 편집 → {output_file}")
    return output_file

def convert_all_sessions():
    """모든 기존 세션을 DQN 형식으로 변환"""
    
    sessions_dir = Path("sessions")
    output_dir = Path("data/training_samples")
    
    print("🔄 기존 세션 → DQN 학습 데이터 변환 시작")
    print("=" * 60)
    
    if not sessions_dir.exists():
        print("❌ sessions 폴더가 없습니다!")
        return
    
    # JSON 파일들 찾기
    session_files = list(sessions_dir.glob("*.json"))
    print(f"📊 발견된 세션 파일: {len(session_files)}개")
    
    if len(session_files) == 0:
        print("❌ 변환할 세션이 없습니다!")
        return
    
    # 변환 실행
    converted_count = 0
    total_edits = 0
    
    print("\n🔄 변환 중...")
    
    for i, session_file in enumerate(session_files, 1):
        print(f"\n[{i}/{len(session_files)}] {session_file.name}")
        
        try:
            # 세션 분석
            session_data = analyze_existing_session(session_file)
            if not session_data:
                continue
            
            # DQN 형식으로 변환
            result_file = convert_session_to_dqn(session_file, output_dir)
            
            if result_file:
                converted_count += 1
                
                # 변환된 편집 수 카운트
                with open(result_file, 'r', encoding='utf-8') as f:
                    converted_data = json.load(f)
                    total_edits += len(converted_data.get('samples', []))
        
        except Exception as e:
            print(f"   ❌ 변환 실패: {e}")
            continue
    
    print("\n" + "=" * 60)
    print("🎉 변환 완료!")
    print(f"✅ 성공: {converted_count}/{len(session_files)}개 세션")
    print(f"📈 총 학습 샘플: {total_edits:,}개")
    print(f"💾 저장 위치: {output_dir}")
    
    if converted_count > 0:
        print(f"\n🚀 다음 단계:")
        print(f"1. python run_process2.py  # 학습 실행")
        print(f"2. 또는 메인 프로그램에서 'DQN 학습' → '모델 학습 시작'")
    
    return converted_count, total_edits

def preview_conversion():
    """변환 미리보기 (1개 파일만)"""
    
    sessions_dir = Path("sessions")
    session_files = list(sessions_dir.glob("*.json"))
    
    if not session_files:
        print("❌ 미리보기할 세션이 없습니다!")
        return
    
    # 첫 번째 파일로 미리보기
    sample_file = session_files[0]
    print(f"🔍 미리보기: {sample_file.name}")
    print("=" * 40)
    
    session_data = analyze_existing_session(sample_file)
    if session_data:
        skeleton = session_data.get('skeleton', [])
        labels = session_data.get('labels', {})
        deleted_points = session_data.get('deleted_points', {})
        
        # 편집 시뮬레이션
        edits = simulate_user_edits(skeleton, labels, deleted_points)
        
        print(f"\n📊 변환 결과 미리보기:")
        print(f"   원본 스켈레톤: {len(skeleton)}개")
        print(f"   시뮬레이션된 편집: {len(edits)}개")
        
        # 편집 유형별 카운트
        add_count = sum(1 for e in edits if e['action'] == 'add')
        remove_count = sum(1 for e in edits if e['action'] == 'remove')
        
        print(f"   - 추가 행동: {add_count}개")
        print(f"   - 제거 행동: {remove_count}개")
        
        # 샘플 편집 표시
        if edits:
            print(f"\n📝 샘플 편집 (처음 3개):")
            for i, edit in enumerate(edits[:3]):
                action = edit['action']
                category = edit['category']
                pos = edit['position']
                conf = edit['context']['confidence']
                print(f"   {i+1}. {action} {category} at ({pos[0]:.1f}, {pos[1]:.1f}) (신뢰도: {conf})")

def main():
    """메인 함수"""
    
    print("🎯 기존 세션 → DQN 학습 데이터 변환기")
    print("38개의 기존 라벨링 세션을 DQN 학습용으로 변환합니다.")
    print()
    
    while True:
        print("📋 옵션을 선택하세요:")
        print("1. 미리보기 (1개 파일만 분석)")
        print("2. 전체 변환 실행")
        print("3. 종료")
        
        choice = input("\n선택 (1-3): ").strip()
        
        if choice == "1":
            preview_conversion()
            
        elif choice == "2":
            convert_all_sessions()
            break
            
        elif choice == "3":
            print("👋 종료합니다.")
            break
            
        else:
            print("❌ 잘못된 선택입니다.")
        
        print("\n" + "-" * 40 + "\n")

if __name__ == "__main__":
    main()