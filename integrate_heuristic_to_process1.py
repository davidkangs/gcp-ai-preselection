"""
Process 1에 통합 휴리스틱 적용
교차점과 커브를 모두 휴리스틱으로 검출
"""

import sys
from pathlib import Path
import shutil
from datetime import datetime

def integrate_to_process1():
    """process1_labeling_tool.py에 통합 휴리스틱 추가"""
    
    process1_path = Path("process1_labeling_tool.py")
    
    if not process1_path.exists():
        print("❌ process1_labeling_tool.py 파일을 찾을 수 없습니다.")
        return False
    
    # 백업
    backup_path = f"{process1_path}.backup_heuristic_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy(process1_path, backup_path)
    print(f"✓ 백업 생성: {backup_path}")
    
    # 파일 읽기
    with open(process1_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 1. import 추가
    old_imports = "from src.core.skeleton_extractor import SkeletonExtractor"
    new_imports = """from src.core.skeleton_extractor import SkeletonExtractor
from integrated_heuristic_detector import IntegratedHeuristicDetector"""
    
    content = content.replace(old_imports, new_imports)
    
    # 2. ProcessingThread 수정
    old_init = """def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path
        self.skeleton_extractor = SkeletonExtractor()"""
    
    new_init = """def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path
        self.skeleton_extractor = SkeletonExtractor()
        self.heuristic_detector = IntegratedHeuristicDetector()"""
    
    content = content.replace(old_init, new_init)
    
    # 3. run 메서드 수정
    old_run = """# 3단계: 결과 정리
            self.progress.emit(90, "결과 정리 중...")
            result = {
                'success': True,
                'gdf': gdf,
                'skeleton': skeleton,
                'intersections': intersections,
                'processing_time': processing_time
            }"""
    
    new_run = """# 3단계: 통합 휴리스틱 적용
            self.progress.emit(85, "통합 휴리스틱 검출 중...")
            
            # 교차점과 커브 통합 검출
            detected = self.heuristic_detector.detect_all(gdf, skeleton, intersections)
            
            # 4단계: 결과 정리
            self.progress.emit(90, "결과 정리 중...")
            result = {
                'success': True,
                'gdf': gdf,
                'skeleton': skeleton,
                'intersections': detected['intersection'],  # 필터링된 교차점
                'curves': detected['curve'],  # 휴리스틱 커브
                'processing_time': processing_time
            }"""
    
    content = content.replace(old_run, new_run)
    
    # 4. on_processing_finished 수정
    old_finished = """# 교차점 설정 (휴리스틱)
            self.canvas_widget.canvas.points['intersection'] = [
                (float(x), float(y)) for x, y in result['intersections']
            ]"""
    
    new_finished = """# 교차점 설정 (필터링된 휴리스틱)
            self.canvas_widget.canvas.points['intersection'] = [
                (float(x), float(y)) for x, y in result['intersections']
            ]
            
            # 커브 설정 (휴리스틱)
            self.canvas_widget.canvas.points['curve'] = [
                (float(x), float(y)) for x, y in result.get('curves', [])
            ]"""
    
    content = content.replace(old_finished, new_finished)
    
    # 5. 통계 업데이트
    old_stats = '''stats_text = f"""=== 라벨링 통계 ===
교차점: {len(points.get('intersection', []))}개 (휴리스틱 자동 검출)
커브: {len(points.get('curve', []))}개 (수동 라벨링)
끝점: {len(points.get('endpoint', []))}개 (수동 라벨링)
━━━━━━━━━━━━━━━━━━━━━
전체: {sum(len(v) for v in points.values())}개

※ 교차점은 AI 학습에서 제외됩니다."""'''
    
    new_stats = '''stats_text = f"""=== 라벨링 통계 ===
교차점: {len(points.get('intersection', []))}개 (휴리스틱 - 필터링됨)
커브: {len(points.get('curve', []))}개 (휴리스틱 - 자동 검출)
끝점: {len(points.get('endpoint', []))}개 (수동 라벨링)
━━━━━━━━━━━━━━━━━━━━━
전체: {sum(len(v) for v in points.values())}개

※ 교차점과 커브는 휴리스틱으로 자동 검출됩니다.
※ 5m 이내 중복 제거 및 연결 도로 수 기반 필터링 적용"""'''
    
    content = content.replace(old_stats, new_stats)
    
    # 6. 단축키 설명 수정
    old_shortcuts = '''shortcuts_label = QLabel(
            "• 좌클릭: 커브 추가 (도로가 꺾이는 지점)\\n"
            "• 우클릭: 끝점 추가 (도로의 끝)\\n"
            "• Shift+클릭: 제거\\n"
            "• D: 가장 가까운 점 삭제\\n"
            "• Space: 화면 맞춤\\n"
            "\\n※ 교차점은 자동 검출됩니다"
        )'''
    
    new_shortcuts = '''shortcuts_label = QLabel(
            "• 좌클릭: 커브 수정 (필요시)\\n"
            "• 우클릭: 끝점 추가 (도로의 끝)\\n"
            "• Shift+클릭: 제거\\n"
            "• D: 가장 가까운 점 삭제\\n"
            "• Space: 화면 맞춤\\n"
            "\\n※ 교차점과 커브는 자동 검출됩니다"
        )'''
    
    content = content.replace(old_shortcuts, new_shortcuts)
    
    # 파일 저장
    with open(process1_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✓ process1_labeling_tool.py 수정 완료!")
    return True

def update_learning_process():
    """학습 프로세스 업데이트 - 끝점만 학습"""
    
    update_code = '''"""
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
'''
    
    with open("learning_config_update.py", 'w', encoding='utf-8') as f:
        f.write(update_code)
    
    print("✓ learning_config_update.py 생성 완료!")

def main():
    print("🔧 통합 휴리스틱 시스템 적용")
    print("="*60)
    
    print("\n적용 내용:")
    print("1. 교차점: 휴리스틱 (연결 도로 수 기반 필터링)")
    print("2. 커브: 휴리스틱 (각도 변화 + DBSCAN)")
    print("3. 끝점: AI 학습 (변경 없음)")
    
    print("\n필터링 규칙:")
    print("- 5m 이내 교차점: 더 많은 도로가 연결된 것만 유지")
    print("- 교차점 5m 이내 커브: 제거")
    
    # 1. Process 1 수정
    print("\n1. Process 1 수정 중...")
    if integrate_to_process1():
        print("✓ 성공!")
    else:
        print("❌ 실패")
    
    # 2. 학습 설정 업데이트
    print("\n2. 학습 설정 업데이트...")
    update_learning_process()
    
    print("\n" + "="*60)
    print("✅ 통합 완료!")
    
    print("\n다음 단계:")
    print("1. integrated_heuristic_detector.py를 프로젝트에 추가")
    print("2. Process 1 실행:")
    print("   python run_process1.py")
    print("3. 자동으로 교차점과 커브가 검출됨")
    print("4. 끝점만 수동으로 라벨링")
    
    print("\n💡 이제 AI는 끝점 검출에만 집중할 수 있습니다!")

if __name__ == "__main__":
    main()