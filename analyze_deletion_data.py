import json
import os
import glob
from collections import defaultdict, Counter

def analyze_session_files():
    """세션 파일들에서 삭제 행동 분석"""
    sessions_dir = 'sessions'
    class_counter = Counter()
    action_counter = Counter()
    total_samples = 0
    remove_files = []
    sample_files = []

    print('🔍 세션 파일 분석 중...')
    
    json_files = glob.glob(os.path.join(sessions_dir, '*.json'))
    print(f'총 {len(json_files)}개 세션 파일 발견')

    for i, file_path in enumerate(json_files):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # samples 키 확인
            if 'samples' in data:
                file_samples = len(data['samples'])
                total_samples += file_samples
                file_removes = 0
                sample_files.append(os.path.basename(file_path))
                
                for sample in data['samples']:
                    action = sample.get('action')
                    category = sample.get('category', 'unknown')
                    
                    if isinstance(action, int):
                        action_counter[action] += 1
                        if action == 4:
                            file_removes += 1
                    elif isinstance(action, str):
                        action_counter[action] += 1
                        if action == 'remove':
                            file_removes += 1
                    
                    if category:
                        class_counter[category] += 1
                
                if file_removes > 0:
                    remove_files.append((os.path.basename(file_path), file_removes, file_samples))
            
            # 프로그레스 표시
            if (i + 1) % 50 == 0:
                print(f'  진행률: {i+1}/{len(json_files)}')
                
        except Exception as e:
            print(f'❌ 파일 읽기 실패: {os.path.basename(file_path)} - {e}')

    print()
    print(f'📊 전체 분석 결과 (총 {len(json_files)}개 파일)')
    print(f'DQN 샘플이 있는 파일: {len(sample_files)}개')
    
    if action_counter:
        print('\n📊 액션 분포:')
        for action, count in sorted(action_counter.items()):
            percentage = (count / total_samples * 100) if total_samples > 0 else 0
            print(f'  {action}: {count:,}개 ({percentage:.1f}%)')

    if class_counter:
        print('\n📊 카테고리 분포:')
        for category, count in sorted(class_counter.items()):
            percentage = (count / total_samples * 100) if total_samples > 0 else 0
            print(f'  {category}: {count:,}개 ({percentage:.1f}%)')

    print('\n🗑️ 삭제 행동이 포함된 파일들:')
    if remove_files:
        for filename, remove_count, total_count in remove_files:
            percentage = (remove_count / total_count * 100) if total_count > 0 else 0
            print(f'  {filename}: 삭제 {remove_count}개/{total_count}개 ({percentage:.1f}%)')
    else:
        print('  삭제 행동이 기록된 파일이 없습니다.')

    print(f'\n총 DQN 샘플 수: {total_samples:,}개')
    
    # 몇 개 샘플 예시 출력
    if sample_files:
        print(f'\n📝 DQN 샘플이 있는 파일 목록 (처음 10개):')
        for filename in sample_files[:10]:
            print(f'  - {filename}')

if __name__ == '__main__':
    analyze_session_files() 