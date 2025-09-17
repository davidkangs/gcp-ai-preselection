# fix_run_hybrid.py - run_process3_hybrid.py 전체 교체
#!/usr/bin/env python3
"""
Process3 하이브리드 실행 - 휴리스틱 중심 + AI 보조 (패치 적용)
실행: python run_process3_hybrid.py
"""

import sys
from pathlib import Path
import numpy as np

# 경로 설정
sys.path.append(str(Path(__file__).parent))

# 필수 디렉토리 생성
print("🔧 디렉토리 초기화...")
for folder in ['sessions', 'models', 'results', 'logs']:
    Path(folder).mkdir(exist_ok=True)
    print(f"  ✅ {folder}/")

print("\n🎯 휴리스틱 중심 + AI 보조 모드 시작...")
print("📊 특징:")
print("  🔹 휴리스틱이 메인 검출 담당")
print("  🔹 AI는 신뢰도 0.8+ 만 보조")
print("  🔹 안정적이고 실용적인 접근")
print("  🔹 실시간 신뢰도 모니터링")

# 패치 먼저 적용
def apply_hybrid_patch():
    try:
        from heuristic_centered_inference import HybridPredictionWorker
        
        def new_merge_results(self, skeleton_array, heuristic_results, ai_filtered):
            """휴리스틱 100% 보존 + AI 보조만"""
            
            # 1. 휴리스틱 결과 100% 보존
            final_results = {
                'intersection': list(heuristic_results['intersection']),
                'curve': list(heuristic_results['curve']),                
                'endpoint': list(heuristic_results['endpoint']),
                'delete': []
            }
            
            print(f"\n🔧 휴리스틱 100% 보존:")
            print(f"  교차점: {len(final_results['intersection'])}개")
            print(f"  커브: {len(final_results['curve'])}개") 
            print(f"  끝점: {len(final_results['endpoint'])}개")
            
            # 2. AI 보조 - 휴리스틱 30m 밖에서만
            heuristic_exclusion_radius = 30.0
            all_heuristic_points = []
            all_heuristic_points.extend(heuristic_results['intersection'])
            all_heuristic_points.extend(heuristic_results['curve'])
            all_heuristic_points.extend(heuristic_results['endpoint'])
            
            ai_added = {'intersection': 0, 'curve': 0, 'endpoint': 0, 'delete': 0}
            
            for i, (idx, pred, conf) in enumerate(zip(
                ai_filtered['indices'], 
                ai_filtered['predictions'], 
                ai_filtered['confidences']
            )):
                if idx >= len(skeleton_array):
                    continue
                    
                point = tuple(skeleton_array[idx])
                
                # 휴리스틱 범위 체크
                too_close_to_heuristic = False
                for h_point in all_heuristic_points:
                    dist = np.linalg.norm(np.array(point) - np.array(h_point))
                    if dist < heuristic_exclusion_radius:
                        too_close_to_heuristic = True
                        break
                
                if too_close_to_heuristic:
                    continue
                
                # 신뢰도 0.9+ 고품질만
                if conf < 0.9:
                    continue
                
                # AI 보조 추가
                if pred == 1:
                    final_results['intersection'].append(point)
                    ai_added['intersection'] += 1
                elif pred == 2:
                    final_results['curve'].append(point)
                    ai_added['curve'] += 1
                elif pred == 3:
                    final_results['endpoint'].append(point)
                    ai_added['endpoint'] += 1
                elif pred == 4:
                    final_results['delete'].append(point)
                    ai_added['delete'] += 1
            
            print(f"\n🤖 AI 보조 추가:")
            print(f"  교차점: +{ai_added['intersection']}개")
            print(f"  커브: +{ai_added['curve']}개")
            print(f"  끝점: +{ai_added['endpoint']}개")
            
            return final_results
        
        HybridPredictionWorker.merge_results = new_merge_results
        print("\n✅ 휴리스틱 우선 패치 적용됨")
        
    except ImportError as e:
        print(f"❌ 패치 적용 실패: {e}")

# 패치 적용 후 실행
try:
    apply_hybrid_patch()
    from heuristic_centered_inference import main
    main()
except ImportError as e:
    print(f"❌ 모듈 import 오류: {e}")
except Exception as e:
    print(f"❌ 실행 오류: {e}")
    import traceback
    traceback.print_exc()